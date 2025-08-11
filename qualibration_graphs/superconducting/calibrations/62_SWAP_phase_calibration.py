from dataclasses import asdict

from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot

from quam_config import Quam
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm.qua import *
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.swap_phase_calibration.parameters import Parameters
from calibration_utils.swap_phase_calibration.analysis import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    FitParameters,
)
from calibration_utils.swap_phase_calibration.plotting import plot_raw_data_with_fit

# %% {Description}
description = """

"""

node = QualibrationNode(name="swap_phase_calibration", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["q0-1"]
    node.parameters.reset_type = "active"
    node.parameters.use_state_discrimination = True
    node.parameters.phase_min = 0.0
    node.parameters.phase_max = 1.0
    node.parameters.phase_steps_number = 21


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


@node.run_action()
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and node parameters."""

    u = unit(coerce_to_integer=True)
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Define and store the amplitudes for the flux pulses
    # pulse_amplitudes = {}
    # for qp in qubit_pairs:
    #     detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency
    #     pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))

    node.namespace["qubits"] = qubits = [qp.qubit_control for qp in qubit_pairs] + [
        qp.qubit_target for qp in qubit_pairs
    ]

    # Loop parameters
    n_avg = node.parameters.num_shots  # The number of averages
    phases = np.linspace(node.parameters.phase_min, node.parameters.phase_max, node.parameters.phase_steps_number)

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "phases": xr.DataArray(phases, attrs={"long_name": "phase ", "units": "rad"}),
    }

    with program() as node.namespace["qua_program"]:
        phase = declare(float)
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for target in list(node.machine.active_qubits) + [
            qp.coupler for qp in node.machine.active_qubit_pairs if hasattr(qp, "coupler") and qp.coupler is not None
        ]:
            node.machine.initialize_qpu(target=target)
        wait(1000)  # TODO: DO WE NEED THIS WAIT?

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                for temp in range(1):
                    with for_(*from_array(phase, phases)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            # Qubit initialization
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()

                            # setting the qubits ot the initial state
                            qp.qubit_target.xy.play("x90")

                            qp.align()

                            qp.gates["SWAP_Coupler"].execute()  # TODO: WHAT DOES THIS DO?

                            qp.align()
                            qp.xy.frame_rotation_2pi(phase)
                            qp.align()

                            qp.qubit_control.xy.play("x90")

                            qp.align()

                        # measure both qubits
                        if node.parameters.use_state_discrimination:
                            qp.qubit_control.readout_state(state_c[ii])
                            qp.qubit_target.readout_state(state_t[ii])
                            assign(state[ii], state_c[ii] * 2 + state_t[ii])
                            save(state_c[ii], state_c_st[ii])
                            save(state_t[ii], state_t_st[ii])
                            save(state[ii], state_st[ii])
                        else:
                            qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                            qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                            save(I_c[ii], I_c_st[ii])
                            save(Q_c[ii], Q_c_st[ii])
                            save(I_t[ii], I_t_st[ii])
                            save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(phases)).average().save(f"state_control{i}")
                    state_t_st[i].buffer(len(phases)).average().save(f"state_target{i}")
                    state_st[i].buffer(len(phases)).average().save(f"state{i}")
                else:
                    I_c_st[i].buffer(len(phases)).average().save(f"I_control{i}")
                    Q_c_st[i].buffer(len(phases)).average().save(f"Q_control{i}")
                    I_t_st[i].buffer(len(phases)).average().save(f"I_target{i}")
                    Q_t_st[i].buffer(len(phases)).average().save(f"Q_target{i}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    ds_fit, fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


@node.run_action()
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_control, fig_traget = plot_raw_data_with_fit(
        node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"], node.results["fit_results"]
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {"raw_fit_control": fig_control, "raw_fit_target": fig_traget}


@node.run_action()
def update_state(node: QualibrationNode[Parameters, Quam]):
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in node.namespace["qubit_pairs"]:
                qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude = node.results["results"][qp.name][
                    "SWAP_amplitude"
                ]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
