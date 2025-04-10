# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from quam_experiments.experiments.stark_detuning_calibration import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_experiments.parameters.qubits_experiment import get_qubits
from quam_experiments.workflow import simulate_and_plot
from qualibration_libs.xarray_data_fetcher import XarrayDataFetcher
from qualibration_libs.trackable_object import tracked_updates


# %% {Initialisation}
description = """
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses
successively for different DRAG detunings.
After such a sequence, the qubit is expected to always be in the ground state if the AC
Stark shift is properly compensated by the DRAG detuning.
One can then take a line cut for a given number of pulse and fit the 1D trace with a
parabola to get the optimum detuning and update its value in the configuration.

This protocol is described in more details in
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under
      study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy,
      rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude,
      duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the DRAG detuning and set-point (alpha) in the state.
"""


node = QualibrationNode[Parameters, Quam](
    name="10a_stark_detuning",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    # Update the readout power to match the desired range, this change will be reverted at the end of the node.
    node.namespace["tracked_qubits"] = []
    for q in qubits:
        with tracked_updates(q, auto_revert=False) as q:
            if node.parameters.DRAG_setpoint is not None:
                q.xy.operations[node.parameters.operation].alpha = node.parameters.DRAG_setpoint
            q.xy.operations[node.parameters.operation].detuning = 0
            node.namespace["tracked_qubits"].append(q)

    n_avg = node.parameters.num_averages  # The number of averages
    # Pulse frequency sweep
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)
    # Number of applied Rabi pulses sweep
    N_pi = node.parameters.max_number_pulses_per_sweep  # Maximum number of qubit pulses
    N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "nb_of_pulses": xr.DataArray(N_pi_vec, attrs={"long_name": "number of pulses"}),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit detuning", "units": "Hz"}),
    }
    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.qua_declaration()
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for the qubit drive amplitude pre-factor
        npi = declare(int)  # QUA variable for the number of qubit pulses
        count = declare(int)  # QUA variable for counting the qubit pulses

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(npi, N_pi_vec)):
                    with for_(*from_array(df, dfs)):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        # Qubit manipulation
                        for i, qubit in multiplexed_qubits.items():
                            # Update the qubit frequency after initialization for active reset
                            update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)
                            with for_(count, 0, count < npi, count + 1):
                                if node.parameters.operation == "x180":
                                    qubit.xy.play(node.parameters.operation)
                                    qubit.xy.play(node.parameters.operation, amplitude_scale=-1.0)
                                elif node.parameters.operation == "x90":
                                    qubit.xy.play(node.parameters.operation)
                                    qubit.xy.play(node.parameters.operation)
                                    qubit.xy.play(node.parameters.operation, amplitude_scale=-1.0)
                                    qubit.xy.play(node.parameters.operation, amplitude_scale=-1.0)

                            # Update the qubit frequency back to the resonance frequency for active reset
                            update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                state_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"state{i + 1}")
                I_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).buffer(N_pi).average().save(f"Q{i + 1}")


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
            # print_progress_bar(job, iteration_variable="n", total_number_of_iterations=node.parameters.num_averages)
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_averages,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(f"Job execution report:\n{job.execution_report()}")
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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], node=node)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    # Revert the change done at the beginning of the node
    for qubit in node.namespace.get("tracked_qubits", []):
        qubit.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            # fit_result = node.results["fit_results"][q.name]
            #
            # q.xy.operations[operation].detuning = float(fit_result["detuning"])
            # if node.parameters.DRAG_setpoint is not None:
            #     q.xy.operations[operation].alpha = node.parameters.DRAG_setpoint


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
