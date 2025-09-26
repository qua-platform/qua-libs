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
from calibration_utils.xy_crosstalk_coupling_magnitude import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from calibration_utils.data_process_utils import *
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


# TODO: Implement this protocol


# %% {Description}
description = """
Characterizes the magnitude component of the microwave crosstalk matrix.

The experiment is Rabi-type: the driven qubit (Qd) is pulsed at the resonance
frequency of the probed qubit (Qp), and the induced Rabi rate on Qp is fitted.
By normalizing this rate to the self-Rabi rate obtained when driving Qp
directly, the relative crosstalk strength |C_ij| can be extracted.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="20a_XY_crosstalk_coupling_magnitude",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.multiplexed = False
    node.parameters.qubits = ["q1", "q2", "q3"]
    node.parameters.use_state_discrimination = False

    node.parameters.num_shots = 10


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
    num_qubit_pairs = num_qubits ** 2

    n_avg = node.parameters.num_shots  # The number of averages
    state_discrimination = node.parameters.use_state_discrimination
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "qubit_probed": xr.DataArray(qubits.get_names()),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        I_d, I_d_st, Q_d, Q_d_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        I_p, I_p_st, Q_p, Q_p_st, _, _ = node.machine.declare_qua_variables(num_IQ_pairs=num_qubit_pairs)
        if state_discrimination:
            state_d = [declare(int) for _ in range(num_qubit_pairs)]
            state_p = [declare(int) for _ in range(num_qubit_pairs)]
            state_d_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_p_st = [declare_stream() for _ in range(num_qubit_pairs)]
        t = declare(int)

        # Reset explicitly
        reset_global_phase()

        for qubit in qubits:
            node.machine.initialize_qpu(target=qubit)
        align()

        # TODO:
        # ebable to iterate over subset of qubits and generate consistent xarray accordingly
        # not necessarily having to iterate over num_qubits ** 2 number of pairs  
        for j, qb_driven in enumerate(qubits):

            for k, qb_probed in enumerate(qubits):

                i = j * num_qubits + k
                assign(n, i)
                save(n, n_st)
                
                # Set the xy drive frequency back to the qubit frequency for active reset
                qb_driven.xy.update_frequency(
                    qb_probed.xy.upconverter_frequency + \
                    qb_probed.xy.intermediate_frequency - \
                    qb_driven.xy.upconverter_frequency
                )

                with for_(n, 0, n < n_avg, n + 1):
 
                    with for_(*from_array(t, pulse_durations // 4)):

                        # Qubit initialization
                        qb_driven.reset(node.parameters.reset_type, node.parameters.simulate)
                        qb_probed.reset(node.parameters.reset_type, node.parameters.simulate)

                        align(qb_driven.xy.name, qb_probed.xy.name)
                        
                        # Qubit manipulation
                        qb_driven.xy.play("x180", duration=t)

                        align(qb_driven.xy.name, qb_driven.resonator.name, qb_probed.resonator.name)

                        # Measure the state of the resonators
                        if state_discrimination:
                            qb_driven.readout_state(state_d[i])
                            qb_probed.readout_state(state_p[i])
                            save(state_d[i], state_d_st[i])
                            save(state_p[i], state_p_st[i])
                        else:
                            qb_driven.resonator.measure("readout", qua_vars=(I_d[i], Q_d[i]))
                            qb_probed.resonator.measure("readout", qua_vars=(I_p[i], Q_p[i]))
                            # save data
                            save(I_d[i], I_d_st[i])
                            save(Q_d[i], Q_d_st[i])
                            save(I_p[i], I_p_st[i])
                            save(Q_p[i], Q_p_st[i])

                        # Reset the frame of the qubits in order not to accumulate rotations
                        reset_frame(qb_driven.xy.name)
                        reset_frame(qb_probed.xy.name)

                        # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                        qb_driven.resonator.wait(qb_driven.resonator.depletion_time // 4)
                        qb_probed.resonator.wait(qb_probed.resonator.depletion_time // 4)

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if state_discrimination:
                    state_d_st[i].buffer(len(pulse_durations)).average().save(f"state_d{i + 1}")
                    state_p_st[i].buffer(len(pulse_durations)).average().save(f"state_p{i + 1}")
                else:
                    I_d_st[i].buffer(len(pulse_durations)).average().save(f"I_d{i + 1}")
                    Q_d_st[i].buffer(len(pulse_durations)).average().save(f"Q_d{i + 1}")
                    I_p_st[i].buffer(len(pulse_durations)).average().save(f"I_p{i + 1}")
                    Q_p_st[i].buffer(len(pulse_durations)).average().save(f"Q_p{i + 1}")


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
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
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
    for tracked_qubit in node.namespace.get("tracked_qubits", []):
        tracked_qubit.revert_changes()

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
