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
from calibration_utils.common_utils.experiment import get_sensors, get_qubits
from calibration_utils.qubit_spectroscopy_parity_diff import (
    Parameters,
    # process_raw_dataset,
    # fit_raw_data,
    # log_fitted_results,
    # plot_raw_data_with_fit,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        QUBIT SPECTROSCOPY PARITY DIFFERENCE
This sequence involves parking the qubit at the manipulation bias point, and playing pulses of varying frequency
to drive the qubit. When the pulse frequency is the Larmor frequency, the qubit is driven, and the parity is measured
via PSB.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.


State update:
    - The qubit frequency (and optionally the corresponding LO/IF plan) for the specified qubit operation.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="07a_qubit_spectroscopy_parity_diff",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # # Get the active qubits from the node and organize them by batches
    # node.namespace["qubits"] = qubits = get_qubits(node)
    # num_qubits = len(qubits)
    # # Get the relevant sensor dots rom the node
    # # node.namespace["sensors"] = sensors = get_sensors(node)
    # # num_sensors = len(sensors)
    #
    # n_avg = node.parameters.num_shots  # The number of averages
    # operation = node.parameters.operation  # The qubit operation to play
    # # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    # amps = np.arange(
    #     node.parameters.min_amp_factor,
    #     node.parameters.max_amp_factor,
    #     node.parameters.amp_factor_step,
    # )
    # # Register the sweep axes to be added to the dataset when fetching data
    # node.namespace["sweep_axes"] = {
    #     "qubit": xr.DataArray(qubits.get_names()),
    #     "amp_prefactor": xr.DataArray(amps, attrs={"long_name": "pulse amplitude prefactor"}),
    # }
    #
    # # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    # with program() as node.namespace["qua_program"]:
    #
    #     I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_qubits)
    #     if node.parameters.use_state_discrimination:
    #         state = [declare(int) for _ in range(num_qubits)]
    #         state_st = [declare_stream() for _ in range(num_qubits)]
    #     a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    #
    #     for multiplexed_qubits in qubits.batch():
    #         # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
    #         for qubit in multiplexed_qubits.values():
    #             # node.machine.initialize_qpu(target=qubit)
    #             qubit.step_to_point("empty")
    #         align()
    #
    #         with for_(n, 0, n < n_avg, n + 1):
    #             save(n, n_st)
    #             with for_(*from_array(a, amps)):
    #                 # Qubit initialization
    #                 for i, qubit in multiplexed_qubits.items():
    #                     qubit.initialize(node.parameters.reset_type, node.parameters.target_state, node.parameters.simulate)
    #                     # qubit.step_to_point("initialization")
    #                 align()
    #
    #                 # Qubit manipulation
    #                 for i, qubit in multiplexed_qubits.items():
    #                     # qubit.step_to_point("manipulation", duration=t)
    #                     qubit.xy_channel.play(operation, amplitude_scale=a)
    #
    #                 align()
    #
    #                 # Qubit readout
    #                 for i, qubit in multiplexed_qubits.items():
    #                     qubit.step_to_point("readout")
    #                     qubit.quantum_dot_pair.sensor_dot.resonator.measure()
    #                     qubit.machine.sensor_dots
    #                     if node.parameters.use_state_discrimination:
    #                         # qubit.readout_state(state[i])
    #                         save(state[i], state_st[i])
    #                     else:
    #                         qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
    #                         save(I[i], I_st[i])
    #                         save(Q[i], Q_st[i])
    #                 align()
    #
    #     with stream_processing():
    #         n_st.save("n")
    #         for i in range(num_qubits):
    #             I_st[i].buffer(len(amps)).average().save(f"I{i + 1}")
    #             Q_st[i].buffer(len(amps)).average().save(f"Q{i + 1}")


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
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    # node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    # node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    # node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    #
    # # Log the relevant information extracted from the data analysis
    # log_fitted_results(node.results["fit_results"], log_callable=node.log)
    # node.outcomes = {
    #     sensor_name: ("successful" if fit_result["success"] else "failed")
    #     for sensor_name, fit_result in node.results["fit_results"].items()
    # }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""
    # fig_raw_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["sensors"])
    # fig_fit_amplitude = plot_raw_amplitude_with_fit(
    #     node.results["ds_raw"], node.namespace["sensors"], node.results["ds_fit"]
    # )
    # plt.show()
    # # Store the generated figures
    # node.results["figures"] = {
    #     "phase": fig_raw_phase,
    #     "amplitude": fig_fit_amplitude,
    # }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit spectroscopy parity-diff analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            opt_frequency = node.results["fit_results"][q.name]["frequency"]
            q.larmor_frequency = opt_frequency
            q.xy.intermediate_frequency = opt_frequency
