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
from quam_config import QubitQuam

Quam = QubitQuam
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
from qualibration_libs.core import tracked_updates

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
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)
    operation = node.parameters.operation  # The qubit operation to play

    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    operation_amp_factor = node.parameters.operation_amplitude_factor

    # Change the pulse amplitude and duration as a tracked change
    node.namespace["tracked_resonators"] = []
    for i, qubit in enumerate(qubits):
        with tracked_updates(qubit.xy, auto_revert=False, dont_assign_to_none=True) as xy:
            xy.operations[operation].amplitude = xy.operations[operation].amplitude * operation_amp_factor
            xy.operations[operation].length = operation_len
            node.namespace["tracked_resonators"].append(xy)

    u = unit(coerce_to_integer=True)
    n_avg = node.parameters.num_shots  # The number of averages

    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "drive frequency", "units": "Hz"}),
    }

    with program() as node.namespace["qua_program"]:
        df = declare(int)
        n = declare(int)

        p1 = declare(int, size=num_qubits)
        p2 = declare(int, size=num_qubits)

        p1_st = {qubit.name: declare_stream() for qubit in qubits}
        p2_st = {qubit.name: declare_stream() for qubit in qubits}
        pdiff_st = {qubit.name: declare_stream() for qubit in qubits}
        n_st = declare_stream()

        for batched_qubits in qubits.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(df, dfs)):

                    # Update the frequency before the sequence
                    for i, qubit in batched_qubits.items():
                        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + df)

                    # ---------------------------------------------------------
                    # Step 1a: Empty - step to empty point (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.empty()

                    # ---------------------------------------------------------
                    # Step 1b: Initialize - load electron into dot (fixed duration)
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.initialize(duration=node.parameters.gap_wait_time_in_ns)

                    # ---------------------------------------------------------
                    # Step 2: Measure state after initialization, no operation
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        assign(
                            p1[i], Cast.to_int(qubit.measure())
                        )  # qubit.measure() handles the step to point + measurement

                    # ---------------------------------------------------------
                    # Step 3: Apply operation macro
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        # qubit.apply macros moves the qubit to the relevant operation point, and applies the operation
                        # The operation amplitude and length are already set as the tracked change
                        qubit.apply(operation)

                    # ---------------------------------------------------------
                    # Step 4: Measure - move to PSB and measure
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        assign(p2[i], Cast.to_int(qubit.measure()))

                    # ---------------------------------------------------------
                    # Step 5: Apply compensation pulse to reset DC bias
                    # ---------------------------------------------------------
                    align()
                    for i, qubit in batched_qubits.items():
                        qubit.voltage_sequence.apply_compensation_pulse()

                    # ---------------------------------------------------------
                    # Save results
                    # ---------------------------------------------------------
                    for i, qubit in batched_qubits.items():
                        save(p1[i], p1_st[qubit.name])
                        save(p2[i], p2_st[qubit.name])

                        with if_(p1[i] == p2[i]):
                            save(0, pdiff_st[qubit.name])
                        with else_():
                            save(1, pdiff_st[qubit.name])

        with stream_processing():
            n_st.save("n")
            n_dfs = len(dfs)
            for qubit in qubits:
                p1_st[qubit.name].buffer(n_dfs).average().save(f"p1_{qubit.name}")
                p2_st[qubit.name].buffer(n_dfs).average().save(f"p2_{qubit.name}")
                pdiff_st[qubit.name].buffer(n_dfs).average().save(f"pdiff_{qubit.name}")


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
    node.namespace["qubits"] = [node.machine.qubits[q] for q in node.parameters.qubits]


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
            q.xy.RF_frequency = opt_frequency
