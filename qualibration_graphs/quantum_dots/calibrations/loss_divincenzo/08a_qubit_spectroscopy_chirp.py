# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.qubit_spectroscopy_chirp_parity_diff import (
    Parameters,
    fit_raw_data,
    find_frequency_by_threshold,
    log_fitted_results,
    plot_raw_data_with_fit,
    generate_simulated_dataset,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        CHIRPED QUBIT SPECTROSCOPY
This sequence involves parking the qubit at the manipulation bias point, and probing the qubit. This node is designed
to roughly estimate the qubit frequency by chirping through a series of frequency bands and measuring the parity. When
the qubit frequency is within the chirped frequency band, the qubit is partially driven, and the measured parity is used
to estimate the Larmor frequency.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.


State update:
    - The approximate qubit frequency (and optionally the corresponding LO/IF plan) for the specified qubit operation.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="08a_chirped_qubit_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["Q1", "Q2", "Q3", "Q4"]
    # node.parameters.operation_len_in_ns = 2000
    # node.parameters.use_simulated_data = False
    # node.parameters.frequency_span_in_mhz = 100
    # node.parameters.frequency_step_in_mhz = 1
    # node.parameters.num_shots = 1
    # node.parameters.simulate = True
    # node.parameters.simulation_duration_ns = 100_000
    # node.parameters.use_simulated_data = False
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.use_simulated_data
)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    operation = node.parameters.operation  # The qubit operation to play

    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    operation_len = node.parameters.operation_len_in_ns

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    operation_amp_factor = node.parameters.operation_amplitude_factor

    chirp_rate = (
        node.parameters.frequency_step_in_mhz
        * 1e6
        / node.parameters.operation_len_in_ns
    )  # Rate will be Hz/nsec

    for qubit in qubits:
        qubit.x.update(
            amplitude_scale=operation_amp_factor,
            duration=operation_len,
        )
        # qubit.xy.operations[operation].length = operation_len
        # qubit.xy.operations[operation].amplitude = (
        #     qubit.xy.operations[operation].amplitude * operation_amp_factor
        # )

    u = unit(coerce_to_integer=True)
    n_avg = node.parameters.num_shots  # The number of averages

    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs, attrs={"long_name": "drive frequency", "units": "Hz"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        df = declare(int)
        n = declare(int)

        p2, p1, parity_streams = declare_parity_streams(node, qubits)

        n_st = declare_stream()
        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        n_loops_st = (
            {qubit.name: declare_stream() for qubit in qubits}
            if heralded_and_return_n_loops
            else {}
        )

        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    intermediate_frequency = qubit.xy.intermediate_frequency
                    qubit.xy.update_frequency(intermediate_frequency + df)

                    if node.parameters.parity_pre_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    n_init = qubit.initialize()
                    if heralded_and_return_n_loops:
                        save(n_init, n_loops_st[qubit.name])
                    qubit.xy.play(operation, chirp=(chirp_rate, "Hz/nsec"))

                    align()
                    a2 = qubit.measure()

                    align()
                    qubit.voltage_sequence.ramp_to_zero()

                    qubit.xy.update_frequency(intermediate_frequency)

                    assign(p2, Cast.to_int(a2))

                    if node.parameters.parity_pre_measurement:
                        assign(p1, Cast.to_int(a1))

                    save_parity_measurement(node, qubit.name, p1, p2, parity_streams)

        with stream_processing():
            n_st.save("n")
            n_dfs = len(dfs)
            for qubit in qubits:
                buffer_parity_streams(node, qubit.name, parity_streams, n_dfs)
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_dfs).average().save(
                        f"n_loops_{qubit.name}"
                    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
)
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
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated chirp spectroscopy data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome parity streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("detuning",),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data using threshold detection, and optionally peak fitting."""
    # Always run threshold-based frequency detection
    threshold_results = find_frequency_by_threshold(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in threshold_results.items()}
    log_fitted_results(
        node.results["fit_results"], log_callable=node.log, label="Threshold"
    )

    # Optionally run peak fitting and compare
    if node.parameters.fit_peak:
        node.results["ds_fit"], peak_results = fit_raw_data(
            node.results["ds_raw"], node
        )
        node.results["peak_fit_results"] = {
            k: asdict(v) for k, v in peak_results.items()
        }
        log_fitted_results(
            node.results["peak_fit_results"], log_callable=node.log, label="Peak fit"
        )

        for q_name, thr in node.results["fit_results"].items():
            peak = node.results["peak_fit_results"].get(q_name, {})
            if not (thr.get("success") and peak.get("success")):
                continue
            tolerance = thr["fwhm"] / 2 if thr["fwhm"] > 0 else np.inf
            diff = abs(peak["frequency"] - thr["frequency"])
            if diff > tolerance:
                node.log(
                    f"WARNING {q_name}: peak fit ({1e-9 * peak['frequency']:.4f} GHz) and "
                    f"threshold ({1e-9 * thr['frequency']:.4f} GHz) disagree by "
                    f"{1e-3 * diff:.1f} kHz (tolerance: {1e-3 * tolerance:.1f} kHz)"
                )

    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        fits=node.results.get("ds_fit"),
        threshold_results=node.results["fit_results"],
        signal_threshold=node.parameters.signal_threshold,
        analysis_signal=node.parameters.analysis_signal,
    )
    plt.show()
    node.results["figures"] = {
        "qubit_spectroscopy_chirp": fig_raw_fit,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the qubit frequency from threshold-based analysis."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            opt_frequency = node.results["fit_results"][q.name]["frequency"]
            q.larmor_frequency = opt_frequency
            q.x.update(frequency=opt_frequency)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
