# %% {Imports}
from calibration_utils.common_utils.parity_streams import process_parity_streams
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam

from calibration_utils.measurement_utils import (
    save_measurement, 
    buffer_streams, 
    declare_streams,
    process_streams,
)
from calibration_utils.qubit_spectroscopy_chirp import (
    Parameters,
    fit_raw_data,
    find_frequency_by_threshold,
    log_fitted_results,
    plot_raw_data_with_fit,
    generate_simulated_dataset,
    resolve_operation_name,
    get_durations_and_chirp_rates,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
# from qualibration_libs.core import tracked_updates

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
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Find the operation name to play based on the pulse_family saved in the Quam state
    operation_name = resolve_operation_name(node, node.parameters.operation)

    # Quam's .play() function takes the CHIRP RATE as an arg, along with a unit
    # For each frequency step (in Hz) divide by the desired operation length. This is calcualted per qubit
    op_len_per_qubit, chirp_rate_per_qubit = get_durations_and_chirp_rates(node, node.parameters.operation)

    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    operation_amp_factor = node.parameters.operation_amplitude_factor

    n_avg = node.parameters.num_shots  # The number of averages

    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Register the sweep axes to be added to the dataset when fetching data
    # Shift the dfs coordinate by half a step, so that the stored df always represents the centre of a chirp
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs + step / 2, attrs={"long_name": "drive frequency", "units": "Hz"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # n  : shot counter
        # p1 : post-manipulation measurement outcome (0 = empty, 1 = loaded)
        # p0 : pre-manipulation measurement outcome (only used when parity_measurement=True, otherwise None)
        # df : integer frequency detuning centre value
        n = declare(int)
        df = declare(int)
        
        # Streams: 
        # measurement_streams : stores the per-qubit assigned value, parity difference if node.parameters.parity_measurement = True
        # n_st : stores the shot counter n, allowing the PC to track the progress
        n_st = declare_stream()
        p1, p0, measurement_streams = declare_streams(node, qubits)
        
        # Python loop over the qubits specified in the node parameters
        for qubit in qubits:
            # Extract the qubit's intermediate frequency. Stored as an attribute of the qubit's XY drive object
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTER LOOP: average over shots ───────────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)# tell the PC which shot we are on

                # ── INNER LOOP: sweep frequency detuning ────────────────
                with for_(*from_array(df, dfs)):
                    # Global align at the start of each shot
                    align()

                    # ── STEP 1: Preparation & Initialization ────────────────
                
                    # Optional pre-measurement at the empty bias point (parity readout)
                    if node.parameters.parity_measurement:
                        qubit.empty()
                        a1 = qubit.measure()

                    # Perform the initialize macro
                    qubit.initialize(
                        target_state=node.parameters.target_state,
                        max_loops=node.parameters.max_loops,
                    )

                    # Detune the IF of the qubit's XY component
                    align()
                    qubit.xy.update_frequency(intermediate_frequency + df)

                    # ── STEP 2: Drive ────────────────

                    # Align and play the x180 pulse
                    align()
                    qubit.xy.play(operation_name, amplitude_scale = operation_amp_factor, duration = op_len_per_qubit[qubit.name] // 4, chirp=(chirp_rate_per_qubit[qubit.name], "Hz/nsec"))
                    align()

                    # ── STEP 3: Measure the resulting state ────────────────

                    # Measure the resulting state, returning thresholded state value (bool)
                    a2 = qubit.measure()

                    # Set any remaining offset to zero. TODO: Consider whether necessary
                    qubit.voltage_sequence.ramp_to_zero()

                    align()
                    
                    # If performing a parity measurement, assign the thresholded bool to an integer (0 or 1) for averaging
                    if node.parameters.parity_measurement:
                        assign(p0, Cast.to_int(a1))

                    # Assign the thresholded bool to an integer (0 or 1) for averaging
                    assign(p1, Cast.to_int(a2))

                    # Save the measurement QUA variables to the relevant streams
                    save_measurement(node, qubit.name, p0, p1, measurement_streams)

                    # Set the frequency to the value stored in the Quam state
                    # This is so that for each shot, the initialisation macro uses the previosuly assigned (correct) frequency
                    qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            n_dfs = len(dfs)
            for qubit in qubits:
                # Each save() is one frequency point.
                # .buffer(n_dfs) : group points along the frequency axis
                # .average()      : average over all shots (n_avg repetitions)
                # Result: 1D measured counts vs frequency detuning per qubit
                # Stream shapes depends on whether parity streams are being used. 
                if node.parameters.parity_measurement:
                    for key in ("p0_p0", "p0_p1", "p1_p0", "p1_p1"):
                        measurement_streams[key][qubit.name].buffer(n_dfs).average().save(
                            f"{key}_{qubit.name}_parity_diff"
                        )
                else:
                    measurement_streams["p"][qubit.name].buffer(n_dfs).average().save(
                        f"p_{qubit.name}_parity_diff"
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
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
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
            progress_counter(
                data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start
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
    node.results["ds_raw"] = process_streams(
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
    log_fitted_results(node.results["fit_results"], log_callable=node.log, label="Threshold")

    # Optionally run peak fitting and compare
    if node.parameters.fit_peak:
        node.results["ds_fit"], peak_results = fit_raw_data(node.results["ds_raw"], node)
        node.results["peak_fit_results"] = {k: asdict(v) for k, v in peak_results.items()}
        log_fitted_results(node.results["peak_fit_results"], log_callable=node.log, label="Peak fit")

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
