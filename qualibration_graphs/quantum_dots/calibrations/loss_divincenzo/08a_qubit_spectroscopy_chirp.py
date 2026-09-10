# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import QubitQuam as Quam

from calibration_utils.qubit_spectroscopy_chirp import (
    Parameters,
    fit_raw_data,
    generate_simulated_dataset,
    resolve_operation_name,
    get_durations_and_chirp_rates,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from calibration_utils.common_utils.macro_updates import change_macro_tracked, revert_tracked_macros


# %% {Node initialization}
description = """
CHIRPED QUBIT SPECTROSCOPY

This node sweeps the qubit drive frequency around the current RF estimate by
chirping across narrow frequency bands and measuring the resulting response via
PSB. When the qubit frequency falls within one of the chirped bands, the signal
is elevated and can be used to estimate the qubit Larmor frequency.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the PSB readout scheme.
    - Having a reasonable initial RF frequency estimate for the selected qubits.

Datasets:
    - ``ds_raw``: untouched thresholded ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: optional processed chirp-spectroscopy traces plus peak-fit outputs when ``fit_peak=True``.
    - ``fit_results``: compact per-qubit threshold-analysis dict. Used by logging, ``node.outcomes``, and ``update_state``.
    - ``peak_fit_results``: optional per-qubit peak-fit diagnostics dict.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the threshold-based frequency estimate passed the node criteria.
    - ``frequency`` [Hz]: threshold-estimated qubit Larmor frequency.
    - ``relative_freq`` [Hz]: threshold-estimated detuning relative to the current RF frequency.
    - ``fwhm`` [Hz]: width of the above-threshold detuning region.

Figures (``node.results["figures"]``):
    - ``"qubit_spectroscopy_chirp"``: state-probability trace with optional threshold and peak-fit overlays.

State update:
    - Update the qubit Larmor frequency from the threshold-based chirp analysis.
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
    """Build the 1D chirped-frequency sweep and the QUA pulse sequence."""

    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    change_macro_tracked(node, qubits) # Applies any node parameter updates to custom macros

    # Resolve the spectroscopy pulse from the pulse family stored in the Quam state.
    operation_name = resolve_operation_name(node, node.parameters.operation)

    # Quam's .play() function takes the CHIRP RATE as an arg, along with a unit
    # For each frequency step (in Hz) divide by the desired operation length. This is calcualted per qubit
    op_len_per_qubit, chirp_rate_per_qubit = get_durations_and_chirp_rates(node, node.parameters.operation)

    # Pulse amplitude prefactor must stay within [-2, 2) for QUA fixed-point arithmetic.
    operation_amp_factor = node.parameters.operation_amplitude_factor

    n_avg = node.parameters.num_shots  # repetitions averaged at each detuning point

    # Frequency axis: chirp-band offsets relative to each qubit's RF frequency [Hz]
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Shift the stored coordinate by half a step so each point labels the chirp-band centre.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs + step / 2, attrs={"long_name": "drive frequency", "units": "Hz"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # state[i] : thresholded post-manipulation measurement (0/1) for qubit i
        # n       : shot counter
        # df      : integer frequency detuning centre value
        n = declare(int)
        df = declare(int)

        # Streams:
        # state_st[i] : per-qubit thresholded state stream
        # n_st : shot counter for progress reporting
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_output_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        # Python loop over the qubits specified in the node parameters
        for i, qubit in enumerate(qubits):
            # Remember calibrated IF so we can restore it after the detuning sweep
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTER LOOP: average over shots ───────────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── INNER LOOP: sweep frequency detuning ────────────────
                with for_(*from_array(df, dfs)):

                    # Set the qubit drive frequency to the stored IF, for initialization
                    qubit.xy.update_frequency(intermediate_frequency)

                    # Perform the initialize macro
                    qubit.initialize()
                    align()

                    # Retune the XY drive to (calibrated IF + df)
                    qubit.xy.update_frequency(intermediate_frequency + df)

                    align()
                    # Play the chirped spectroscopy pulse.
                    qubit.xy.play(
                        operation_name,
                        amplitude_scale=operation_amp_factor,
                        duration=op_len_per_qubit[qubit.name] // 4,
                        chirp=(chirp_rate_per_qubit[qubit.name], "Hz/nsec"),
                    )
                    align()

                    # Thresholded PSB readout → averaged state probability
                    s = qubit.measure()
                    assign(state[i], Cast.to_int(s))
                    save(state[i], state_st[i])

                    # Return gate voltages to zero before the next point.
                    align()
                    qubit.voltage_sequence.ramp_to_zero()

            # Restore the qubit's calibrated drive frequency after the sweep
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Each save() is one frequency point.
                # .buffer(len(dfs)) : group points along the frequency axis
                # .average()      : average over all shots (n_avg repetitions)
                # Result: 1D state-probability trace vs detuning per qubit.
                state_st[i].buffer(len(dfs)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
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
        # "samples": samples,
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
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Fit the spectroscopy response and store both the fitted dataset and fit summary."""
    node.results["ds_processed"] = ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    (
        node.results["ds_fit"],
        node.results["fit_results"],
        node.results["peak_fit_results"],
        node.outcomes,
    ) = fit_raw_data(ds_processed, node, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results.get("ds_processed", node.results["ds_raw"]),
        node.namespace["qubits"],
        fits=node.results.get("ds_fit"),
        threshold_results=node.results["fit_results"],
        signal_threshold=node.parameters.signal_threshold,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.use_simulated_data)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the qubit frequency from threshold-based analysis."""
    revert_tracked_macros(node)
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
    """Persist the node results and any recorded state updates."""
    node.save()
