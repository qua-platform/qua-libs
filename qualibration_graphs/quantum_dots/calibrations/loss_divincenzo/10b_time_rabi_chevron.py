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
from quam_config import Quam

from calibration_utils.time_rabi_chevron import (
    Parameters,
    fit_raw_data,
    generate_simulated_dataset,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        TIME RABI CHEVRON
After initialization to the target spin state, this sequence applies an XY drive pulse whose duration and
frequency detuning are swept and measures the spin state with thresholded PSB readout. Averaged state probabilities
versus pulse duration and detuning form a 2D chevron that reveals the resonant drive frequency and π-pulse duration.

Prerequisites:
    - Having calibrated the resonators coupled to the sensor dots.
    - Having calibrated the voltage points (empty, initialization, measurement), including sensor dot bias.
    - Having a rough qubit XY drive calibration (amplitude, frequency, and duration).

Datasets:
    - ``ds_raw``: untouched ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed 2D sweeps plus analysis outputs. Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with ``asdict``). Used by
      logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``optimal_frequency`` [Hz]: resonant drive frequency at the chevron crossing.
    - ``optimal_duration`` [ns]: π-pulse duration at resonance.
    - ``rabi_frequency`` [rad / ns]: fitted on-resonance Rabi frequency.
    - ``decay_rate`` [1 / ns]: fitted decay rate of the Rabi envelope (γ ≈ 1/T₂*).

Figures (``node.results["figures"]``):
    - ``"chevron"``: 2D heatmap of state vs pulse duration and drive detuning.
    - ``"fft_2d"``: 2D FFT magnitude map with hyperbolic ridge overlay.
    - ``"diagnostics"``: FFT at resonance and t_π vs detuning with Rabi fit per qubit.

State update:
    - The pulse duration of the selected operation (``node.parameters.operation``).
    - The qubit Larmor frequency, adjusted by the fitted frequency offset from the current XY intermediate frequency.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="10b_time_rabi_chevron",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 2D detuning × duration chevron sweep and the QUA pulse sequence."""
    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each (detuning, duration) point
    operation = node.parameters.operation  # qubit gate whose duration is swept (x180 or x90)

    # Duration axis [ns]: sweep around the expected π-pulse length (quantised to 4 ns on the OPX)
    pulse_durations = np.arange(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.time_step_in_ns,
    )

    # Frequency axis: offsets relative to each qubit's current XY intermediate frequency [Hz]
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    # Metadata for data fetching: labels the saved state arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "drive frequency detuning", "units": "Hz"}),
        "pulse_duration": xr.DataArray(pulse_durations, attrs={"long_name": "qubit pulse duration", "units": "ns"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        # t         : current manipulation pulse duration [ns]
        # df        : current XY drive frequency offset [Hz]
        # n         : shot counter
        # state[i]  : thresholded post-manipulation measurement (0/1) for qubit i
        t = declare(int)
        df = declare(int)
        n = declare(int)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        for i, qubit in enumerate(qubits):
            # Remember calibrated IF so we can restore it after the detuning sweep
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTERMOST LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── MIDDLE LOOP: sweep drive detuning ────────────────────
                with for_(*from_array(df, dfs)):

                    # ── INNER LOOP: sweep pulse duration in clock cycles ─────────────────
                    with for_(*from_array(t, pulse_durations // 4)):

                        # Retune the XY drive to (calibrated IF + df)
                        qubit.xy.update_frequency(intermediate_frequency + df)

                        # Perform the initialize macro
                        qubit.initialize()
                        align()

                        # Play the selected gate at the current duration (chevron / time-Rabi)
                        qubit.macros[operation].apply(duration=t)
                        align()

                        # Thresholded PSB readout → averaged state probability
                        assign(state[i], Cast.to_int(qubit.measure()))
                        save(state[i], state_st[i])

                        # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                        align()
                        qubit.voltage_sequence.ramp_to_zero()

            # Restore the qubit's calibrated drive frequency after the sweep
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Save order: for each detuning, sweep all duration values.
                # .buffer(len(pulse_durations)) → inner axis = pulse_duration
                # .buffer(len(dfs))             → outer axis = detuning
                # .average()                    → average over shots
                # Result: 2D state vs (detuning, pulse_duration) per qubit
                state_st[i].buffer(len(pulse_durations)).buffer(len(dfs)).average().save(f"state{i + 1}")


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
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated Rabi-chevron data so the full analysis pipeline can run without hardware."""
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
    """Process state streams, fit chevron data, and store results."""
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {qname: ("successful" if r["success"] else "failed") for qname, r in fit_results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes[qubit.name] == "failed":
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.macros[node.parameters.operation].update(
                duration=fit_result["optimal_duration"],
            )
            # Shift Larmor by the offset between fitted resonance and the current XY IF.
            qubit.larmor_frequency += fit_result["optimal_frequency"] - qubit.xy.intermediate_frequency


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
