# %% {Imports}
import logging
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.loops import from_array
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.ramsey_detuning import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialization}
description = """
RAMSEY DETUNING (TWO-τ)

Sweeps the drive-frequency detuning at two fixed idle times (τ_short
and τ_long) and measures the resulting state probability with thresholded PSB readout.

The two traces act as a Vernier: wide fringes (short τ) localise the
resonance coarsely, narrow fringes (long τ) sharpen the estimate.  Each
trace is fitted independently with a profiled differential-evolution
search over the oscillation frequency (linear parameters solved by
least-squares).  The resonance detuning δ₀ is the amplitude-weighted
mean of the per-trace estimates.  The amplitude ratio between traces
gives the exponential decay rate γ and dephasing time T₂*.

Prerequisites:
    - Calibrated resonators and voltage points (empty - init - measure).
    - Calibrated X90 pulse amplitude and frequency.

Datasets:
    - ``ds_raw``: untouched ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed sweeps plus analysis outputs. Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict. Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the two-trace joint fit passed the node criteria.
    - ``freq_offset`` [Hz]: fitted resonance detuning.
    - ``contrast``: short-τ Ramsey contrast.
    - ``decay_rate`` [1 / ns]: fitted Ramsey envelope decay rate.
    - ``t2_star`` [ns]: fitted Ramsey dephasing time.

Figures (``node.results["figures"]``):
    - ``"raw_data_with_fit"``: short-τ and long-τ detuning sweeps with cosine-fit overlays.

State update:
    - qubit.xy.intermediate_frequency
"""


node = QualibrationNode[Parameters, Quam](
    name="11b_ramsey_detuning",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 2D detuning × two-τ Ramsey sweep and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    u = unit(coerce_to_integer=True)

    # Select which qubits participate in this calibration
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each (detuning, tau) point

    # Two idle times in clock cycles (4 ns each)
    idle_times_cc = np.array(
        [
            node.parameters.idle_time_ns // 4,
            node.parameters.idle_time_long_ns // 4,
        ]
    )
    # Store the ns array as a float dtype
    idle_times_ns = idle_times_cc.astype(float) * 4

    # Construct the array of frequency detunings
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "tau": xr.DataArray(idle_times_ns, attrs={"long_name": "idle time", "units": "ns"}),
        "detuning": xr.DataArray(detuning_values, attrs={"long_name": "frequency detuning", "units": "Hz"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Real-time variables:
        # t       : idle time in clock cycles
        # df      : drive-frequency detuning [Hz]
        # n       : shot counter
        # state[i]: thresholded post-manipulation measurement (0/1) for qubit i
        t = declare(int)
        df = declare(int)
        n = declare(int)
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_output_stream() for _ in range(num_qubits)]
        n_st = declare_output_stream()

        # Python loop over the relevant qubits
        for i, qubit in enumerate(qubits):
            # Remember calibrated IF so we can restore it after the detuning sweep
            intermediate_frequency = qubit.xy.intermediate_frequency

            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── MIDDLE LOOP: sweep all detuning values ───────────────────────
                with for_(*from_array(df, detuning_values)):

                    # ── INNER LOOP: repeat the sweep at the short and long idle times ─────
                    with for_(*from_array(t, idle_times_cc)):

                        # Set the qubit drive frequency to the stored IF, for initialization
                        qubit.xy.update_frequency(intermediate_frequency)
                        reset_frame(qubit.xy.name)
                        align()

                        # Perform the initialize macro
                        qubit.initialize()
                        align()

                        # Retune the XY drive to (calibrated IF + df)
                        qubit.xy.update_frequency(intermediate_frequency + df)

                        align()
                        # Apply the Ramsey π/2 – idle – π/2 sequence for this detuning/τ point.
                        with strict_timing_():
                            qubit.x90()
                            wait(t, qubit.xy.name)
                            qubit.x90()
                        align()

                        # Thresholded PSB readout → averaged state probability
                        s = qubit.measure()
                        assign(state[i], Cast.to_int(s))
                        save(state[i], state_st[i])

                        # Return gate voltages to zero before the next shot.
                        align()
                        qubit.voltage_sequence.ramp_to_zero()

            # Restore the calibrated IF after the detuning sweep.
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Save order: for each detuning, sweep both idle-time values.
                # .buffer(len(idle_times_cc))    → inner axis = tau
                # .buffer(len(detuning_values))  → outer axis = detuning
                # .average()                     → average over shots
                # Result: 2D state vs (detuning, tau) per qubit
                state_st[i].buffer(len(idle_times_cc)).buffer(len(detuning_values)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
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
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program, and fetch raw state data into ``ds_raw``."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            # Display the progress bar
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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data."""
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results_full = fit_raw_data(ds_processed, node)
    fit_results = {k: {kk: vv for kk, vv in v.items() if kk != "_diag"} for k, v in fit_results_full.items()}
    node.namespace["_fit_results_full"] = fit_results_full
    node.results["fit_results"] = fit_results
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubits"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=fit_with_diag,
        show=False,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if node.outcomes.get(qubit.name) != "successful":
                continue

            fit_result = node.results["fit_results"][qubit.name]
            try:
                qubit.larmor_frequency = qubit.larmor_frequency + fit_result["freq_offset"]

            except ValueError as exc:
                logger.warning("%s: skipping state update — %s", qubit.name, exc)
                node.outcomes[qubit.name] = "failed"


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()
