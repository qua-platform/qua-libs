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
from calibration_utils.ramsey_chevron import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    process_raw_dataset,
)
from qualibration_libs.parameters.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.sweep import get_idle_times_in_clock_cycles

# %% {Node initialization}
description = """
        RAMSEY CHEVRON
This sequence performs a Ramsey measurement to characterize the qubit detuning and idle time.
The measurement involves sweeping the detuning frequency of the qubit, and performing a sequence of
two π/2 rotations with a swept idle time in between to create a 2D measurement. PSB is used to measure the
resulting state.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X90 pulse amplitude and frequency).

Datasets:
    - ``ds_raw``: untouched ``state`` stream fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed 2D sweeps plus analysis outputs. Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict. Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the chevron analysis passed the node criteria.
    - ``freq_offset`` [Hz]: fitted resonance detuning.
    - ``t2_star`` [ns]: fitted Ramsey dephasing time.
    - ``decay_rate`` [1 / ns]: fitted exponential decay component.
    - ``gauss_decay_rate`` [1 / ns]: fitted Gaussian decay component.

Figures (``node.results["figures"]``):
    - ``"raw_data_with_fit"``: Ramsey chevron heatmap plus mean-signal resonance diagnostics.

State update:
    - The qubit Larmor frequency.
    - The qubit  T2* (Ramsey) time.
"""


node = QualibrationNode[Parameters, Quam](
    name="11c_ramsey_chevron",
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
    """Build the 2D detuning × idle-time Ramsey chevron and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    u = unit(coerce_to_integer=True)

    # Select which qubits participate in this calibration
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    n_avg = node.parameters.num_shots  # repetitions averaged at each (detuning, tau) point

    # Idle time sweep (in clock cycles of 4ns)
    tau_values = get_idle_times_in_clock_cycles(node.parameters)

    # Construct the array of frequency detunings
    detuning_values = np.arange(
        -node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_span_in_mhz / 2 * u.MHz,
        node.parameters.detuning_step_in_mhz * u.MHz,
    )

    # Register the sweep axes to be added to the dataset when fetching data.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            detuning_values,
            attrs={"long_name": "frequency detuning", "units": "Hz"},
        ),
        "tau": xr.DataArray(
            tau_values * 4,
            attrs={"long_name": "idle time", "units": "ns"},
        ),
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

                    # ── INNER LOOP: sweep idle times t ───────────────────────
                    with for_each_(t, tau_values):

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

                        # Return gate voltages to zero before the next shot to avoid accumulation of fixed point errors
                        align()
                        qubit.voltage_sequence.ramp_to_zero()

            # Restore the calibrated IF after the detuning sweep.
            qubit.xy.update_frequency(intermediate_frequency)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                # Save order: for each detuning, sweep all idle-time values.
                # .buffer(len(tau_values))      → inner axis = tau
                # .buffer(len(detuning_values)) → outer axis = detuning
                # .average()                    → average over shots
                # Result: 2D state vs (detuning, tau) per qubit
                state_st[i].buffer(len(tau_values)).buffer(len(detuning_values)).average().save(f"state{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        # "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
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
        node.log(job.execution_report())
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
    """Analyse the raw data to extract frequency offset and T2*."""
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
    """Plot processed Ramsey-chevron data and fit overlays; store figures in ``node.results["figures"]``."""
    fit_with_diag = node.namespace.get("_fit_results_full", node.results.get("fit_results", {}))
    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["qubits"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=fit_with_diag,
    )
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.larmor_frequency = qubit.larmor_frequency + fit_result["freq_offset"]
            qubit.T2ramsey = fit_result["t2_star"] * 1e-9


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist the node results and any recorded state updates."""
    node.save()
