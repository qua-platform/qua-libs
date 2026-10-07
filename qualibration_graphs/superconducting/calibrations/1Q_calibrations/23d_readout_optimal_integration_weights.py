# %% {Imports}
import time
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import (
    align,
    declare,
    declare_stream,
    fixed,
    for_,
    program,
    reset_if_phase,
    save,
    stream_processing,
)
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool
from qualibration_libs.core import BatchableList
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from calibration_utils.readout_drachma_common import (
    assign_core_labels,
    build_batch_groups,
)
from calibration_utils.readout_optimal_integration_weights import (
    Parameters,
    fit_raw_data,
    has_variance,
    log_fitted_results,
    plot_demod_comparison,
    plot_demod_spectrum,
    plot_normalization,
    plot_raw_data_with_fit,
    plot_variance,
    process_raw_dataset,
)
from qualibrate import QualibrationNode

# %% {Node initialisation}
description = """
        OPTIMAL READOUT INTEGRATION WEIGHTS (OPX1000 MW-FEM)
Measures the averaged complex ADC trace for the qubit in |g> and in |e>, demodulates it in software
and builds the time-dependent matched-filter weight W(t) = env_e(t) - env_g(t) that best separates
the two states on the MW-FEM dual_demod readout. The weight is normalized against the OPX fixed-point
overflow limits and chunked to the 4 ns weight grid. Optionally the IQ imbalance is folded into the
weight and the envelopes are low-pass filtered before the fit. Diagnostics give the predicted SNR gain
over the current constant weights.

Every shot streams a full 1 ns trace, so the node is limited by stream-processing bandwidth: keep
num_shots modest, or the OPX drops samples and the job returns an empty result.

A stock quam ReadoutPulse cannot hold a complex, time-varying weight, so update_state writes only the
weight amplitude (phase discarded). Review the diagnostics before relying on the update.

Details (weight construction, IQ imbalance, overflow bounds, data volume): calibration_utils/readout_optimal_integration_weights/README.md

Prerequisites:
    - Having calibrated the mixer or the Octave (nodes 01a or 01b).
    - Having calibrated the readout parameters (nodes 02a, 02b and/or 02c).
    - Having calibrated the qubit x180 pulse parameters (nodes 03a_qubit_spectroscopy.py and 04b_power_rabi.py).

State update:
    - q.resonator.operations["readout"].integration_weights: |W| as a list of (weight, 4 ns) pairs.
      Re-run 07_iq_blobs afterward: the existing threshold and integration_weights_angle no longer
      match the new weights.
"""

node = QualibrationNode[Parameters, Quam](
    name="23d_readout_optimal_integration_weights",
    description=description,
    parameters=Parameters(),
)

node.machine = Quam.load()


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


def _build_batch_program(node: QualibrationNode[Parameters, Quam], batch_qubits: dict):
    """QUA program for ONE batch of simultaneously measured qubits (`batch_qubits` maps global
    qubit index -> qubit). Variables/streams are declared only for the batch and named with the
    global index."""
    with program() as prog:
        n = declare(int)
        n_st = declare_stream()
        idx = list(batch_qubits)
        I_g = {i: declare(fixed) for i in idx}
        Q_g = {i: declare(fixed) for i in idx}
        I_e = {i: declare(fixed) for i in idx}
        Q_e = {i: declare(fixed) for i in idx}
        I_g_st = {i: declare_stream() for i in idx}
        Q_g_st = {i: declare_stream() for i in idx}
        I_e_st = {i: declare_stream() for i in idx}
        Q_e_st = {i: declare_stream() for i in idx}
        adc_g = {i: declare_stream(adc_trace=True) for i in idx}
        adc_e = {i: declare_stream(adc_trace=True) for i in idx}
        multiplexed_qubits = batch_qubits

        # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
        for qubit in multiplexed_qubits.values():
            node.machine.initialize_qpu(target=qubit)
        align()

        with for_(n, 0, n < node.parameters.num_shots, n + 1):
            save(n, n_st)

            # --- ground: no preparation (qubit in ground after reset) ---
            for i, qubit in multiplexed_qubits.items():
                qubit.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
            align()
            for i, qubit in multiplexed_qubits.items():
                # Reset the digital-oscillator phase so raw traces add coherently across shots.
                reset_if_phase(qubit.resonator.name)
                qubit.resonator.measure("readout", qua_vars=(I_g[i], Q_g[i]), stream=adc_g[i])
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
            for i, qubit in multiplexed_qubits.items():
                qubit.resonator.wait(qubit.resonator.depletion_time // 4)
            align()

            # --- excited: x180 right before the readout ---
            for i, qubit in multiplexed_qubits.items():
                qubit.reset(node.parameters.reset_type, node.parameters.simulate, log_callable=node.log)
            align()
            for i, qubit in multiplexed_qubits.items():
                qubit.xy.play("x180")
                qubit.align()
                reset_if_phase(qubit.resonator.name)
                qubit.resonator.measure("readout", qua_vars=(I_e[i], Q_e[i]), stream=adc_e[i])
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])
                qubit.resonator.wait(qubit.resonator.depletion_time // 4)
            align()  # for sticky twpa

        with stream_processing():
            n_st.save("n")
            for i, qubit in multiplexed_qubits.items():
                port = qubit.resonator.opx_input.port_id
                for state, adc_state, i_st, q_st in (
                    ("g", adc_g, I_g_st, Q_g_st),
                    ("e", adc_e, I_e_st, Q_e_st),
                ):
                    stream = adc_state[i].input1() if port == 1 else adc_state[i].input2()
                    s_i = stream.real()
                    s_q = stream.image()
                    # Every single-shot trace is streamed (shape (num_shots, T)); the per-sample
                    # mean and variance are computed on the PC in execute_qua_program. This
                    # replaces the OPX-side average and the second-moment stream, whose ADC-stream
                    # arithmetic returns x[t]*x[0] instead of x[t]**2.
                    s_i.save_all(f"adcI_{state}{i + 1}")
                    s_q.save_all(f"adcQ_{state}{i + 1}")
                    i_st[i].average().save(f"I_{state}{i + 1}")
                    q_st[i].average().save(f"Q_{state}{i + 1}")
    return prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the two-state (g/e) acquisition: for each qubit, prepare g (as-is after reset) and
    e (x180), and for each, `reset_if_phase` then `measure` the readout pulse with BOTH the raw
    ADC trace streamed (for our own demodulation) AND the ordinary dual-demod I/Q returned (using
    whatever weights are CURRENTLY active -- a free diagnostic, see analysis.py's
    `_demod_phase_offset`). No extra hardware time over a plain ADC-trace acquisition: it's the
    same `measure()` call either way.
    """
    all_qubits = get_qubits(node)

    # Each qubit's own length/smearing (per-pulse calibrated properties -- see the module
    # docstring). The captured window is per qubit; the shared readout_time axis used below for
    # the sweep/dataset is padded to the longest one.
    lengths_ns = [q.resonator.operations["readout"].length for q in all_qubits]
    smearings_ns = [q.resonator.smearing for q in all_qubits]
    captured_lengths_ns = [length + 2 * smearing for length, smearing in zip(lengths_ns, smearings_ns)]
    node.namespace["readout_length_ns_per_qubit"] = lengths_ns
    node.namespace["trace_window_offset_ns_per_qubit"] = smearings_ns
    captured_length_ns = max(captured_lengths_ns)

    # FEM-aware batching (reused from readout_depletion_validation): non-multiplexed runs give
    # each qubit its own batch; multiplexed runs cap simultaneous same-FEM qubits at
    # max_qubits_per_fem (default 1 -- see Parameters' docstring for why).
    batch_groups = build_batch_groups(all_qubits, node.parameters, max_per_fem=node.parameters.max_qubits_per_fem)
    node.log(
        f"Batching: {len(all_qubits)} qubits into {len(batch_groups)} batch(es) of sizes "
        f"{[len(b) for b in batch_groups]} (multiplexed={node.parameters.multiplexed})."
    )
    # Execution order: batch 0's shots all play before batch 1's (plain Python loop below).
    for batch_idx, group in enumerate(batch_groups):
        member_names = [all_qubits[i].name for i in group]
        node.log(f"  batch {batch_idx}: {member_names}")
        print(f"  batch {batch_idx}: {member_names}")

    # Share cores wherever safe (see assign_core_labels' own docstring) -- needed as soon as
    # more than one qubit per FEM can be simultaneously active.
    twpas = list(getattr(node.machine, "twpas", {}).values())
    core_labels, twpa_labels = assign_core_labels(all_qubits, twpas)
    for qubit, label in zip(all_qubits, core_labels):
        qubit.xy.core = qubit.resonator.core = label
    for twpa in twpas:
        label = twpa_labels[twpa.name]
        twpa.pump.core = label
        if twpa.pump_ is not None:
            twpa.pump_.core = label
    # Printed for user verification: qubits sharing a (FEM, core) label must never be in the
    # same batch, and each qubit's feedline input shows what shares one ADC capture.
    core_msg = {
        q.name: f"{label} (fem {q.resonator.opx_output.fem_id}, in port {q.resonator.opx_input.port_id})"
        for q, label in zip(all_qubits, core_labels)
    }
    node.log(f"Core assignment: {core_msg}")
    node.log(f"TWPA core assignment: {twpa_labels}")
    print("Core assignment (qubit: label):")
    for name, msg in core_msg.items():
        print(f"  {name}: {msg}")
    print(f"TWPA core assignment: {twpa_labels}")

    node.namespace["qubits"] = qubits = BatchableList(all_qubits, batch_groups)

    # dims= is mandatory here (unlike fetcher-based nodes): XarrayDataFetcher.preprocess_axes
    # renames the default "dim_0" dim to the dict key for you, but this node builds ds_raw by
    # hand (see execute_qua_program's docstring) so a bare, dim-less DataArray here would leave
    # BOTH coords as "dim_0" and collide when xr.Dataset merges them.
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names(), dims="qubit"),
        "readout_time": xr.DataArray(
            np.arange(captured_length_ns),
            dims="readout_time",
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    # One program (= one job, see execute_qua_program) per batch: each job then streams only its
    # own batch's raw ADC traces. Stream names keep the GLOBAL qubit index (i + 1), so the
    # fetch side needs no remapping.
    batches = list(qubits.batch())
    node.namespace["batch_global_indices"] = [list(b.keys()) for b in batches]
    node.namespace["qua_programs"] = [_build_batch_program(node, b) for b in batches]
    node.namespace["qua_program"] = node.namespace["qua_programs"][0]  # simulate: batch 0 only


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    _samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}


def _assert_handles_present(job, names, node: QualibrationNode[Parameters, Quam]) -> None:
    """Fail with a diagnosable message when the backend didn't return every stream we saved.

    `fetching_tool` raises a bare `Warning: <name> is not saved in the stream processing` for the
    first missing handle, which points at the stream processing even when the program saved it
    correctly: on the cloud backend a job that lost samples comes back with an EMPTY result
    dict (CloudResultHandles only sets attributes for handles it actually received), so even the
    scalar "n" disappears. Report the real cause and the two knobs that fix it instead.
    """
    handles = job.result_handles
    available = set(handles.keys()) if hasattr(handles, "keys") else {n for n in names if hasattr(handles, n)}
    missing = [name for name in names if name not in available]
    if not missing:
        return
    raise RuntimeError(
        f"The backend returned {len(available)} of the {len(names)} streams this node saved; "
        f"missing {len(missing)}, e.g. {missing[:5]}. If the run logged "
        '"Data loss detected in data for job ...", the OPX dropped samples because this node '
        "streams a full ADC trace per shot, per qubit and per state (each batch already runs as "
        f"its own job): lower num_shots (currently {node.parameters.num_shots}), lower "
        f"max_qubits_per_fem (currently {node.parameters.max_qubits_per_fem})."
    )


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute one QUA program (= one job) per batch, and fetch the raw data by hand (not via
    XarrayDataFetcher: each qubit's ADC trace has its own native length, and the fetcher requires
    every fetched handle to share one common shape). Every handle -- "n" plus every trace and
    hardware-demod diagnostic -- is fetched together, and the values are then assembled into a
    NaN-padded "qubit"-indexed dataset by hand.

    Every single-shot ADC trace is streamed (`save_all`, shape (num_shots, T)), so the handles are
    fetched ONCE after the job is done (polling would re-download the growing arrays each pass),
    then reduced per qubit and state to the per-sample first moment (adcI_*/adcQ_*) and second
    moment (adcI2_*/adcQ2_*), in raw ADC units -- the layout analysis.process_raw_dataset already
    expects. The single shots are dropped right after the reduction, so ds_raw stays small and peak
    memory is one batch. Volume, fetch time and reduction time are logged per batch.

    The cloud backend (`qm.execute` -> CloudQuantumMachine.execute) ships the program to a
    remote worker, blocks until the job is over, and returns one dict of already-collected results,
    so nothing here can prevent the OPX from dropping samples. The defence against "Data loss
    detected" is to keep the streamed volume small: see Parameters.num_shots.
    """
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    qubits = node.namespace["qubits"]
    num_qubits = len(qubits)
    lengths_ns = node.namespace["readout_length_ns_per_qubit"]
    offsets_ns = node.namespace["trace_window_offset_ns_per_qubit"]
    readout_time_size = node.namespace["sweep_axes"]["readout_time"].size
    programs = node.namespace["qua_programs"]
    batch_indices = node.namespace["batch_global_indices"]

    # Mirrors the saves in _build_batch_program's stream_processing: asking for a handle that was
    # never saved would fail the fetch outright.
    trace_prefixes = ("adcI_", "adcQ_")

    raw_arrays = {
        f"{prefix}{state}": np.full((num_qubits, readout_time_size), np.nan)
        for prefix in ("adcI_", "adcI2_", "adcQ_", "adcQ2_")
        for state in ("g", "e")
    }
    hw_values = {f"{prefix}{state}": np.full(num_qubits, np.nan) for prefix in ("I_", "Q_") for state in ("g", "e")}

    node.namespace["jobs"] = []
    for batch_idx, (program_b, idx) in enumerate(zip(programs, batch_indices)):
        batch_names = [qubits[i].name for i in idx]
        msg = f"Executing batch {batch_idx + 1}/{len(programs)}: {batch_names}"
        node.log(msg)
        print(msg)

        trace_names = [f"{prefix}{state}{i + 1}" for prefix in trace_prefixes for state in ("g", "e") for i in idx]
        hw_names = [f"{prefix}{state}{i + 1}" for prefix in ("I_", "Q_") for state in ("g", "e") for i in idx]
        all_names = ["n", *trace_names, *hw_names]

        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(program_b, terminal_output=True, options={"timeout": node.parameters.timeout})
            node.namespace["job"] = job
            node.namespace["jobs"].append(job)

            # Logged before fetching_tool below: a QUA runtime error, or a data loss that made the
            # backend return no results at all, otherwise surfaces only as fetching_tool's opaque
            # "<name> is not saved in the stream processing" and this report is never reached.
            node.log(job.execution_report())
            _assert_handles_present(job, all_names, node)

            t_fetch = time.perf_counter()
            results = fetching_tool(job, all_names, mode="wait_for_all")
            fetched = dict(zip(all_names, results.fetch_all()))
            fetch_s = time.perf_counter() - t_fetch

        # Reduce this batch's single shots to per-sample first/second moments (raw ADC units).
        t_reduce = time.perf_counter()
        n_bytes = 0
        for prefix in trace_prefixes:
            for state in ("g", "e"):
                for i in idx:
                    shots = np.asarray(fetched.pop(f"{prefix}{state}{i + 1}"), dtype=float)
                    n_bytes += shots.nbytes
                    # The backend returns (num_shots, 1, T) for save_all'd ADC traces: drop the singleton axis.
                    if shots.ndim == 3 and shots.shape[1] == 1:
                        shots = shots[:, 0, :]
                    needed = offsets_ns[i] + lengths_ns[i]
                    if shots.ndim != 2 or shots.shape[1] < needed:
                        raise RuntimeError(
                            f"Qubit {qubits[i].name}: fetched '{prefix}{state}{i + 1}' has shape "
                            f"{shots.shape}, expected (num_shots, >= {needed})."
                        )
                    if shots.shape[0] != node.parameters.num_shots:
                        node.log(
                            f"{qubits[i].name} {prefix}{state}: got {shots.shape[0]} shots, "
                            f"asked for {node.parameters.num_shots}."
                        )
                    raw_arrays[f"{prefix}{state}"][i, : shots.shape[1]] = shots.mean(axis=0)
                    raw_arrays[f"{prefix[:-1]}2_{state}"][i, : shots.shape[1]] = (shots * shots).mean(axis=0)
                    del shots
        reduce_s = time.perf_counter() - t_reduce
        msg = (
            f"Batch {batch_idx + 1}: fetched {n_bytes / 1e6:.1f} MB of single-shot traces in "
            f"{fetch_s:.1f} s, reduced in {reduce_s:.2f} s."
        )
        node.log(msg)
        print(msg)
        for base, values in hw_values.items():
            for i in idx:
                values[i] = float(fetched[f"{base}{i + 1}"])

    raw_vars = {base: (("qubit", "readout_time"), padded) for base, padded in raw_arrays.items()}
    for base, values in hw_values.items():
        raw_vars[base] = ("qubit", list(values))

    coords = dict(node.namespace["sweep_axes"])
    coords["readout_length_ns"] = ("qubit", lengths_ns)
    coords["trace_window_offset_ns"] = ("qubit", offsets_ns)
    node.results["ds_raw"] = xr.Dataset(raw_vars, coords=coords)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)
    # Per-qubit readout_length_ns/trace_window_offset_ns travel as coords on the saved ds_raw
    # (see execute_qua_program) -- nothing else needs restoring here.


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data: convert to volts and compute per-sample variance, then run the
    matched-filter construction (weights.py) per qubit."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    if node.parameters.smooth_bandwidth_hz is not None:
        node.log(
            f"env_g/env_e smoothing: zero-phase Hann low-pass at "
            f"{node.parameters.smooth_bandwidth_hz / 1e6:.1f} MHz applied before the weight fit."
        )
    else:
        node.log("env_g/env_e smoothing: off (Parameters.smooth_bandwidth_hz is None).")

    log_fitted_results(node.results["ds_fit"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the always-on set (predicted SNR, weight spectrum, demodulated envelopes, normalized
    weight, IQ trajectory), plus the extra debug plots when `node.parameters.debug_plots` is True
    (variance, before/after demodulation in time and frequency, normalization headroom)."""
    ds_fit = node.results["ds_fit"]
    qubits = node.namespace["qubits"]

    figures = plot_raw_data_with_fit(ds_fit, qubits)
    if node.parameters.debug_plots:
        if has_variance(ds_fit):
            figures["variance"] = plot_variance(ds_fit, qubits)
        figures["demod_comparison"] = plot_demod_comparison(ds_fit, qubits)
        figures["demod_spectrum"] = plot_demod_spectrum(ds_fit, qubits)
        figures["normalization"] = plot_normalization(ds_fit, qubits)

    plt.show()
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the AMPLITUDE of the computed weights onto the qubit's 'readout' operation as real
    (weight, 4 ns) pairs. The phase of the complex matched filter is discarded until complex
    integration weights are supported, so the deployed weight is only an approximation of the
    computed one. This invalidates the pulse's existing threshold and integration_weights_angle
    -- re-run 07_iq_blobs afterward for both.
    """
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit = node.results["ds_fit"].sel(qubit=q.name)
            ro = q.resonator.operations["readout"]

            # W_chunked is NaN-padded to the longest qubit's chunk count; keep only this pulse's own.
            w_amp = np.abs(fit.W_chunked.values)[: ro.length // 4]
            w_amp = w_amp[np.isfinite(w_amp)]
            if w_amp.size == 0:
                node.log(f"{q.name}: no finite weights; skipping.")
                continue

            iw = [(float(w), 4) for w in w_amp]
            total_len = 4 * len(iw)
            if total_len != ro.length:
                iw[-1] = (iw[-1][0], iw[-1][1] + (ro.length - total_len))

            # integration_weights is a QuAM reference by default; it must be cleared before overwriting.
            ro.integration_weights = None
            ro.integration_weights = iw
            node.log(f"{q.name}: wrote {len(iw)} integration weight segments.")


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
    # pass


# %%
