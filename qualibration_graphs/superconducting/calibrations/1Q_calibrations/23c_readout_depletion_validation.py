# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import (
    align,
    declare,
    declare_output_stream,
    fixed,
    for_,
    program,
    reset_if_phase,
    save,
    stream_processing,
)
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibration_libs.core import BatchableList
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam.components.pulses import SquareReadoutPulse
from quam_config import Quam

from calibration_utils.readout_depletion_validation import (
    Parameters,
    compute_ge_depletion_time,
    compute_stat_depletion_time,
    fetch_sliced_iq_traces,
    log_depletion_summary,
    plot_drachma_residuals,
    plot_ge_pvalue_grid,
    plot_pvalue_grid,
    process_raw_dataset,
    resolve_conditions,
)
from calibration_utils.readout_drachma_common import (
    assign_core_labels,
    build_batch_groups,
)
from qualibrate import QualibrationNode

# %% {Description}
description = """
        RESONATOR DRACHMA READOUT - RESIDUAL PHOTON NUMBER DIAGNOSTIC
Checks how many photons are left in the readout resonator right after a DRACHMA readout pulse
(Jerger et al., arXiv:2406.04891), whose shaped waveform is designed to ring the cavity down to
vacuum without separate depletion segments. For each condition (DRACHMA pulse, no operation as a
vacuum reference, optionally a regular readout pulse) and for both |g> and |e>, a zero-amplitude
probe pulse is played immediately afterwards and its sliced dual-demodulation result is streamed
over the probe window. Comparing the residual |IQ| vs time with the no-operation trace, and the
ground vs excited traces with each other, gives the depletion time of the resonator.

Details (probe pulse, sliced demodulation, statistical tests): calibration_utils/readout_depletion_validation/README.md

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b) and the qubit x180 pulse (nodes 03a, 04b).
    - A "readout_drachma" operation (DrachmaReadoutPulse) on qubit.resonator.operations. Qubits
      without it are skipped.

State update:
    - qubit.resonator.depletion_time: the ground-vs-excited depletion time of the DRACHMA condition
      (at least min_depletion_time_ns). A qubit not depleted within the probe window gets a warning
      and probe_length as its depletion time.
"""


# %% {Initialisation}
node = QualibrationNode[Parameters, Quam](
    name="23c_readout_depletion_validation",
    description=description,
    parameters=Parameters(),
)

node.machine = Quam.load()

STATES = ("ground", "excited")
"""The two qubit states each condition is tested in: |g> (as-is after reset) and |e> (an
x180 pulse played right before the condition's test pulse)."""

PROBE_OPERATION = "readout_probe"
"""Name of the temporary zero-amplitude SquareReadoutPulse (length probe_length) built per qubit
in create_qua_program and removed again in save_results before node.save()
persists machine state -- it must never leak into the shared QuAM state."""


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["qB1", "qB2", "qB3"]
    node.parameters.num_shots = 1000
    # node.parameters.load_data_id = 255
    node.parameters.multiplexed = True
    node.parameters.timeout = 300
    node.parameters.include_readout_baseline = False


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the per-qubit DRACHMA pulse, then for each (condition, state) pair -- condition
    in conditions (drachma / no_operation, plus readout if include_readout_baseline), state
    in (ground / excited) -- prepare the qubit, play the test pulse, and stream the sliced
    dual-demodulation result of the zero-amplitude residual-photon probe that immediately
    follows it."""
    all_qubits = get_qubits(node)
    operation = node.parameters.operation
    drachma_operation = node.parameters.drachma_operation
    probe_length = node.parameters.probe_length
    segment_length_ns = node.parameters.segment_length_ns
    # measure_sliced needs the probe length to be an exact multiple of the segment length, and
    # segments are whole clock cycles. Only enforced by the QM compiler at execution time, so
    # check here instead of failing on hardware with a per-port error list.
    if segment_length_ns <= 0 or segment_length_ns % 4 != 0 or probe_length % segment_length_ns != 0:
        raise ValueError(
            f"segment_length_ns={segment_length_ns} must be a positive multiple of 4 that divides "
            f"probe_length={probe_length}."
        )
    num_segments = probe_length // segment_length_ns
    node.namespace["num_segments"] = num_segments
    depletion_time = node.parameters.depletion_time
    if depletion_time <= 0 or depletion_time % 4 != 0:
        raise ValueError(f"depletion_time={depletion_time} must be a positive multiple of 4.")

    conditions, ge_test_conditions = resolve_conditions(node.parameters)
    node.namespace["conditions"] = conditions
    node.namespace["ge_test_conditions"] = ge_test_conditions
    states = STATES

    # This node does not build the DRACHMA pulse itself -- qubit.resonator.operations[drachma_operation]
    # must already exist (a DrachmaReadoutPulse). Qubits missing it
    # are excluded from the run rather than failing the whole node -- e.g. in multiplexed mode
    # a single qubit playing an undefined operation aborts the shared real-time program for
    # every qubit, silently dropping ALL streamed results (not just that qubit's).
    kept_qubits = [q for q in all_qubits if q.resonator.operations.get(drachma_operation)]
    skipped_names = [q.name for q in all_qubits if q not in kept_qubits]
    if skipped_names:
        message = f"Skipping qubits without a pre-built '{drachma_operation}' operation: {skipped_names}."
        node.log(message)
        print(message)
    if not kept_qubits:
        raise RuntimeError(f"No qubits have a pre-built '{drachma_operation}' operation.")

    # Dedicated zero-amplitude probe pulse, same length for every qubit. Removed again in
    # save_results before node.save() persists machine state.
    for q in kept_qubits:
        q.resonator.operations[PROBE_OPERATION] = SquareReadoutPulse(length=probe_length, amplitude=0.0)

    batch_groups = build_batch_groups(kept_qubits, node.parameters)
    node.log(
        f"DRACHMA batching: {len(kept_qubits)} qubits into {len(batch_groups)} "
        f"batch(es) of sizes {[len(b) for b in batch_groups]} "
        f"(multiplexed={node.parameters.multiplexed})."
    )
    # Batches are consumed in this order by the `for multiplexed_qubits in qubits.batch():`
    # loop below (a plain Python loop, unrolled at QUA-compile time -- not a QUA-level
    # parallel construct), so this listing is also the real-time execution order: batch 0's
    # shots all play before batch 1's, etc.
    for batch_idx, group in enumerate(batch_groups):
        member_names = [kept_qubits[i].name for i in group]
        node.log(f"  batch {batch_idx}: {member_names}")
        print(f"  batch {batch_idx}: {member_names}")

    # A qubit's xy and resonator never play simultaneously, and qubits on the same FEM in
    # different batches never overlap in time either -- share cores wherever safe so config
    # generation doesn't try to allocate a dedicated core per qubit per FEM (which can exceed
    # the FEM's physical core count once enough qubits are assigned to it). Each TWPA's pump
    # runs continuously for the whole program, so it gets its own core reserved from qubit reuse.
    twpas = list(node.machine.twpas.values())
    core_labels, twpa_labels = assign_core_labels(kept_qubits, twpas)
    for qubit, label in zip(kept_qubits, core_labels):
        qubit.xy.core = qubit.resonator.core = label
    for twpa in twpas:
        label = twpa_labels[twpa.name]
        twpa.pump.core = label
        if twpa.pump_ is not None:
            twpa.pump_.core = label
    node.log(f"Core assignment: {dict(zip((q.name for q in kept_qubits), core_labels))}")
    node.log(f"TWPA core assignment: {twpa_labels}")
    node.namespace["qubits"] = qubits = BatchableList(kept_qubits, batch_groups)
    num_qubits = len(qubits)

    # Every qubit has the same segment count and length; kept as a per-qubit coord for analysis.
    segment_length_arr = np.full(num_qubits, segment_length_ns)
    node.namespace["segment_length_ns"] = segment_length_arr
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names(), dims="qubit"),
        "segment": xr.DataArray(
            np.arange(num_segments),
            dims="segment",
            attrs={"long_name": "probe segment index"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()

        # Dummy I/Q pair reused for the test-pulse measurement (drachma/readout) whose result
        # is never read -- declared once and passed as qua_vars to every rr.measure() call
        # below instead of letting it declare (and discard) a fresh pair each time.
        I_dummy = declare(fixed)
        Q_dummy = declare(fixed)

        # Four sliced dual-demod streams (II, IQ, QI, QQ) per (condition, state, qubit); kept
        # separate rather than combined on-chip into I/Q, so the combination happens in
        # analysis.py instead of adding a QUA for_ loop mid-shot (which could introduce timing
        # gaps in the experiment).
        slice_st = {
            c: {
                s: [{key: declare_output_stream() for key in ("II", "IQ", "QI", "QQ")} for _ in range(num_qubits)]
                for s in states
            }
            for c in conditions
        }

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                for condition in conditions:
                    for state in states:
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                            align()
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            # Prepare |e> right before the test pulse -- same loop iteration
                            # as its use. Qubits have finite T1 (the excited state is not
                            # persistent), so state prep and its use must not be separated
                            # across a loop/qubit boundary; matches 07_iq_blobs.py's
                            # reset -> align -> [x180 -> align -> measure] convention.
                            if state == "excited":
                                qubit.xy.play("x180")
                                qubit.align()
                            # Play the test pulse to populate the cavity (nothing for no_operation).
                            reset_if_phase(rr.name)
                            if condition == "drachma":
                                rr.measure(drachma_operation, qua_vars=(I_dummy, Q_dummy))
                            elif condition == "readout":
                                rr.measure(operation, qua_vars=(I_dummy, Q_dummy))
                            # Probe the residual cavity field: no drive (PROBE_OPERATION's own
                            # amplitude is 0), sliced dual demod instead of a full raw-ADC-trace
                            # stream. Always PROBE_OPERATION, regardless of condition -- the
                            # probe must be identical across conditions for them to be directly
                            # comparable.
                            reset_if_phase(rr.name)
                            II, IQ, QI, QQ = rr.measure_sliced(PROBE_OPERATION, num_segments=num_segments)
                            # num_segments is a plain Python int (derived from Parameters, not a QUA
                            # variable), so this loop unrolls at compile time into num_segments
                            # sequential save() calls -- no QUA-level control flow, no added
                            # timing gap versus a single array save.
                            for seg in range(num_segments):
                                save(II[seg], slice_st[condition][state][i]["II"])
                                save(IQ[seg], slice_st[condition][state][i]["IQ"])
                                save(QI[seg], slice_st[condition][state][i]["QI"])
                                save(QQ[seg], slice_st[condition][state][i]["QQ"])
                            rr.wait(depletion_time // 4)
                        align()

        with stream_processing():
            n_st.save("n")
            for condition in conditions:
                for state in states:
                    for i, qubit in enumerate(qubits):
                        for key in ("II", "IQ", "QI", "QQ"):
                            slice_st[condition][state][i][key].buffer(num_segments).average().save(
                                f"{key}_{condition}_{state}{i + 1}"
                            )
                        # Second moments of I = II+IQ and Q = QI+QQ (combined per shot, before
                        # averaging) -- gives Var(I), Var(Q) downstream for the
                        # residual-amplitude shot-noise std. Mirrors 08a's I_g_sq/Q_g_sq
                        # stream-arithmetic pattern. The I*Q cross moment (Cov(I,Q)) is
                        # intentionally not streamed -- see README.md for why it was measured
                        # and dropped as negligible.
                        I_stream = slice_st[condition][state][i]["II"] + slice_st[condition][state][i]["IQ"]
                        Q_stream = slice_st[condition][state][i]["QI"] + slice_st[condition][state][i]["QQ"]
                        (I_stream * I_stream).buffer(num_segments).average().save(f"I_sq_{condition}_{state}{i + 1}")
                        (Q_stream * Q_stream).buffer(num_segments).average().save(f"Q_sq_{condition}_{state}{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(
            node.namespace["qua_program"], terminal_output=True, options={"timeout": node.parameters.timeout}
        )

        # Logged before fetching_tool below: if the real-time program hits a runtime error
        # (e.g. a timing violation), the "n" stream can come back empty and fetching_tool
        # raises before ever reaching the execution_report() call further down -- surfacing
        # the QUA runtime error here first makes that failure mode diagnosable.
        node.log(job.execution_report())
        # Only track the scalar "n" progress stream here; the II/IQ/QI/QQ sliced-demod
        # streams are fetched directly via job.result_handles below instead of through
        # XarrayDataFetcher/fetching_tool.
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            progress_counter(results.fetch_all()[0], node.parameters.num_shots)

        raw_traces = fetch_sliced_iq_traces(
            job,
            node.namespace["qubits"],
            node.namespace["conditions"],
            STATES,
            node.namespace["num_segments"],
        )
    coords = dict(node.namespace["sweep_axes"])
    coords["segment_length_ns"] = ("qubit", node.namespace["segment_length_ns"])
    node.results["ds_raw"] = xr.Dataset(
        {name: (("qubit", "segment"), arr) for name, arr in raw_traces.items()},
        coords=coords,
    )


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    # alpha/depletion_debounce_segments are post-hoc analysis knobs (re-analysing the same
    # loaded dataset with a different threshold/debounce), not acquisition parameters -- unlike
    # num_shots/multiplexed/include_readout_baseline, they must survive load_from_id below
    # overwriting node.parameters with the historical run's saved values.
    alpha = node.parameters.alpha
    depletion_debounce_segments = node.parameters.depletion_debounce_segments
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.parameters.alpha = alpha
    node.parameters.depletion_debounce_segments = depletion_debounce_segments
    node.namespace["qubits"] = get_qubits(node)
    # Resolved from the just-restored (historical) parameters, not the CURRENT default --
    # otherwise a replay of a run acquired with a different include_readout_baseline setting
    # would silently drop or KeyError on the "readout" condition. See resolve_conditions.
    conditions, ge_test_conditions = resolve_conditions(node.parameters)
    node.namespace["conditions"] = conditions
    node.namespace["ge_test_conditions"] = ge_test_conditions


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Convert the sliced dual-demod counts to volts, stack the per-(condition, state) streams into "condition" and "state" dimensions, compute the residual-field amplitude |IQ| vs segment and its shot-noise std, run the statistical (chi-squared) depletion-time test, and run a ground-vs-excited distinguishability test as a further independent check."""
    conditions = node.namespace["conditions"]
    ge_test_conditions = node.namespace["ge_test_conditions"]

    ds = process_raw_dataset(node.results["ds_raw"], conditions, STATES)

    test_conditions = tuple(c for c in conditions if c != "no_operation")
    p_value, t_dep_stat_ns = compute_stat_depletion_time(ds, node, test_conditions)
    ds = ds.assign(p_value=p_value, depletion_time_stat_ns=t_dep_stat_ns)

    p_value_ge, t_dep_ge_ns = compute_ge_depletion_time(ds, node, ge_test_conditions)
    ds = ds.assign(p_value_ge=p_value_ge, depletion_time_ge_ns=t_dep_ge_ns)

    node.results["ds_fit"] = ds
    node.outcomes = {q: "successful" for q in ds.qubit.values}

    log_depletion_summary(ds, t_dep_stat_ns, t_dep_ge_ns, test_conditions, ge_test_conditions)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the residual photon amplitude |IQ| (µV) vs probe segment, with ±std (shot noise)
    error bars, one subplot per qubit -- 3 traces per subplot (drachma+ground, drachma+excited,
    no_operation+ground baseline). Also plot the statistical depletion-time test's p-value vs
    segment, one subplot per qubit with ground/excited overlaid, for drachma (and readout, if
    enabled) vs no_operation. Also plot the ground-vs-excited distinguishability test's p-value
    vs segment, one subplot per qubit with a single trace (drachma only)."""
    ds = node.results["ds_fit"]
    conditions = node.namespace["conditions"]
    ge_test_conditions = node.namespace["ge_test_conditions"]

    fig = plot_drachma_residuals(ds, node.namespace["qubits"], conditions, STATES)

    test_conditions = tuple(c for c in conditions if c != "no_operation")
    fig_pvalue = plot_pvalue_grid(
        ds,
        node.namespace["qubits"],
        test_conditions,
        STATES,
        node.parameters.alpha,
        t_dep_stat_ns=ds["depletion_time_stat_ns"],
    )

    fig_ge_pvalue = plot_ge_pvalue_grid(
        ds,
        node.namespace["qubits"],
        ge_test_conditions,
        node.parameters.alpha,
        t_dep_ge_ns=ds["depletion_time_ge_ns"],
    )

    node.results["figures"] = {
        "drachma_residual": fig,
        "drachma_pvalue": fig_pvalue,
        "drachma_ge_pvalue": fig_ge_pvalue,
    }
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the ground-vs-excited depletion time (drachma vs. no reference needed) into
    qubit.resonator.depletion_time (ns), clamped to at least min_depletion_time_ns. If a qubit
    never reached depletion within the probe window (NaN), log a warning and fall back to the
    probe pulse length instead."""
    ds = node.results["ds_fit"]
    ge_condition = node.namespace["ge_test_conditions"][0]
    t_dep_ge_ns = ds["depletion_time_ge_ns"].sel(test_condition=ge_condition)
    min_t_ns = node.parameters.min_depletion_time_ns

    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            t_ns = float(t_dep_ge_ns.sel(qubit=q.name))
            if np.isnan(t_ns):
                t_ns = node.parameters.probe_length
                message = (
                    f"{q.name}: not depleted within the measured window -- setting "
                    f"resonator.depletion_time to the probe length ({t_ns} ns)."
                )
                node.log(message, level="warning")
                print(message)
            measured_t_ns = t_ns
            t_ns = max(round(t_ns), min_t_ns)
            if t_ns != round(measured_t_ns):
                message = (
                    f"{q.name}: measured ge depletion time {measured_t_ns:.0f} ns below "
                    f"minimum -- setting resonator.depletion_time to {t_ns} ns."
                )
                node.log(message)
                print(message)
            q.resonator.depletion_time = t_ns


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    # Core assignment (create_qua_program) is scoped to this run's selected qubits, not a
    # calibration result -- clear it before node.save() persists node.machine so it doesn't
    # leak into the shared QuAM state used by every other node/graph. Likewise PROBE_OPERATION
    # (the temporary zero-amplitude probe pulse built in create_qua_program) -- it must never
    # leak into the shared QuAM state either.
    for qubit in node.namespace["qubits"]:
        qubit.xy.core = None
        qubit.resonator.core = None
        qubit.resonator.operations.pop(PROBE_OPERATION, None)
    for twpa in node.machine.twpas.values():
        twpa.pump.core = None
        if twpa.pump_ is not None:
            twpa.pump_.core = None
    node.save()


# %%
