"""Probe what the CMA-ES orbit `opx_execute` time is spent doing.

The orbit node spends ~96% of its wall time in `opx_execute` (the OPX running
the per-generation QUA body). The timing node records exactly when this phase
starts and stops:

    after host push_to_input_stream calls complete
    before host fetch/score/CMA-ES tell begins

Therefore `opx_execute` includes input-stream advances, the candidate/qubit/
variant/circuit/shot loops, all init/gate/measure/ramp/save QUA work, stream
processing, and any real-time QUA control overhead.

This companion script runs a simplified single-qubit cumulative probe to expose
the dominant per-shot cost:

    V0  loop_only          : reset_frame + align + align + save        -> overhead
    V1  +init              : + qubit.initialize()                      -> init   = V1-V0
    V2  +gates             : + depth-m Clifford sequence               -> gates  = V2-V1
    V3  +measure (= full)  : + measure() + ramp_to_zero()              -> measure = V3-V2

The simplified probe is useful, but it is not a full production-program clone:
it does not include the population/circuit/qubit hierarchy, frequency updates,
input-stream advance/unblock instructions, or production stream-processing
shape. The script therefore reports both the simplified probe and the residual
against the measured production `opx_execute` time.

Run:  python calibrations/cmaes/cmaes_orbit_opx_breakdown.py
Uses the non-heralded init (qubit-level `Initialize1QMacro`, deterministic).
"""
import sys
import time

import numpy as np
from qm.qua import *

from quam_config import Quam
from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    NUM_CLIFFORDS,
    build_single_qubit_clifford_tables,
)

# ── Config of the realistic load we are explaining ──────────────────────────
PAIR = "q1_q2"
QUBIT_ROLE = "target"          # which qubit to profile (both are symmetric)
DEPTH = 30                     # orbit_depth in the realistic run
N_SHOTS = 8000                 # shots per variant (steady-state timing)
SEED = 42

# Per-generation event counts in the realistic run (pop 10, 2 qubits, 2
# variants, 20 circuits, 500 shots). Each "event" is one init+gates+measure body.
GEN_POP, GEN_QUBITS, GEN_VARIANTS, GEN_CIRCUITS, GEN_SHOTS = 10, 2, 2, 20, 500
EVENTS_PER_GEN = GEN_POP * GEN_QUBITS * GEN_VARIANTS * GEN_CIRCUITS * GEN_SHOTS
CLIFFORD_STEPS_PER_GEN = EVENTS_PER_GEN * DEPTH
PI_PREP_X180_PER_GEN = GEN_POP * GEN_QUBITS * GEN_CIRCUITS * GEN_SHOTS
FREQ_UPDATES_PER_GEN = 4 * GEN_POP
INPUT_STREAM_ADVANCES_PER_GEN = 6
MEASURED_OPX_PER_GEN = 125.836  # s/gen, from the realistic timing run


def log(*a):
    print(*a)
    sys.stdout.flush()


_TABLES = build_single_qubit_clifford_tables()


def _play_gate_scaled(qubit, gate_int, amplitude_scale, duration):
    with switch_(gate_int, unsafe=True):
        with case_(0):
            qubit.x90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(1):
            qubit.x180(amplitude_scale=amplitude_scale, duration=duration)
        with case_(2):
            qubit.x_neg90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(3):
            qubit.y90(amplitude_scale=amplitude_scale, duration=duration)
        with case_(4):
            qubit.y180(amplitude_scale=amplitude_scale, duration=duration)
        with case_(5):
            qubit.y_neg90(amplitude_scale=amplitude_scale, duration=duration)


def build_variant(qubit, level, cliffords_flat, amp_scale, dur):
    """Build an N_SHOTS program containing the per-shot body up to `level`.

    level: 0=loop_only, 1=+init, 2=+gates, 3=+measure(full).
    """
    with program() as prog:
        shot = declare(int)
        cliff_loop_idx = declare(int)
        gate_idx = declare(int)
        rand_clifford = declare(int)
        decomp_offset = declare(int)
        decomp_length = declare(int)
        current_gate = declare(int)
        state = declare(int)
        st = declare_output_stream()

        cliffords_qua = declare(int, value=cliffords_flat)
        decomp_qua = declare(int, value=_TABLES["decomp_flat"])
        offsets_qua = declare(int, value=_TABLES["decomp_offsets"])
        lengths_qua = declare(int, value=_TABLES["decomp_lengths"])

        with for_(shot, 0, shot < N_SHOTS, shot + 1):
            reset_frame(qubit.xy.name)
            align()
            if level >= 1:
                qubit.initialize()
                align()
            if level >= 2:
                with for_(cliff_loop_idx, 0, cliff_loop_idx < DEPTH, cliff_loop_idx + 1):
                    assign(rand_clifford, cliffords_qua[cliff_loop_idx])
                    assign(decomp_offset, offsets_qua[rand_clifford])
                    assign(decomp_length, lengths_qua[rand_clifford])
                    with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                        assign(current_gate, decomp_qua[decomp_offset + gate_idx])
                        _play_gate_scaled(qubit, current_gate, amp_scale, dur)
                align()
            if level >= 3:
                p = qubit.measure()
                align()
                qubit.voltage_sequence.ramp_to_zero()
                align()
                assign(state, Cast.to_int(p))
            else:
                assign(state, 0)
            save(state, st)

        with stream_processing():
            st.save_all("st")
    return prog


def time_per_shot(qm, prog):
    """Steady-state per-shot OPX time (excludes compile / first-shot startup)."""
    job = qm.execute(prog)
    h = job.result_handles.get("st")
    k1 = max(1, N_SHOTS // 10)   # skip first 10% as warm-up
    k2 = N_SHOTS
    while h.count_so_far() < k1:
        time.sleep(0.002)
    t1 = time.perf_counter()
    while h.count_so_far() < k2:
        time.sleep(0.002)
    t2 = time.perf_counter()
    job.cancel()
    return (t2 - t1) / (k2 - k1)


def main():
    m = Quam.load()
    pairs = m.qubit_pairs.values() if hasattr(m.qubit_pairs, "values") else m.qubit_pairs
    qp = next(p for p in pairs if getattr(p, "name", "") == PAIR)
    qubit = qp.qubit_target if QUBIT_ROLE == "target" else qp.qubit_control
    cal_dur = qubit.macros["x90"].pulse.length
    init_inferred_us = qubit.macros["initialize"].inferred_duration * 1e6
    measure_inferred_us = qubit.macros["measure"].inferred_duration * 1e6
    log(f"Profiling qubit '{qubit.name}' (role={QUBIT_ROLE}), x90 length={cal_dur} ns, "
        f"depth={DEPTH}, N_SHOTS={N_SHOTS}")
    log(f"Macro inferred durations: init={init_inferred_us:.2f} us "
        f"(known to undercount this balanced init), measure={measure_inferred_us:.2f} us")

    rng = np.random.default_rng(SEED)
    production_normal = rng.integers(
        0, NUM_CLIFFORDS, size=GEN_CIRCUITS * DEPTH
    ).tolist()
    production_pi = rng.integers(
        0, NUM_CLIFFORDS, size=GEN_CIRCUITS * DEPTH
    ).tolist()
    cliffords_flat = production_normal[:DEPTH]
    production_sequence_native = GEN_POP * GEN_QUBITS * GEN_SHOTS * int(
        np.asarray(_TABLES["decomp_lengths"])[production_normal].sum()
        + np.asarray(_TABLES["decomp_lengths"])[production_pi].sum()
    )
    production_total_native = production_sequence_native + PI_PREP_X180_PER_GEN
    mean_native = np.asarray(_TABLES["decomp_lengths"])[cliffords_flat].mean()
    n_gates = float(np.asarray(_TABLES["decomp_lengths"])[cliffords_flat].sum())
    log(f"This depth-{DEPTH} sequence: {n_gates:.0f} native gates "
        f"(mean {mean_native:.3f}/Clifford)")

    qmm = m.connect()
    qmm.close_all_quantum_machines()
    config = m.generate_config()
    qm = qmm.open_qm(config, close_other_machines=False)

    levels = {0: "loop_only", 1: "+init", 2: "+gates", 3: "+measure (full)"}
    t = {}
    try:
        for lvl, name in levels.items():
            prog = build_variant(qubit, lvl, cliffords_flat, 1.0, cal_dur)
            t[lvl] = time_per_shot(qm, prog)
            log(f"  V{lvl} {name:18s}: {t[lvl]*1e6:9.2f} us/shot")
    finally:
        qm.close()

    overhead = t[0]
    init = t[1] - t[0]
    gates = t[2] - t[1]
    measure = t[3] - t[2]
    full = t[3]
    production_per_body = MEASURED_OPX_PER_GEN / EVENTS_PER_GEN
    residual = production_per_body - full

    log("\n" + "=" * 64)
    log("SIMPLIFIED SINGLE-QUBIT BODY PROBE (us/shot)")
    log("=" * 64)
    rows = [
        ("loop/align overhead", overhead),
        ("initialize delta", init),
        (f"gate sequence (depth {DEPTH})", gates),
        ("measure + ramp_to_zero delta", measure),
        ("SIMPLIFIED body total", full),
    ]
    for label, v in rows:
        log(f"  {label:30s} {v*1e6:9.2f} us  {100*v/full:5.1f}%")
    log(f"  per native gate (~{n_gates:.0f}/shot): {gates/n_gates*1e6:.3f} us "
        f"(incl. switch_/lookup overhead)")
    if init <= 0 or measure <= 0:
        log("  NOTE: init/measure deltas are below this probe's timing resolution; "
            "use simulator/hardware spans for those macros.")

    # Scale to per generation and compare with the measured opx_execute.
    log("\n" + "=" * 64)
    log(f"PRODUCTION OPX_EXECUTE COMPARISON ({EVENTS_PER_GEN:,} shot bodies/gen)")
    log("=" * 64)
    scaled_full = full * EVENTS_PER_GEN
    for label, v in rows[:-1]:
        log(f"  {label:30s} {v*EVENTS_PER_GEN:8.1f} s  {100*v/full:5.1f}%")
    log(f"  {'SIMPLIFIED scaled body':30s} {scaled_full:8.1f} s")
    log(f"  {'MEASURED production opx':30s} {MEASURED_OPX_PER_GEN:8.1f} s/gen "
        f"(ratio {scaled_full/MEASURED_OPX_PER_GEN:.2f})")
    log(f"  {'production per body':30s} {production_per_body*1e6:8.2f} us")
    log(f"  {'production residual/body':30s} {residual*1e6:8.2f} us "
        f"({100*residual/production_per_body:4.1f}% of measured production)")

    log("\n" + "=" * 64)
    log("EXACT PRODUCTION WORK INCLUDED IN ONE OPX_EXECUTE GENERATION")
    log("=" * 64)
    log(f"  shot bodies                 {EVENTS_PER_GEN:,}")
    log(f"  initializes                 {EVENTS_PER_GEN:,}")
    log(f"  measurements                {EVENTS_PER_GEN:,}")
    log(f"  ramp_to_zero calls          {EVENTS_PER_GEN:,}")
    log(f"  reset_frame calls           {EVENTS_PER_GEN:,}")
    log(f"  stream saves                {EVENTS_PER_GEN:,}")
    log(f"  Clifford loop iterations    {CLIFFORD_STEPS_PER_GEN:,}")
    log(f"  native XY Clifford gates    {production_sequence_native:,}")
    log(f"  pi-prep x180 gates          {PI_PREP_X180_PER_GEN:,}")
    log(f"  total native XY gates       {production_total_native:,}")
    log(f"  frequency updates           {FREQ_UPDATES_PER_GEN:,}")
    log(f"  input-stream advances       {INPUT_STREAM_ADVANCES_PER_GEN:,}")
    log("  plus stream processing into survival_target/survival_control")
    log("\nNote: the pi-variant adds one extra x180 (~1 us) on half the shots; "
        "control qubit durations are symmetric (same inferred macro lengths). "
        "The current non-heralded balanced initialize simulated as an 8.4 us "
        "waveform span even though inferred_duration reports 4.416 us.")


if __name__ == "__main__":
    main()
