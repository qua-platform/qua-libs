"""MWE: zero-delta plays on sibling sticky channels corrupt plunger hold.

Three stages, each performing:
  empty(-0.1V, 100ns) → align → init(+0.05V, 40ns) → align → 15 XY pulses → align → measure(-0.05V, 200ns)

The init step is deliberately short (40ns) while 15 Gaussian XY pulses
take ~1500ns, so a correct sticky hold is clearly visible.

Stage 1  Play ONLY on plunger_1 (single sticky channel)
Stage 2  Play on plunger_1 AND 5 other sticky channels with amplitude_scale=0
         (mimics what VoltageSequence._common_voltages_change does)
Stage 3  Same as Stage 2 but XY drive is on an LF-FEM SingleChannel
         (IQ at 100 MHz IF) instead of the MW-FEM MWChannel

Expected: Stages 1-3 should all hold the plunger at +0.05V during XY.
Observed: Stage 2 (and possibly 3) corrupt the sticky hold.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_CONFIG = SCRIPT_DIR.parent / ".qm_cluster_config.json"

CONFTEST_DIR = SCRIPT_DIR / "qualibration_graphs" / "quantum_dots" / "calibrations" / "loss_divincenzo"
sys.path.insert(0, str(CONFTEST_DIR))

from quam_factory import create_minimal_quam  # type: ignore[import-not-found]

# Shared timing constants
EMPTY_DUR = 100  # ns
INIT_DUR = 40  # ns — deliberately << XY total
MEASURE_DUR = 200  # ns
NUM_PULSES = 15  # ~1500 ns of XY drive
SIM_DURATION = 3000  # clock cycles (12 µs)


def get_connection_params() -> dict:
    data = json.loads(Path(CLUSTER_CONFIG).read_text())
    return {"host": data["host"], "cluster_name": data["cluster_name"]}


def add_native_ops(machine):
    from quam.components import pulses as quam_pulses

    for qubit in machine.qubits.values():
        xy = qubit.xy
        ref = xy.operations.get("X180")
        if ref is None:
            continue
        if "x180" not in xy.operations:
            xy.operations["x180"] = quam_pulses.GaussianPulse(
                length=ref.length,
                amplitude=ref.amplitude,
                sigma=ref.sigma,
            )


# ── helpers ──────────────────────────────────────────────────────────────


def _plunger_timeline(job, plunger_key="4-1"):
    """Return (t_ns, voltage) arrays for the plunger channel."""
    samples = job.get_simulated_samples()
    analog = samples.con1.analog
    if plunger_key not in analog:
        return None, None
    data = np.real(np.array(analog[plunger_key]))
    t = np.arange(len(data))  # LF-FEM @ 1 GS/s → 1 ns/sample
    return t, data


def _xy_window(job, mw_key="1-1-1"):
    """Return (start_ns, end_ns) for the first contiguous XY pulse block."""
    samples = job.get_simulated_samples()
    analog = samples.con1.analog
    if mw_key not in analog:
        return None, None
    data = np.real(np.array(analog[mw_key]))
    dt = 2 if "-" in mw_key and len(mw_key.split("-")) == 3 else 1
    t = np.arange(len(data)) * dt
    nonzero = np.where(np.abs(data) > 0.001)[0]
    if nonzero.size == 0:
        return None, None
    start = t[nonzero[0]]
    end = start
    for idx in range(1, len(nonzero)):
        if t[nonzero[idx]] - t[nonzero[idx - 1]] > 40:
            break
        end = t[nonzero[idx]]
    return float(start), float(end)


def _analyse(job, label, mw_key="1-1-1", plunger_key="4-1"):
    """Print numeric timeline and return plunger mean during XY window."""
    t_pl, d_pl = _plunger_timeline(job, plunger_key)
    xy_start, xy_end = _xy_window(job, mw_key)
    if t_pl is None or xy_start is None:
        print(f"  {label}: could not extract traces")
        return float("nan")

    print(f"  {label}")
    print(f"    XY window: {xy_start:.0f} – {xy_end:.0f} ns ({xy_end - xy_start:.0f} ns)")

    print(f"    Plunger (100ns bins):")
    for t0 in range(0, 4000, 100):
        m = (t_pl >= t0) & (t_pl < t0 + 100)
        chunk = d_pl[m]
        if chunk.size > 0:
            print(f"      {t0:5d}-{t0+100:5d}: {np.mean(chunk):+.4f}")

    mask = (t_pl >= xy_start) & (t_pl <= xy_end)
    pl = d_pl[mask]
    mean = float(np.mean(pl)) if pl.size > 0 else float("nan")
    print(f"    Plunger during XY: mean={mean:+.4f}")
    return mean


# ── Stage 1: plunger only ───────────────────────────────────────────────


def stage1_plunger_only(machine):
    qubit = machine.qubits["Q1"]
    xy = qubit.xy
    plunger = machine.physical_channels["plunger_1"]

    with program() as prog:
        i = declare(int)
        reset_frame(xy.name)

        plunger.play("half_max_square", amplitude_scale=-0.1 / 0.25, duration=EMPTY_DUR // 4)
        align()
        plunger.play("half_max_square", amplitude_scale=0.15 / 0.25, duration=INIT_DUR // 4)
        align()
        with for_(i, 0, i < NUM_PULSES, i + 1):
            xy.play("x180")
        align()
        plunger.play("half_max_square", amplitude_scale=-0.10 / 0.25, duration=MEASURE_DUR // 4)
        align()

    return prog


# ── Stage 2: all 6 sticky channels (zero-delta on 5) ────────────────────


def stage2_all_channels(machine):
    qubit = machine.qubits["Q1"]
    xy = qubit.xy
    plunger = machine.physical_channels["plunger_1"]
    others = [
        machine.physical_channels[n] for n in ["plunger_2", "plunger_3", "plunger_4", "sensor_DC_1", "sensor_DC_2"]
    ]

    def _play_all(amp_plunger, dur_cc):
        plunger.play("half_max_square", amplitude_scale=amp_plunger, duration=dur_cc)
        for ch in others:
            ch.play("half_max_square", amplitude_scale=0.0, duration=dur_cc)

    with program() as prog:
        i = declare(int)
        reset_frame(xy.name)

        _play_all(-0.1 / 0.25, EMPTY_DUR // 4)
        align()
        _play_all(0.15 / 0.25, INIT_DUR // 4)
        align()
        with for_(i, 0, i < NUM_PULSES, i + 1):
            xy.play("x180")
        align()
        _play_all(-0.10 / 0.25, MEASURE_DUR // 4)
        align()

    return prog


# ── Stage 3: XY on LF-FEM (IQ at 100 MHz IF) ───────────────────────────


def _add_lf_xy_to_config(config: dict, fem_id: int = 4, port_i: int = 5, port_q: int = 6):
    """Manually inject an IQ element on LF-FEM ports into the QUA config."""
    import scipy.signal as sig

    IF_FREQ = 100_000_000  # 100 MHz
    LO_FREQ = 0
    length = 100
    sigma = length / 6
    amp = 0.2
    t = np.arange(length)
    gauss = amp * np.exp(-0.5 * ((t - length / 2) / sigma) ** 2)

    elem_name = "lf_xy_drive"

    # waveforms
    config["waveforms"]["lf_xy_gauss_wf"] = {"type": "arbitrary", "samples": gauss.tolist()}
    config["waveforms"]["lf_xy_zero_wf"] = {"type": "arbitrary", "samples": [0.0] * length}

    # pulses
    config["pulses"]["lf_xy_x180_pulse"] = {
        "operation": "control",
        "length": length,
        "waveforms": {"I": "lf_xy_gauss_wf", "Q": "lf_xy_zero_wf"},
    }

    # element
    config["elements"][elem_name] = {
        "mixInputs": {},
        "intermediate_frequency": IF_FREQ,
        "RF_inputs": {},
        "operations": {"x180": "lf_xy_x180_pulse"},
    }

    con = config["controllers"]["con1"]
    fem = con["fems"][fem_id]
    if "analog_outputs" not in fem:
        fem["analog_outputs"] = {}
    for p in [port_i, port_q]:
        fem["analog_outputs"][p] = {"offset": 0.0, "output_mode": "direct"}

    config["elements"][elem_name] = {
        "intermediate_frequency": IF_FREQ,
        "operations": {"x180": "lf_xy_x180_pulse"},
        "mixInputs": {
            "I": ("con1", fem_id, port_i),
            "Q": ("con1", fem_id, port_q),
            "lo_frequency": LO_FREQ,
        },
    }

    return elem_name


def stage3_lf_xy(machine):
    """XY drive on LF-FEM IQ channel on the SAME FEM as the plungers."""
    config = machine.generate_config()
    elem_name = _add_lf_xy_to_config(config)

    plunger = machine.physical_channels["plunger_1"]
    others = [
        machine.physical_channels[n] for n in ["plunger_2", "plunger_3", "plunger_4", "sensor_DC_1", "sensor_DC_2"]
    ]

    def _play_all(amp_plunger, dur_cc):
        plunger.play("half_max_square", amplitude_scale=amp_plunger, duration=dur_cc)
        for ch in others:
            ch.play("half_max_square", amplitude_scale=0.0, duration=dur_cc)

    with program() as prog:
        i = declare(int)
        reset_frame(elem_name)

        _play_all(-0.1 / 0.25, EMPTY_DUR // 4)
        align()
        _play_all(0.15 / 0.25, INIT_DUR // 4)
        align()
        with for_(i, 0, i < NUM_PULSES, i + 1):
            play("x180", elem_name)
        align()
        _play_all(-0.10 / 0.25, MEASURE_DUR // 4)
        align()

    return prog, config, f"{4}-{5}"


# ── main ─────────────────────────────────────────────────────────────────


def main():
    conn = get_connection_params()
    qmm = QuantumMachinesManager(**conn)

    stages = [
        ("Stage 1: plunger only (MW-FEM XY)", stage1_plunger_only, False),
        ("Stage 2: all 6 channels + zero-delta (MW-FEM XY)", stage2_all_channels, False),
        ("Stage 3: all 6 channels + zero-delta (LF-FEM XY)", stage3_lf_xy, True),
    ]

    fig, axes = plt.subplots(len(stages), 1, figsize=(16, 5 * len(stages)))
    fig.suptitle(
        f"Sticky zero-delta bug  (init={INIT_DUR}ns, XY≈{NUM_PULSES * 100}ns)",
        fontsize=14,
    )

    results = {}

    for idx, (label, builder, custom_config) in enumerate(stages):
        print(f"\n{'='*60}\n  {label}\n{'='*60}")
        ax = axes[idx]

        machine = create_minimal_quam()
        add_native_ops(machine)
        config = machine.generate_config()
        mw_key = "1-1-1"

        try:
            if custom_config:
                prog, config, mw_key = builder(machine)
            else:
                prog = builder(machine)
        except Exception as e:
            print(f"  BUILD FAILED: {e}")
            import traceback

            traceback.print_exc()
            ax.set_title(f"{label} — BUILD FAILED", color="red")
            continue

        try:
            job = qmm.simulate(config, prog, SimulationConfig(duration=SIM_DURATION))
        except Exception as e:
            print(f"  SIM FAILED: {e}")
            import traceback

            traceback.print_exc()
            ax.set_title(f"{label} — SIM FAILED", color="red")
            continue

        mean = _analyse(job, label, mw_key=mw_key)
        results[label] = mean

        # plot
        t_pl, d_pl = _plunger_timeline(job)
        xy_s, xy_e = _xy_window(job, mw_key)
        if t_pl is not None:
            ax.plot(t_pl, d_pl, color="blue", lw=1.2, label="plunger_1 (4-1)")
        samples = job.get_simulated_samples().con1.analog
        if mw_key in samples:
            d_mw = np.real(np.array(samples[mw_key]))
            dt = 2 if "-" in mw_key and len(mw_key.split("-")) == 3 else 1
            t_mw = np.arange(len(d_mw)) * dt
            ax.plot(t_mw, d_mw, color="orange", lw=0.4, alpha=0.8, label=f"XY ({mw_key})")
        if xy_s is not None:
            ax.axvspan(xy_s, xy_e, color="orange", alpha=0.07)
        ax.axhline(0.05, color="green", ls="--", lw=0.6, alpha=0.5)
        ax.set_xlim(0, 5000)
        ax.set_ylim(-0.15, 0.12)
        ok = "PASS" if abs(mean - 0.05) < 0.005 else "FAIL"
        ax.set_title(
            f"{label}  — plunger during XY: {mean:+.4f}V  [{ok}]", fontsize=10, color="green" if ok == "PASS" else "red"
        )
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel("V")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = Path("tests/simulation/artifacts/mwe_sticky_zero_delta.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nSaved → {out}")

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for label, mean in results.items():
        ok = "PASS" if abs(mean - 0.05) < 0.005 else "FAIL"
        print(f"  [{ok}]  {label}  →  plunger mean = {mean:+.4f}V")


if __name__ == "__main__":
    main()
