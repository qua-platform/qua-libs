"""CMA-ES single-qubit gate optimisation using orbit separation scoring.

TIMING / PROFILING VARIANT
--------------------------
This is an instrumented copy of ``03b_cmaes_orbit`` that adds a full
phase-resolved timing analysis of the host ↔ OPX optimisation loop.  The
CMA-ES iteration (normally delegated to ``run_cmaes_optimization``) is
inlined here so each phase can be timed independently:

    Per-iteration (per generation) phases
        cmaes_ask        CMA-ES sampling of the candidate population
        push             host → OPX: streaming the candidate parameters in
        opx_execute      OPX runs the queries (host waits for results)
        fetch            OPX → host: retrieving the survival probabilities
        score_compute    host-side numpy scoring of the population
        cmaes_tell       CMA-ES covariance / step-size update

    One-time (setup) phases
        connect          opening the QMM connection
        generate_config  building the QUA config dict
        build_program    constructing the QUA program AST (host)
        session_open     entering the ``qm_session`` context
        compile_upload   ``qm.execute`` — compile + upload + job start

The communication time is ``push + fetch``; the compile time is
``compile_upload``; the execute time is ``opx_execute``; the CMA-ES
calculation time is ``cmaes_ask + cmaes_tell``.  Each is reported per
iteration and as a percentage of the total wall-clock run time, and saved
to ``node.results["timing"]`` plus a summary figure.

Overview
--------
This node optimises the x90 pulse amplitude, duration, and drive frequency
independently for each qubit in a pair, maximising an orbit separation score
that quantifies gate quality without requiring a multi-depth RB sweep.

For each qubit in the pair, two survival probabilities are measured at a
single fixed Clifford depth:

    P_normal:  initialize |0⟩ → random Clifford sequence → measure
    P_pi:      initialize |0⟩ → π-pulse → random Clifford sequence → measure

Each variant uses independent random circuit instances to avoid correlations.
The score is maximised separation:

    score_qubit = P_normal − P_pi

For perfect single-qubit gates this equals p^m where p = 2F − 1 is the
depolarising parameter and m is the orbit depth.  The per-pair score is:

    score = (score_target + score_control) / 2

Optimal orbit depth
-------------------
The depth is chosen to maximise the Fisher information of the orbit score
with respect to gate fidelity.  The sensitivity condition:

    1 + m·ln(p) = p^(2m)

yields the closed-form approximation (valid for small r = 1−p):

    m* ≈ 0.8 / (2·(1 − F))

For the target fidelity F = 99% this gives m* = 40, at which the expected
score is p^40 = 0.98^40 ≈ 0.45 — well above the noise floor while
retaining maximum gradient for CMA-ES to exploit.

Algorithm
---------
CMA-ES (Covariance Matrix Adaptation Evolution Strategy) maintains a
multivariate Gaussian distribution over six parameters:

    θ = [amplitude_scale_target, duration_offset_target, freq_detuning_target,
         amplitude_scale_control, duration_offset_control, freq_detuning_control]

At each generation, CMA-ES:

1. **Samples** ``population_size`` candidate parameter vectors.
2. **Evaluates** all candidates in a single QUA program execution.
3. **Updates** μ, σ, and C via the CMA-ES adaptation rules.

QUA program architecture
------------------------
For each candidate and each qubit:

    for circuit in range(num_circuits):
        for shot in range(num_shots):
            — Variant 1 (normal): initialize → play sequence → measure
            — Variant 2 (pi):     initialize → x180 → play sequence → measure

Stream processing produces arrays of shape ``(pop_size, 2)`` per qubit per
generation, where the two columns are [P_normal, P_pi].

Prerequisites:
    - Calibrated x90 and x180 pulse parameters (amplitude, duration).
    - Calibrated initialization, measurement, and PSB threshold.
    - Native gate operations registered on qubit.xy channel.

State update:
    - Updates the x90/x180 pulse amplitude and duration via the
      XYDriveMacro.update() mechanism independently on each qubit.
    - The frequency detuning is optimised but not applied to the qubit
      state; the original intermediate frequency is preserved.
"""

# %% {Imports}
import time
import warnings
from collections import defaultdict
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np

from qm.qua import *

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from qualibrate.core.models.outcome import Outcome
from quam_config import Quam

from calibration_utils.cmaes import (
    OptimizationResult,
    run_cmaes_optimization,
    analyse_optimization,
    log_optimization_results,
    plot_parameter_evolution,
    plot_score_convergence_on_ax,
)
from calibration_utils.cmaes.cmaes_orbit_parameters import CMAESOrbitParameters
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_qubit_pairs
from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    NUM_CLIFFORDS,
    build_single_qubit_clifford_tables,
)
from qualibration_libs.runtime import simulate_and_plot


# %% {Node initialisation}
description = """
        CMA-ES GATE OPTIMISATION — ORBIT SEPARATION (QUBIT-PAIR, PER-QUBIT PARAMS)
Uses CMA-ES to optimise single-qubit gate parameters (x90 amplitude scale,
duration offset, and frequency detuning) independently for each qubit in a
pair, maximising the orbit separation score.

The orbit score measures the difference in survival probability between two
preparations at a fixed Clifford depth m:
    score = P(init=|0⟩, depth=m) − P(init=|1⟩, depth=m)
         ≈ p^m  (depolarising parameter to the m-th power)

This is equivalent to the information in a full RB curve but measured at a
single depth, making it faster to evaluate.

The search space is 6-dimensional:
    [amp_scale_target, dur_offset_target, freq_detuning_target,
     amp_scale_control, dur_offset_control, freq_detuning_control]

CMA-ES evaluates a full population of candidates per generation; all
candidates are pushed to the OPX in a single compiled program execution
via input streams (no recompilation between generations).

Prerequisites:
    - Calibrated x90 and x180 pulse parameters (amplitude, duration).
    - Calibrated initialization, measurement, and PSB threshold.
    - Native gate operations registered on qubit.xy channel.

State update:
    - Updates the x90/x180 pulse amplitude and duration via the
      XYDriveMacro.update() mechanism independently on each qubit.
    - The frequency detuning is optimised but not applied to the qubit
      state; the original intermediate frequency is preserved.
"""

node = QualibrationNode[CMAESOrbitParameters, Quam](
    name="03b_cmaes_orbit_timing",
    description=description,
    parameters=CMAESOrbitParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Debug-only parameter overrides; skipped when run externally."""
    # Realistic production-size measurement, bounded generation count.
    node.parameters.max_generations = 10
    node.parameters.qubit_pairs = ["q1_q2"]
    node.parameters.num_shots = 500
    node.parameters.num_circuits = 20
    node.parameters.population_size = 10
    node.parameters.orbit_depth = 30
    # node.parameters.simulate = True


node.machine = Quam.load()

_CLIFFORD_TABLES = build_single_qubit_clifford_tables()

# ── Helpers ──────────────────────────────────────────────────────────────

_PARAM_NAMES = [
    "amplitude_scale_target",
    "duration_offset_target",
    "freq_detuning_target",
    "amplitude_scale_control",
    "duration_offset_control",
    "freq_detuning_control",
]

_OPX_EXECUTE_PROBE = {
    "source": "cmaes_orbit_opx_breakdown.py helper run, q1_q2/q2, seed=42",
    "loop_align_save_us_per_body": 21.96,
    "gate_loop_us_per_native_gate": 3.995,
}


def _quantize_duration(val: float, min_val: int = 16) -> int:
    """Quantize a continuous duration to a multiple of 4 ns, clamped."""
    return max(min_val, int(round(val / 4.0)) * 4)


def _generate_orbit_circuits(
    num_circuits: int,
    depth: int,
    rng: np.random.Generator,
) -> list[int]:
    """Pre-generate random Clifford index sequences for orbit measurement (no recovery).

    Returns
    -------
    cliffords_flat
        Flattened Clifford indices, row-major with stride ``depth``.
        Circuit ``i`` occupies ``cliffords_flat[i*depth : (i+1)*depth]``.
        Decomposition into native gates is done on-chip via the Clifford tables.
    """
    cliffords_flat = []
    for _ in range(num_circuits):
        cliffords_flat.extend(rng.integers(0, NUM_CLIFFORDS, size=depth).tolist())
    return cliffords_flat


def _play_gate_scaled(qubit, gate_int, amplitude_scale=None, duration=None):
    """Play a single native gate with optional amplitude/duration override.

    Gate integers 0-5 match the alternative decomposition (physical gates only).
    """
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


# %% {Timing profiler}
class TimingProfiler:
    """Accumulate phase-resolved wall-clock timings for the CMA-ES loop.

    Two kinds of phase are tracked:

    * **one-time** phases (setup, e.g. connect/compile) — summed; one entry
      may be recorded per qubit pair, so the count is retained too.
    * **per-iteration** phases — one value appended per generation, so the
      full distribution (mean / min / max / total) is available.

    All durations are in seconds, measured with ``time.perf_counter`` (a
    monotonic high-resolution clock unaffected by wall-clock adjustments).
    """

    # Logical groupings used for the human-readable summary.
    COMM_PHASES = ("push", "fetch")
    CMAES_PHASES = ("cmaes_ask", "cmaes_tell")
    EXECUTE_PHASES = ("opx_execute",)
    COMPILE_PHASES = ("compile_upload",)

    def __init__(self):
        self.one_time: dict[str, float] = defaultdict(float)
        self.one_time_count: dict[str, int] = defaultdict(int)
        self.per_iter: dict[str, list[float]] = defaultdict(list)
        self.counters: dict[str, int] = defaultdict(int)
        self.opx_workloads: list[dict] = []
        self._wall_start: float | None = None
        self._wall_end: float | None = None
        self._score_t0: float | None = None

    def start(self):
        self._wall_start = time.perf_counter()

    def stop(self):
        self._wall_end = time.perf_counter()

    @property
    def total_runtime(self) -> float:
        if self._wall_start is None:
            return 0.0
        end = self._wall_end if self._wall_end is not None else time.perf_counter()
        return end - self._wall_start

    @contextmanager
    def one_time_phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.one_time[name] += time.perf_counter() - t0
            self.one_time_count[name] += 1

    @contextmanager
    def iter_phase(self, name: str):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.per_iter[name].append(time.perf_counter() - t0)

    def count(self, name: str, n: int = 1):
        self.counters[name] += n

    def add_opx_workload(self, workload: dict):
        self.opx_workloads.append(workload)

    # ── Reporting ────────────────────────────────────────────────────────
    @staticmethod
    def _stats(values: list[float]) -> dict:
        if not values:
            return {"total": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0, "n": 0}
        arr = np.asarray(values, dtype=float)
        return {
            "total": float(arr.sum()),
            "mean": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "n": int(arr.size),
        }

    def summary(self) -> dict:
        """Build a structured timing summary with percentages of total runtime."""
        total = self.total_runtime
        pct = (lambda s: 100.0 * s / total if total > 0 else 0.0)

        one_time = {
            name: {
                "total": secs,
                "count": self.one_time_count[name],
                "pct_of_total": pct(secs),
            }
            for name, secs in sorted(self.one_time.items())
        }

        per_iter = {}
        for name, values in sorted(self.per_iter.items()):
            st = self._stats(values)
            st["pct_of_total"] = pct(st["total"])
            per_iter[name] = st

        # Aggregate logical categories.
        def _cat_total(names, source):
            return sum(source.get(n, {}).get("total", 0.0) for n in names)

        # The categories form a clean, non-overlapping partition of the
        # total runtime so they can be summed / shown as a pie:
        #   communication + compile_upload + opx_execute + cmaes_calculation
        #   + host_score_compute + setup_other + unaccounted == total
        comm = _cat_total(self.COMM_PHASES, per_iter)
        cmaes = _cat_total(self.CMAES_PHASES, per_iter)
        execute = _cat_total(self.EXECUTE_PHASES, per_iter)
        compile_t = _cat_total(self.COMPILE_PHASES, one_time)
        score_compute = per_iter.get("score_compute", {}).get("total", 0.0)
        setup_total = sum(v["total"] for v in one_time.values())
        setup_other = setup_total - compile_t  # connect, config, build, etc.
        per_iter_total = sum(v["total"] for v in per_iter.values())
        accounted = setup_total + per_iter_total
        n_gen = max(
            (v["n"] for k, v in per_iter.items() if k in ("cmaes_ask", "opx_execute")),
            default=max((v["n"] for v in per_iter.values()), default=0),
        )

        categories = {
            "communication": {"total": comm, "pct_of_total": pct(comm)},
            "compile_upload": {"total": compile_t, "pct_of_total": pct(compile_t)},
            "opx_execute": {"total": execute, "pct_of_total": pct(execute)},
            "cmaes_calculation": {"total": cmaes, "pct_of_total": pct(cmaes)},
            "host_score_compute": {
                "total": score_compute, "pct_of_total": pct(score_compute),
            },
            "setup_other": {"total": setup_other, "pct_of_total": pct(setup_other)},
            "unaccounted": {
                "total": total - accounted,
                "pct_of_total": pct(total - accounted),
            },
        }

        return {
            "total_runtime_s": total,
            "n_generations": n_gen,
            "counters": dict(self.counters),
            "one_time_phases_s": one_time,
            "per_iteration_phases_s": per_iter,
            "categories_s": categories,
            "opx_workloads": list(self.opx_workloads),
            "per_generation_mean_s": (per_iter_total / n_gen) if n_gen else 0.0,
        }

    def format_report(self) -> str:
        """Render the summary as an aligned, human-readable text table."""
        s = self.summary()
        total = s["total_runtime_s"]
        n_gen = s["n_generations"]
        lines = []
        lines.append("=" * 72)
        lines.append("CMA-ES ORBIT — TIMING / PROFILING REPORT")
        lines.append("=" * 72)
        lines.append(f"Total wall-clock run time : {total:10.3f} s")
        lines.append(f"Generations executed      : {n_gen}")
        if n_gen:
            lines.append(
                f"Mean time per generation  : {s['per_generation_mean_s']:10.3f} s"
            )
        for k, v in s["counters"].items():
            lines.append(f"Counter[{k}] = {v}")

        def _row(label, secs, pct, extra=""):
            return f"  {label:<22} {secs:10.3f} s  {pct:6.1f}%   {extra}"

        lines.append("")
        lines.append("-- One-time setup phases " + "-" * 47)
        for name, d in s["one_time_phases_s"].items():
            lines.append(_row(name, d["total"], d["pct_of_total"]))

        lines.append("")
        lines.append("-- Per-iteration phases (summed over all generations) " + "-" * 18)
        lines.append(
            f"  {'phase':<22} {'total':>10}    {'%tot':>5}   "
            f"{'mean/gen':>10} {'min':>9} {'max':>9}"
        )
        for name, d in s["per_iteration_phases_s"].items():
            extra = (
                f"{d['mean']*1e3:9.2f}ms {d['min']*1e3:8.2f}ms {d['max']*1e3:8.2f}ms"
            )
            lines.append(_row(name, d["total"], d["pct_of_total"], extra))

        lines.append("")
        lines.append("-- Logical categories " + "-" * 50)
        for name, d in s["categories_s"].items():
            lines.append(_row(name, d["total"], d["pct_of_total"]))

        opx_workloads = s.get("opx_workloads") or []
        if opx_workloads:
            opx_mean = s["per_iteration_phases_s"].get("opx_execute", {}).get("mean", 0.0)
            lines.append("")
            lines.append("-- What opx_execute includes " + "-" * 44)
            lines.append(
                "  Starts after the six candidate input-stream pushes complete; "
                "ends when both survival result handles publish the next generation."
            )
            for workload in opx_workloads:
                bodies = workload.get("shot_bodies_per_generation", 0)
                body_us = (opx_mean * 1e6 / bodies) if opx_mean and bodies else 0.0
                native_gates = workload.get("native_gates_total_per_generation", 0)
                gate_us = (opx_mean * 1e6 / native_gates) if opx_mean and native_gates else 0.0
                lines.append(f"  Pair {workload.get('pair', '?')}:")
                lines.append(
                    f"    shot bodies/gen          : {bodies:,} "
                    f"({body_us:.2f} us per body at measured mean)"
                )
                lines.append(
                    f"    initialize / measure/gen : "
                    f"{workload.get('initializes_per_generation', 0):,} / "
                    f"{workload.get('measurements_per_generation', 0):,}"
                )
                lines.append(
                    f"    Clifford steps/gen       : "
                    f"{workload.get('clifford_steps_per_generation', 0):,}"
                )
                lines.append(
                    f"    native XY gates/gen      : {native_gates:,} "
                    f"({gate_us:.2f} us per gate if all OPX time is normalized by gates)"
                )
                lines.append(
                    f"    pi-prep x180 gates/gen   : "
                    f"{workload.get('pi_prep_x180_per_generation', 0):,}"
                )
                lines.append(
                    f"    reset/ramp/save/gen      : "
                    f"{workload.get('reset_frames_per_generation', 0):,} / "
                    f"{workload.get('ramp_to_zero_per_generation', 0):,} / "
                    f"{workload.get('stream_saves_per_generation', 0):,}"
                )
                lines.append(
                    f"    freq updates / stream advances/gen: "
                    f"{workload.get('frequency_updates_per_generation', 0):,} / "
                    f"{workload.get('input_stream_advances_per_generation', 0):,}"
                )
                component_rows = [
                    row for row in _opx_execute_component_breakdown(s)
                    if row["pair"] == workload.get("pair", "?")
                ]
                if component_rows:
                    lines.append("    estimated time contribution:")
                    for row in component_rows:
                        lines.append(
                            f"      {row['component']:<34} "
                            f"{row['total_s']:8.2f} s  "
                            f"{row['pct_of_opx_execute']:6.1f}%"
                        )
        lines.append("=" * 72)
        return "\n".join(lines)


def _plot_timing(profiler: "TimingProfiler", pair_name: str = ""):
    """Build a figure summarising the timing breakdown.

    Three panels: (1) overall category pie, (2) per-iteration phase stacked
    bars across generations, (3) per-phase mean ± spread bar chart.
    """
    s = profiler.summary()
    per_iter = s["per_iteration_phases_s"]
    cats = s["categories_s"]

    fig, axes = plt.subplots(1, 3, figsize=(22, 6.5))
    suffix = f" — {pair_name}" if pair_name else ""

    # Panel 1: overall category breakdown (exclude zero/near-zero slices).
    # Labels go in a legend below the pie (not around it) so they cannot
    # overflow into the neighbouring panel's y-axis.
    cat_items = [
        (k, v["total"]) for k, v in cats.items() if v["total"] > 1e-9
    ]
    if cat_items:
        labels = [k for k, _ in cat_items]
        sizes = [v for _, v in cat_items]
        # Only annotate slices that are large enough to read; tiny slices
        # would otherwise overlap each other near the pie's edge.
        def _autopct(pct):
            return f"{pct:.1f}%" if pct >= 2.0 else ""

        wedges, _, _ = axes[0].pie(
            sizes, autopct=_autopct, startangle=90, radius=0.95,
            pctdistance=0.75, textprops={"fontsize": 8},
        )
        axes[0].legend(
            wedges,
            [f"{lbl} ({v:.1f}s)" for lbl, v in zip(labels, sizes)],
            loc="upper center", bbox_to_anchor=(0.5, -0.02),
            ncol=2, fontsize=8, frameon=False,
        )
    axes[0].set_title(f"Total runtime breakdown{suffix}\n"
                      f"({s['total_runtime_s']:.1f} s total)")

    # Panel 2: per-generation stacked phase durations.
    phase_order = [
        "cmaes_ask", "push", "opx_execute", "fetch",
        "score_compute", "cmaes_tell",
    ]
    phase_order = [p for p in phase_order if p in per_iter and per_iter[p]["n"] > 0]
    phase_order += [p for p in per_iter if p not in phase_order]
    n_gen = s["n_generations"]
    if n_gen:
        gens = np.arange(1, n_gen + 1)
        bottom = np.zeros(n_gen)
        for p in phase_order:
            vals = np.asarray(profiler.per_iter.get(p, []), dtype=float)
            if vals.size != n_gen:
                # Pad/truncate defensively if a phase has a different count.
                padded = np.zeros(n_gen)
                padded[: min(vals.size, n_gen)] = vals[:n_gen]
                vals = padded
            axes[1].bar(gens, vals, bottom=bottom, label=p, width=0.9)
            bottom += vals
        axes[1].set_xlabel("Generation")
        axes[1].set_ylabel("Time per generation (s)")
        axes[1].legend(fontsize="x-small", ncol=2)
    axes[1].set_title(f"Per-generation phase breakdown{suffix}")
    axes[1].grid(True, alpha=0.3)

    # Panel 3: mean per-phase time with min/max whiskers.
    names = [p for p in phase_order if per_iter.get(p, {}).get("n", 0) > 0]
    means = [per_iter[p]["mean"] for p in names]
    mins = [per_iter[p]["min"] for p in names]
    maxs = [per_iter[p]["max"] for p in names]
    if names:
        ypos = np.arange(len(names))
        means_arr = np.asarray(means)
        lerr = means_arr - np.asarray(mins)
        uerr = np.asarray(maxs) - means_arr
        axes[2].barh(ypos, means, color="C0", alpha=0.8)
        axes[2].errorbar(
            means, ypos, xerr=[lerr, uerr], fmt="none",
            ecolor="k", capsize=3,
        )
        axes[2].set_yticks(ypos)
        axes[2].set_yticklabels(names, fontsize=8)
        axes[2].invert_yaxis()
        axes[2].set_xlabel("Mean time per generation (s)")
    axes[2].set_title(f"Per-phase mean (min–max){suffix}")
    axes[2].grid(True, alpha=0.3, axis="x")

    fig.tight_layout()
    # Extra horizontal gap so the pie legend / panel-1 ylabel never collide.
    fig.subplots_adjust(wspace=0.32, bottom=0.18)
    return fig


def _opx_execute_component_breakdown(summary: dict) -> list[dict]:
    """Estimate a time-attribution breakdown inside measured ``opx_execute``.

    The node measures the total OPX wait time exactly. The component split uses
    exact production counts plus per-unit timings from the companion helper, and
    assigns any gap to a residual bucket so the table closes to the measured
    ``opx_execute`` mean.
    """
    opx_mean = (
        summary.get("per_iteration_phases_s", {})
        .get("opx_execute", {})
        .get("mean", 0.0)
    )
    if opx_mean <= 0:
        return []

    rows = []
    for workload in summary.get("opx_workloads", []) or []:
        pair = workload.get("pair", "?")
        bodies = workload.get("shot_bodies_per_generation", 0)
        sequence_native = workload.get(
            "native_gates_in_clifford_sequences_per_generation", 0
        )
        pi_prep = workload.get("pi_prep_x180_per_generation", 0)

        def _nan_if_missing(value):
            return np.nan if value is None else float(value)

        init_ns = np.nanmean([
            _nan_if_missing(workload.get("target_initialize_estimated_ns")),
            _nan_if_missing(workload.get("control_initialize_estimated_ns")),
        ])
        measure_ns = np.nanmean([
            _nan_if_missing(workload.get("target_measure_inferred_ns")),
            _nan_if_missing(workload.get("control_measure_inferred_ns")),
        ])
        x180_ns = np.nanmean([
            _nan_if_missing(workload.get("target_x180_length_ns")),
            _nan_if_missing(workload.get("control_x180_length_ns")),
        ])

        components = [
            {
                "pair": pair,
                "component": "loop / align / save overhead",
                "total_s": bodies * _OPX_EXECUTE_PROBE["loop_align_save_us_per_body"] * 1e-6,
                "basis": (
                    f"{bodies:,} bodies x "
                    f"{_OPX_EXECUTE_PROBE['loop_align_save_us_per_body']:.2f} us"
                ),
            },
            {
                "pair": pair,
                "component": "initialize",
                "total_s": 0.0 if np.isnan(init_ns) else bodies * init_ns * 1e-9,
                "basis": (
                    f"{bodies:,} bodies x {init_ns * 1e-3:.2f} us simulated span"
                    if not np.isnan(init_ns) else "no estimate available"
                ),
            },
            {
                "pair": pair,
                "component": "Clifford XY gate loop",
                "total_s": (
                    sequence_native
                    * _OPX_EXECUTE_PROBE["gate_loop_us_per_native_gate"]
                    * 1e-6
                ),
                "basis": (
                    f"{sequence_native:,} native gates x "
                    f"{_OPX_EXECUTE_PROBE['gate_loop_us_per_native_gate']:.3f} us"
                ),
            },
            {
                "pair": pair,
                "component": "pi-prep x180",
                "total_s": 0.0 if np.isnan(x180_ns) else pi_prep * x180_ns * 1e-9,
                "basis": (
                    f"{pi_prep:,} gates x {x180_ns * 1e-3:.2f} us pulse"
                    if not np.isnan(x180_ns) else "no estimate available"
                ),
            },
            {
                "pair": pair,
                "component": "measure + readout",
                "total_s": 0.0 if np.isnan(measure_ns) else bodies * measure_ns * 1e-9,
                "basis": (
                    f"{bodies:,} bodies x {measure_ns * 1e-3:.2f} us inferred span"
                    if not np.isnan(measure_ns) else "no estimate available"
                ),
            },
        ]

        known_total = sum(row["total_s"] for row in components)
        components.append({
            "pair": pair,
            "component": "residual / production hierarchy",
            "total_s": opx_mean - known_total,
            "basis": (
                "measured opx_execute minus estimated components; includes "
                "candidate/qubit/circuit hierarchy, stream processing, unblock "
                "latency, and probe uncertainty"
            ),
        })

        for row in components:
            row["pct_of_opx_execute"] = 100.0 * row["total_s"] / opx_mean
        rows.extend(components)
    return rows


def _plot_opx_execute_breakdown(profiler: "TimingProfiler", pair_name: str = ""):
    """Build a second figure that expands the measured ``opx_execute`` bucket."""
    s = profiler.summary()
    rows = _opx_execute_component_breakdown(s)
    if not rows:
        return None

    labels = [row["component"] for row in rows]
    totals = np.asarray([row["total_s"] for row in rows], dtype=float)
    pcts = np.asarray([row["pct_of_opx_execute"] for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ypos = np.arange(len(rows))
    colors = ["C0" if v >= 0 else "C3" for v in totals]
    ax.barh(ypos, totals, color=colors, alpha=0.85)
    ax.axvline(0, color="k", linewidth=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Estimated contribution per generation (s)")
    suffix = f" - {pair_name}" if pair_name else ""
    opx_mean = s["per_iteration_phases_s"]["opx_execute"]["mean"]
    ax.set_title(f"OPX execute component breakdown{suffix}\n"
                 f"measured opx_execute = {opx_mean:.2f} s/gen")
    ax.grid(True, axis="x", alpha=0.3)

    x_span = max(abs(totals).max(), 1e-9)
    for y, total_s, pct in zip(ypos, totals, pcts):
        ha = "left" if total_s >= 0 else "right"
        offset = 0.02 * x_span if total_s >= 0 else -0.02 * x_span
        ax.text(
            total_s + offset,
            y,
            f"{total_s:.1f}s ({pct:.1f}%)",
            va="center",
            ha=ha,
            fontsize=8,
        )

    fig.tight_layout()
    return fig


def _timing_markdown(profiler: "TimingProfiler", node, pair_label: str,
                     png_name: str | None = None,
                     opx_png_name: str | None = None) -> str:
    """Render the timing summary as a standalone Markdown document."""
    s = profiler.summary()
    p = node.parameters
    lines = [
        "# CMA-ES Orbit — Timing / Profiling Report",
        "",
        f"- **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- **Qubit pair:** `{pair_label}`",
        f"- **Push mode:** batched array input streams (1 RPC/stream/gen)",
        "",
        "## Load",
        "",
        "| Parameter | Value |",
        "|---|---|",
        f"| generations | {s['n_generations']} |",
        f"| population_size | {p.population_size} |",
        f"| num_shots | {p.num_shots} |",
        f"| num_circuits | {p.num_circuits} |",
        f"| orbit_depth | {p.orbit_depth} |",
        "",
        "## Headline",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Total wall-clock runtime | {s['total_runtime_s']:.3f} s |",
        f"| Mean time per generation | {s['per_generation_mean_s']:.3f} s |",
        f"| input_stream_pushes (total) | {s['counters'].get('input_stream_pushes', 0)} |",
        "",
        "## Logical categories (partition of total runtime)",
        "",
        "| Category | Total (s) | % of total |",
        "|---|---:|---:|",
    ]
    for name, d in s["categories_s"].items():
        lines.append(f"| {name} | {d['total']:.3f} | {d['pct_of_total']:.1f}% |")

    opx_workloads = s.get("opx_workloads") or []
    if opx_workloads:
        opx_mean = s["per_iteration_phases_s"].get("opx_execute", {}).get("mean", 0.0)

        def _fmt_ns(value):
            return "n/a" if value is None else f"{float(value):.1f} ns"

        lines += [
            "",
            "## What `opx_execute` Includes",
            "",
            "`opx_execute` starts immediately after the host finishes the six "
            "`push_to_input_stream` calls for one generation. It ends when both "
            "`survival_target` and `survival_control` have published the next "
            "stream-processed result item. It excludes host-side push, fetch, "
            "score computation, CMA-ES math, compile/upload, and session setup.",
            "",
        ]
        for workload in opx_workloads:
            bodies = workload.get("shot_bodies_per_generation", 0)
            native_gates = workload.get("native_gates_total_per_generation", 0)
            body_us = (opx_mean * 1e6 / bodies) if opx_mean and bodies else 0.0
            gate_us = (opx_mean * 1e6 / native_gates) if opx_mean and native_gates else 0.0
            component_rows = [
                row for row in _opx_execute_component_breakdown(s)
                if row["pair"] == workload.get("pair", "?")
            ]
            lines += [
                f"### Pair `{workload.get('pair', '?')}`",
                "",
                "| Work item per generation | Count |",
                "|---|---:|",
                f"| shot bodies | {bodies:,} |",
                f"| initializes | {workload.get('initializes_per_generation', 0):,} |",
                f"| measurements | {workload.get('measurements_per_generation', 0):,} |",
                f"| `ramp_to_zero` calls | {workload.get('ramp_to_zero_per_generation', 0):,} |",
                f"| `reset_frame` calls | {workload.get('reset_frames_per_generation', 0):,} |",
                f"| stream `save` calls | {workload.get('stream_saves_per_generation', 0):,} |",
                f"| Clifford loop iterations | {workload.get('clifford_steps_per_generation', 0):,} |",
                f"| native XY gates inside Clifford sequences | {workload.get('native_gates_in_clifford_sequences_per_generation', 0):,} |",
                f"| extra pi-variant `x180` gates | {workload.get('pi_prep_x180_per_generation', 0):,} |",
                f"| total native XY gates | {native_gates:,} |",
                f"| frequency updates | {workload.get('frequency_updates_per_generation', 0):,} |",
                f"| input-stream advances | {workload.get('input_stream_advances_per_generation', 0):,} |",
                "",
                "| Estimated OPX time component | Total/gen (s) | % of `opx_execute` | Basis |",
                "|---|---:|---:|---|",
            ]
            for row in component_rows:
                lines.append(
                    f"| {row['component']} | {row['total_s']:.3f} | "
                    f"{row['pct_of_opx_execute']:.1f}% | {row['basis']} |"
                )

            lines += [
                "",
                "| Derived metric | Value |",
                "|---|---:|",
                f"| measured mean `opx_execute` per shot body | {body_us:.2f} us |",
                f"| measured mean `opx_execute` normalized by native XY gate | {gate_us:.2f} us |",
                f"| normal-circuit native gates used | {workload.get('native_gates_normal_circuits', 0):,} |",
                f"| pi-circuit native gates used | {workload.get('native_gates_pi_circuits', 0):,} |",
                f"| target/control x90 length | {workload.get('target_x90_length_ns')} ns / {workload.get('control_x90_length_ns')} ns |",
                f"| target/control x180 length | {workload.get('target_x180_length_ns')} ns / {workload.get('control_x180_length_ns')} ns |",
                f"| target/control initialize inferred duration | {_fmt_ns(workload.get('target_initialize_inferred_ns'))} / {_fmt_ns(workload.get('control_initialize_inferred_ns'))} |",
                f"| target/control initialize estimated span | {_fmt_ns(workload.get('target_initialize_estimated_ns'))} / {_fmt_ns(workload.get('control_initialize_estimated_ns'))} |",
                f"| target/control measure inferred duration | {_fmt_ns(workload.get('target_measure_inferred_ns'))} / {_fmt_ns(workload.get('control_measure_inferred_ns'))} |",
                "",
                "Each shot body is:",
                "",
                f"`{workload.get('shot_body_sequence')}`",
                "",
                "The initialize duration above is the macro-reported value. For the "
                "current non-heralded balanced initialize state, simulation showed "
                "an 8.4 us waveform span, so measured/simulated timing should be "
                "used for budget attribution rather than that inferred property.",
                "",
            ]

    lines += [
        "",
        "## Per-iteration phases (summed over all generations)",
        "",
        "| Phase | Total (s) | % of total | Mean/gen (ms) | Min (ms) | Max (ms) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, d in s["per_iteration_phases_s"].items():
        lines.append(
            f"| {name} | {d['total']:.3f} | {d['pct_of_total']:.1f}% | "
            f"{d['mean']*1e3:.2f} | {d['min']*1e3:.2f} | {d['max']*1e3:.2f} |"
        )

    lines += [
        "",
        "## One-time setup phases",
        "",
        "| Phase | Total (s) | % of total |",
        "|---|---:|---:|",
    ]
    for name, d in s["one_time_phases_s"].items():
        lines.append(f"| {name} | {d['total']:.3f} | {d['pct_of_total']:.1f}% |")

    if png_name:
        lines += ["", "## Figure", "", f"![timing breakdown]({png_name})"]
    if opx_png_name:
        lines += [
            "",
            "## OPX Execute Figure",
            "",
            f"![opx execute breakdown]({opx_png_name})",
        ]

    lines += [
        "",
        "## Raw report",
        "",
        "```",
        profiler.format_report(),
        "```",
        "",
    ]
    return "\n".join(lines)


def _save_timing_artifacts(
    profiler: "TimingProfiler",
    fig,
    node,
    pair_label: str,
    opx_fig=None,
):
    """Write the timing figure (PNG) and report (Markdown) next to the node.

    Saved under ``<node_dir>/timing_reports/`` with stable names so the latest
    run is always easy to find, plus a timestamped copy for history.
    """
    import os

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "timing_reports")
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    png_latest = os.path.join(out_dir, "cmaes_orbit_timing_latest.png")
    png_stamped = os.path.join(out_dir, f"cmaes_orbit_timing_{stamp}.png")
    opx_png_latest = os.path.join(out_dir, "cmaes_orbit_opx_execute_latest.png")
    opx_png_stamped = os.path.join(out_dir, f"cmaes_orbit_opx_execute_{stamp}.png")
    if fig is not None:
        fig.savefig(png_latest, dpi=120, bbox_inches="tight")
        fig.savefig(png_stamped, dpi=120, bbox_inches="tight")
    if opx_fig is not None:
        opx_fig.savefig(opx_png_latest, dpi=120, bbox_inches="tight")
        opx_fig.savefig(opx_png_stamped, dpi=120, bbox_inches="tight")

    md = _timing_markdown(
        profiler,
        node,
        pair_label,
        png_name="cmaes_orbit_timing_latest.png",
        opx_png_name=(
            "cmaes_orbit_opx_execute_latest.png" if opx_fig is not None else None
        ),
    )
    for name in ("cmaes_orbit_timing_latest.md", f"cmaes_orbit_timing_{stamp}.md"):
        with open(os.path.join(out_dir, name), "w") as fh:
            fh.write(md)
    return out_dir


def _opx_workload_summary(
    node,
    qubit_pair,
    cliffords_normal_flat: list[int],
    cliffords_pi_flat: list[int],
) -> dict:
    """Count the exact QUA work performed during one ``opx_execute`` generation."""
    p = node.parameters
    pop = int(p.population_size)
    n_qubits = 2
    n_variants = 2
    n_circuits = int(p.num_circuits)
    n_shots = int(p.num_shots)
    depth = int(p.orbit_depth)

    decomp_lengths = np.asarray(_CLIFFORD_TABLES["decomp_lengths"], dtype=int)
    normal_lengths = decomp_lengths[np.asarray(cliffords_normal_flat, dtype=int)]
    pi_lengths = decomp_lengths[np.asarray(cliffords_pi_flat, dtype=int)]

    shot_bodies = pop * n_qubits * n_variants * n_circuits * n_shots
    pi_prep = pop * n_qubits * n_circuits * n_shots
    sequence_native_gates = pop * n_qubits * n_shots * int(
        normal_lengths.sum() + pi_lengths.sum()
    )
    total_native_gates = sequence_native_gates + pi_prep

    def _duration_ns(obj, attr_name: str):
        try:
            val = getattr(obj, attr_name)
        except Exception:
            return None
        if val is None:
            return None
        return float(val) * 1e9

    def _balanced_initialize_estimated_ns(obj):
        try:
            canonical = obj._resolve_canonical_macro()
        except Exception:
            canonical = obj
        ramp = getattr(canonical, "ramp_duration", None)
        hold = getattr(canonical, "hold_duration", None)
        if ramp is None or hold is None:
            return None
        # BalancedInitializeMacro applies 0 -> -V -> +V -> 0. The waveform span
        # observed in simulation is 4*ramp + 2*hold, while inferred_duration
        # currently undercounts this macro.
        return float(4 * ramp + 2 * hold)

    qubit_target = qubit_pair.qubit_target
    qubit_control = qubit_pair.qubit_control

    return {
        "pair": qubit_pair.name,
        "population_size": pop,
        "num_qubits": n_qubits,
        "num_variants": n_variants,
        "num_circuits": n_circuits,
        "num_shots": n_shots,
        "orbit_depth": depth,
        "shot_bodies_per_generation": shot_bodies,
        "initializes_per_generation": shot_bodies,
        "measurements_per_generation": shot_bodies,
        "ramp_to_zero_per_generation": shot_bodies,
        "reset_frames_per_generation": shot_bodies,
        "stream_saves_per_generation": shot_bodies,
        "cast_to_int_per_generation": shot_bodies,
        "pi_prep_x180_per_generation": pi_prep,
        "clifford_steps_per_generation": shot_bodies * depth,
        "native_gates_in_clifford_sequences_per_generation": sequence_native_gates,
        "native_gates_total_per_generation": total_native_gates,
        "native_gates_normal_circuits": int(normal_lengths.sum()),
        "native_gates_pi_circuits": int(pi_lengths.sum()),
        "mean_native_gates_per_clifford_normal": float(normal_lengths.mean()),
        "mean_native_gates_per_clifford_pi": float(pi_lengths.mean()),
        "frequency_updates_per_generation": 4 * pop,
        "input_stream_advances_per_generation": 6,
        "result_handle_items_per_generation": 2,
        "target_x90_length_ns": int(qubit_target.macros["x90"].pulse.length),
        "control_x90_length_ns": int(qubit_control.macros["x90"].pulse.length),
        "target_x180_length_ns": int(qubit_target.macros["x90"].pi_pulse.length),
        "control_x180_length_ns": int(qubit_control.macros["x90"].pi_pulse.length),
        "target_initialize_inferred_ns": _duration_ns(
            qubit_target.macros["initialize"], "inferred_duration"
        ),
        "control_initialize_inferred_ns": _duration_ns(
            qubit_control.macros["initialize"], "inferred_duration"
        ),
        "target_initialize_estimated_ns": _balanced_initialize_estimated_ns(
            qubit_target.macros["initialize"]
        ),
        "control_initialize_estimated_ns": _balanced_initialize_estimated_ns(
            qubit_control.macros["initialize"]
        ),
        "target_measure_inferred_ns": _duration_ns(
            qubit_target.macros["measure"], "inferred_duration"
        ),
        "control_measure_inferred_ns": _duration_ns(
            qubit_control.macros["measure"], "inferred_duration"
        ),
        "shot_body_sequence": (
            "reset_frame -> align -> initialize -> align -> optional x180 "
            "(pi variant only) -> depth random Clifford loop "
            "(array lookup + switch_ + XY pulse per native gate) -> align -> "
            "measure -> align -> voltage_sequence.ramp_to_zero -> align -> "
            "Cast.to_int -> save"
        ),
        "opx_execute_excludes": (
            "host push_to_input_stream, result fetch, host score computation, "
            "CMA-ES ask/tell, compile/upload, session setup"
        ),
    }


def _run_cmaes_with_timing(
    profiler: TimingProfiler,
    evaluate_fn,
    param_names,
    x0,
    sigma0,
    bounds,
    population_size=10,
    max_generations=50,
    tolx=1e-6,
    tolfun=1e-6,
    log_callable=print,
    *,
    progress_prefix=None,
    log_each_generation=True,
):
    """Inlined, timing-instrumented twin of ``run_cmaes_optimization``.

    Identical behaviour and return value to the shared utility, but wraps
    ``es.ask()`` in the ``cmaes_ask`` phase and ``es.tell()`` in the
    ``cmaes_tell`` phase so the pure CMA-ES calculation cost is separated
    from the OPX evaluation cost (which ``evaluate_fn`` times internally).
    """
    import cma

    x0 = np.asarray(x0, dtype=float)
    n_params = len(x0)
    if len(param_names) != n_params:
        raise ValueError(
            f"param_names length ({len(param_names)}) must match x0 "
            f"length ({n_params})"
        )

    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    with profiler.one_time_phase("cmaes_init"):
        warnings.filterwarnings(
            "ignore", category=cma.evolution_strategy.InjectionWarning
        )
        opts = cma.CMAOptions()
        opts.set("popsize", population_size)
        opts.set("maxiter", max_generations)
        opts.set("tolx", tolx)
        opts.set("tolfun", tolfun)
        opts.set("bounds", [lower_bounds, upper_bounds])
        opts.set("verbose", -9)
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    label = f"[{progress_prefix}] " if progress_prefix else ""
    log_callable(
        f"  {label}CMA-ES start — max {max_generations} generations, "
        f"population_size={population_size}, σ₀={sigma0:g}"
    )

    param_history = []
    score_history = []
    all_candidates = []
    all_scores = []
    best_score_so_far = -np.inf
    best_params_so_far = x0.copy()
    gen = 0

    while not es.stop():
        with profiler.iter_phase("cmaes_ask"):
            candidates = np.array(es.ask())

        scores = np.asarray(evaluate_fn(candidates), dtype=float)

        if scores.shape != (len(candidates),):
            raise ValueError(
                f"evaluate_fn must return shape ({len(candidates)},), "
                f"got {scores.shape}"
            )

        non_finite = ~np.isfinite(scores)
        if non_finite.any():
            n_bad = int(non_finite.sum())
            log_callable(
                f"  {label}WARNING: {n_bad}/{len(scores)} non-finite scores "
                f"replaced with -inf"
            )
            scores = np.where(non_finite, -np.inf, scores)

        with profiler.iter_phase("cmaes_tell"):
            es.tell(candidates.tolist(), (-scores).tolist())

        gen_best_idx = int(np.argmax(scores))
        gen_best_score = float(scores[gen_best_idx])
        if gen_best_score > best_score_so_far:
            best_score_so_far = gen_best_score
            best_params_so_far = candidates[gen_best_idx].copy()

        param_history.append(np.array(es.mean))
        score_history.append(best_score_so_far)
        all_candidates.append(candidates.copy())
        all_scores.append(scores.copy())
        gen += 1

        if log_each_generation:
            pct_max = 100.0 * gen / max_generations if max_generations > 0 else 0.0
            param_str = ", ".join(
                f"{name}={val:.6g}" for name, val in zip(param_names, es.mean)
            )
            log_callable(
                f"  {label}progress {gen}/{max_generations} ({pct_max:5.1f}% of max gen) | "
                f"best = {best_score_so_far:.6f} | "
                f"this gen = {gen_best_score:.6f} | mean: {param_str}"
            )

    stop_conditions = es.stop()
    stop_reason = ", ".join(f"{k}: {v}" for k, v in stop_conditions.items())
    converged = "maxiter" not in stop_conditions

    log_callable(
        f"  {label}CMA-ES finished {gen}/{max_generations} generations. Reason: {stop_reason}"
    )

    return OptimizationResult(
        best_params=best_params_so_far,
        best_score=best_score_so_far,
        param_names=param_names,
        param_history=param_history,
        score_history=score_history,
        all_candidates=all_candidates,
        all_scores=all_scores,
        n_generations=gen,
        converged=converged,
        stop_reason=stop_reason,
    )


# %% {Create_QUA_program}
def _build_qua_program(
    node, qubit_pair, depth,
    cliffords_normal_flat, cliffords_pi_flat,
):
    """Build a QUA program for orbit measurement on both qubits in a pair.

    For each qubit and candidate, runs two variants:
      1. Normal init → random sequence → measure
      2. Normal init → x180 → random sequence → measure

    Each variant uses independent pre-generated random circuits.
    Shape of one fetch per qubit: ``(pop_size, 2)`` where columns are
    [P_normal, P_pi].
    """
    qubit_target = qubit_pair.qubit_target
    qubit_control = qubit_pair.qubit_control

    if_target = qubit_target.xy.intermediate_frequency
    if_control = qubit_control.xy.intermediate_frequency

    num_circuits = node.parameters.num_circuits
    num_shots = node.parameters.num_shots
    pop_size = node.parameters.population_size

    with program() as qua_prog:
        # Batched (array) input streams: one stream carries the whole
        # population (size=pop_size) so the host pushes the generation with a
        # single RPC per stream (6 RPCs/gen) instead of one RPC per value
        # (6*pop_size RPCs/gen).  See the timing analysis — push_to_input_stream
        # is a blocking gRPC round-trip, so RPC count dominates push cost.
        amp_target_in = declare_input_stream(
            "client", stream_id="amp_target", dtype=fixed, size=pop_size
        )
        dur_target_in = declare_input_stream(
            "client", stream_id="dur_target", dtype=int, size=pop_size
        )
        freq_det_target_in = declare_input_stream(
            "client", stream_id="freq_det_target", dtype=int, size=pop_size
        )
        amp_control_in = declare_input_stream(
            "client", stream_id="amp_control", dtype=fixed, size=pop_size
        )
        dur_control_in = declare_input_stream(
            "client", stream_id="dur_control", dtype=int, size=pop_size
        )
        freq_det_control_in = declare_input_stream(
            "client", stream_id="freq_det_control", dtype=int, size=pop_size
        )

        amp_scale_t = declare(fixed)
        dur_t = declare(int)
        freq_det_t = declare(int)
        amp_scale_c = declare(fixed)
        dur_c = declare(int)
        freq_det_c = declare(int)

        # Clifford index sequences; circuit i occupies [i*depth : (i+1)*depth]
        cliffords_normal_qua = declare(int, value=cliffords_normal_flat)
        cliffords_pi_qua = declare(int, value=cliffords_pi_flat)

        # Clifford decomposition lookup tables (alternative decomposition)
        clifford_decomp_qua = declare(int, value=_CLIFFORD_TABLES["decomp_flat"])
        clifford_decomp_offsets_qua = declare(int, value=_CLIFFORD_TABLES["decomp_offsets"])
        clifford_decomp_lengths_qua = declare(int, value=_CLIFFORD_TABLES["decomp_lengths"])

        candidate_idx = declare(int)
        circuit_idx = declare(int)
        shot_idx = declare(int)
        cliff_loop_idx = declare(int)
        gate_idx = declare(int)
        rand_clifford = declare(int)
        decomp_offset = declare(int)
        decomp_length = declare(int)
        current_gate = declare(int)

        state_target = declare(int)
        state_control = declare(int)
        state_target_st = declare_output_stream()
        state_control_st = declare_output_stream()

        with infinite_loop_():
            # One advance per generation loads the full population array for
            # each stream; candidates are then read by indexing.
            advance_input_stream(amp_target_in)
            advance_input_stream(dur_target_in)
            advance_input_stream(freq_det_target_in)
            advance_input_stream(amp_control_in)
            advance_input_stream(dur_control_in)
            advance_input_stream(freq_det_control_in)
            with for_(candidate_idx, 0, candidate_idx < pop_size, candidate_idx + 1):
                assign(amp_scale_t, amp_target_in[candidate_idx])
                assign(dur_t, dur_target_in[candidate_idx])
                assign(freq_det_t, freq_det_target_in[candidate_idx])
                assign(amp_scale_c, amp_control_in[candidate_idx])
                assign(dur_c, dur_control_in[candidate_idx])
                assign(freq_det_c, freq_det_control_in[candidate_idx])

                # --- Orbit on qubit_target ---
                qubit_target.xy.update_frequency(if_target + freq_det_t)

                # Variant 1: normal initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_target.xy.name)
                        align()
                        qubit_target.initialize()
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_normal_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_target, current_gate, amp_scale_t, dur_t)

                        align()
                        p = qubit_target.measure()
                        align()
                        qubit_target.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_target, Cast.to_int(p))
                        save(state_target, state_target_st)

                # Variant 2: pi-pulse initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_target.xy.name)
                        align()
                        qubit_target.initialize()
                        align()
                        qubit_target.x180(amplitude_scale=amp_scale_t, duration=dur_t)
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_pi_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_target, current_gate, amp_scale_t, dur_t)

                        align()
                        p = qubit_target.measure()
                        align()
                        qubit_target.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_target, Cast.to_int(p))
                        save(state_target, state_target_st)

                qubit_target.xy.update_frequency(if_target)

                # --- Orbit on qubit_control ---
                qubit_control.xy.update_frequency(if_control + freq_det_c)

                # Variant 1: normal initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_control.xy.name)
                        align()
                        qubit_control.initialize()
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_normal_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_control, current_gate, amp_scale_c, dur_c)

                        align()
                        p = qubit_control.measure()
                        align()
                        qubit_control.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_control, Cast.to_int(p))
                        save(state_control, state_control_st)

                # Variant 2: pi-pulse initialization
                with for_(circuit_idx, 0, circuit_idx < num_circuits, circuit_idx + 1):
                    with for_(shot_idx, 0, shot_idx < num_shots, shot_idx + 1):
                        reset_frame(qubit_control.xy.name)
                        align()
                        qubit_control.initialize()
                        align()
                        qubit_control.x180(amplitude_scale=amp_scale_c, duration=dur_c)
                        align()

                        with for_(cliff_loop_idx, 0, cliff_loop_idx < depth, cliff_loop_idx + 1):
                            assign(rand_clifford, cliffords_pi_qua[circuit_idx * depth + cliff_loop_idx])
                            assign(decomp_offset, clifford_decomp_offsets_qua[rand_clifford])
                            assign(decomp_length, clifford_decomp_lengths_qua[rand_clifford])
                            with for_(gate_idx, 0, gate_idx < decomp_length, gate_idx + 1):
                                assign(current_gate, clifford_decomp_qua[decomp_offset + gate_idx])
                                _play_gate_scaled(qubit_control, current_gate, amp_scale_c, dur_c)

                        align()
                        p = qubit_control.measure()
                        align()
                        qubit_control.voltage_sequence.ramp_to_zero()
                        align()

                        assign(state_control, Cast.to_int(p))
                        save(state_control, state_control_st)

                qubit_control.xy.update_frequency(if_control)

        with stream_processing():
            # Per qubit: num_circuits shots for normal, then num_circuits for pi
            # → buffer(num_shots) averages shots, buffer(num_circuits) averages
            #   circuits, buffer(2) groups [normal, pi], buffer(pop_size) groups
            #   candidates.
            (
                state_target_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(2)
                .buffer(pop_size)
                .save_all("survival_target")
            )
            (
                state_control_st
                .buffer(num_shots)
                .map(FUNCTIONS.average())
                .buffer(num_circuits)
                .map(FUNCTIONS.average())
                .buffer(2)
                .buffer(pop_size)
                .save_all("survival_control")
            )

    return qua_prog


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Validate parameters, generate orbit circuits, and compile the QUA program."""
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    if not qubit_pairs:
        raise ValueError("No qubit pairs resolved — check qubit_pairs parameter or machine config.")

    depth = node.parameters.orbit_depth
    node.namespace["orbit_depth"] = depth

    rng = np.random.default_rng(node.parameters.seed)

    cliffords_normal_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
    cliffords_pi_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
    node.namespace["orbit_circuits"] = {
        "cliffords_normal_flat": cliffords_normal_flat,
        "cliffords_pi_flat": cliffords_pi_flat,
    }

    qp = qubit_pairs[0]
    node.namespace["qua_program"] = _build_qua_program(
        node, qp, depth, cliffords_normal_flat, cliffords_pi_flat,
    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Run_CMA-ES_loop}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def run_cmaes_loop(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Execute a separate CMA-ES optimisation for each qubit pair.

    The score is the orbit separation: P_normal − P_pi, averaged across
    both qubits in the pair.
    """
    import time as _time

    profiler = TimingProfiler()
    profiler.start()

    qubit_pairs = node.namespace["qubit_pairs"]
    depth = node.namespace["orbit_depth"]

    with profiler.one_time_phase("connect"):
        qmm = node.machine.connect(timeout=node.parameters.compilation_timeout)
    with profiler.one_time_phase("generate_config"):
        config = node.machine.generate_config()

    pop_size = node.parameters.population_size

    lo = np.array([
        node.parameters.amplitude_scale_min,
        node.parameters.duration_offset_min,
        node.parameters.freq_detuning_min,
        node.parameters.amplitude_scale_min,
        node.parameters.duration_offset_min,
        node.parameters.freq_detuning_min,
    ])
    hi = np.array([
        node.parameters.amplitude_scale_max,
        node.parameters.duration_offset_max,
        node.parameters.freq_detuning_max,
        node.parameters.amplitude_scale_max,
        node.parameters.duration_offset_max,
        node.parameters.freq_detuning_max,
    ])
    param_range = hi - lo

    x0_phys = np.array([
        node.parameters.amplitude_scale_initial,
        node.parameters.duration_offset_initial,
        node.parameters.freq_detuning_initial,
        node.parameters.amplitude_scale_initial,
        node.parameters.duration_offset_initial,
        node.parameters.freq_detuning_initial,
    ])
    x0_norm = (x0_phys - lo) / param_range
    bounds_norm = [(0.0, 1.0)] * len(_PARAM_NAMES)

    optimization_results = {}
    measurement_streams = {}

    for qp in qubit_pairs:
        node.log(f"  Starting CMA-ES orbit optimisation for pair {qp.name}...")

        qubit_target = qp.qubit_target
        qubit_control = qp.qubit_control

        with profiler.one_time_phase("generate_circuits"):
            rng = np.random.default_rng(node.parameters.seed)
            cliffords_normal_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
            cliffords_pi_flat = _generate_orbit_circuits(node.parameters.num_circuits, depth, rng)
        profiler.add_opx_workload(
            _opx_workload_summary(node, qp, cliffords_normal_flat, cliffords_pi_flat)
        )

        with profiler.one_time_phase("build_program"):
            qua_prog = _build_qua_program(
                node, qp, depth, cliffords_normal_flat, cliffords_pi_flat,
            )

        cal_dur_target = qubit_target.macros["x90"].pulse.length
        cal_dur_control = qubit_control.macros["x90"].pulse.length

        orbit_history = {
            "p_normal_target": [], "p_pi_target": [],
            "p_normal_control": [], "p_pi_control": [],
            "separation_target": [], "separation_control": [],
            "score_average": [],
            "all_separation_target": [], "all_separation_control": [],
            "all_score_average": [],
            "running_best_score": [],
            "running_best_sep_target": [],
            "running_best_sep_control": [],
            "running_best_p_normal_target": [],
            "running_best_p_pi_target": [],
            "running_best_p_normal_control": [],
            "running_best_p_pi_control": [],
        }

        with profiler.one_time_phase("session_open"):
            session_cm = qm_session(qmm, config, timeout=node.parameters.timeout)
            qm = session_cm.__enter__()
        try:
            with profiler.one_time_phase("compile_upload"):
                job = qm.execute(qua_prog)
            target_handle = job.result_handles.get("survival_target")
            control_handle = job.result_handles.get("survival_control")
            generation_counter = 0

            def evaluate_candidates(candidates_norm: np.ndarray) -> np.ndarray:
                """Push one generation and compute orbit separation scores."""
                nonlocal generation_counter

                candidates_phys = lo + candidates_norm * param_range
                # --- Communication out: stream candidate params host → OPX ---
                # Batched: one list-push per stream (6 RPCs) rather than one
                # push per value (6*pop_size RPCs).  The QUA program declares
                # these streams with size=pop_size and advances once per
                # generation, so a single push carries the whole population.
                with profiler.iter_phase("push"):
                    job.push_to_input_stream(
                        "amp_target", [float(c[0]) for c in candidates_phys]
                    )
                    job.push_to_input_stream(
                        "dur_target",
                        [_quantize_duration(cal_dur_target + c[1]) for c in candidates_phys],
                    )
                    job.push_to_input_stream(
                        "freq_det_target", [int(round(c[2])) for c in candidates_phys]
                    )
                    job.push_to_input_stream(
                        "amp_control", [float(c[3]) for c in candidates_phys]
                    )
                    job.push_to_input_stream(
                        "dur_control",
                        [_quantize_duration(cal_dur_control + c[4]) for c in candidates_phys],
                    )
                    job.push_to_input_stream(
                        "freq_det_control", [int(round(c[5])) for c in candidates_phys]
                    )
                    profiler.count("input_stream_pushes", 6)
                    profiler.count("candidates_evaluated", len(candidates_phys))

                # --- OPX execution: wait until this generation's results land ---
                target_count = generation_counter + 1
                with profiler.iter_phase("opx_execute"):
                    while target_handle.count_so_far() < target_count:
                        _time.sleep(0.005)
                    while control_handle.count_so_far() < target_count:
                        _time.sleep(0.005)

                # --- Communication in: fetch survival probabilities OPX → host ---
                # Shape: (pop_size, 2) where [:, 0] = normal, [:, 1] = pi
                with profiler.iter_phase("fetch"):
                    surv_target = np.asarray(
                        target_handle.fetch(generation_counter, flat_struct=True),
                        dtype=np.float64,
                    )
                    surv_control = np.asarray(
                        control_handle.fetch(generation_counter, flat_struct=True),
                        dtype=np.float64,
                    )
                generation_counter += 1

                profiler._score_t0 = time.perf_counter()
                sep_target = np.abs(surv_target[:, 0] - surv_target[:, 1])
                sep_control = np.abs(surv_control[:, 0] - surv_control[:, 1])
                scores = (sep_target + sep_control) / 2.0

                best_idx = int(np.argmax(scores))
                orbit_history["p_normal_target"].append(float(surv_target[best_idx, 0]))
                orbit_history["p_pi_target"].append(float(surv_target[best_idx, 1]))
                orbit_history["p_normal_control"].append(float(surv_control[best_idx, 0]))
                orbit_history["p_pi_control"].append(float(surv_control[best_idx, 1]))
                orbit_history["separation_target"].append(float(sep_target[best_idx]))
                orbit_history["separation_control"].append(float(sep_control[best_idx]))
                orbit_history["score_average"].append(float(scores[best_idx]))

                orbit_history["all_separation_target"].append(sep_target.copy())
                orbit_history["all_separation_control"].append(sep_control.copy())
                orbit_history["all_score_average"].append(scores.copy())

                current_best = float(scores[best_idx])
                prev_best = (
                    orbit_history["running_best_score"][-1]
                    if orbit_history["running_best_score"]
                    else -1.0
                )
                if current_best >= prev_best:
                    orbit_history["running_best_score"].append(current_best)
                    orbit_history["running_best_sep_target"].append(float(sep_target[best_idx]))
                    orbit_history["running_best_sep_control"].append(float(sep_control[best_idx]))
                    orbit_history["running_best_p_normal_target"].append(float(surv_target[best_idx, 0]))
                    orbit_history["running_best_p_pi_target"].append(float(surv_target[best_idx, 1]))
                    orbit_history["running_best_p_normal_control"].append(float(surv_control[best_idx, 0]))
                    orbit_history["running_best_p_pi_control"].append(float(surv_control[best_idx, 1]))
                else:
                    orbit_history["running_best_score"].append(prev_best)
                    orbit_history["running_best_sep_target"].append(
                        orbit_history["running_best_sep_target"][-1]
                    )
                    orbit_history["running_best_sep_control"].append(
                        orbit_history["running_best_sep_control"][-1]
                    )
                    orbit_history["running_best_p_normal_target"].append(
                        orbit_history["running_best_p_normal_target"][-1]
                    )
                    orbit_history["running_best_p_pi_target"].append(
                        orbit_history["running_best_p_pi_target"][-1]
                    )
                    orbit_history["running_best_p_normal_control"].append(
                        orbit_history["running_best_p_normal_control"][-1]
                    )
                    orbit_history["running_best_p_pi_control"].append(
                        orbit_history["running_best_p_pi_control"][-1]
                    )

                # Host-side scoring + bookkeeping time (between fetch and tell).
                profiler.per_iter["score_compute"].append(
                    time.perf_counter() - profiler._score_t0
                )
                return scores

            try:
                opt_result = _run_cmaes_with_timing(
                    profiler=profiler,
                    evaluate_fn=evaluate_candidates,
                    param_names=_PARAM_NAMES,
                    x0=x0_norm,
                    sigma0=node.parameters.sigma0,
                    bounds=bounds_norm,
                    population_size=pop_size,
                    max_generations=node.parameters.max_generations,
                    tolx=node.parameters.tolx,
                    tolfun=node.parameters.tolfun,
                    log_callable=node.log,
                    progress_prefix=qp.name,
                    log_each_generation=node.parameters.cmaes_log_each_generation,
                )
                opt_result.best_params = lo + opt_result.best_params * param_range
                opt_result.param_history = [
                    lo + h * param_range for h in opt_result.param_history
                ]
                opt_result.all_candidates = [
                    lo + c * param_range for c in opt_result.all_candidates
                ]
                optimization_results[qp.name] = opt_result
                measurement_streams[qp.name] = {"orbit_history": orbit_history}
            finally:
                job.cancel()
        finally:
            session_cm.__exit__(None, None, None)

    profiler.stop()

    # ── Emit and persist the timing report ──────────────────────────────
    report_text = profiler.format_report()
    for line in report_text.splitlines():
        node.log(line)

    timing_summary = profiler.summary()
    node.results["timing"] = timing_summary
    node.results["timing_report"] = report_text
    node.namespace["timing_profiler"] = profiler
    pair_label = qubit_pairs[0].name if qubit_pairs else ""
    timing_fig = None
    opx_timing_fig = None
    try:
        node.results.setdefault("figures", {})
        timing_fig = _plot_timing(profiler, pair_label)
        node.results["figures"]["timing_breakdown"] = timing_fig
        opx_timing_fig = _plot_opx_execute_breakdown(profiler, pair_label)
        if opx_timing_fig is not None:
            node.results["figures"]["opx_execute_breakdown"] = opx_timing_fig
    except Exception as exc:  # plotting must never break the run
        node.log(f"  WARNING: timing figure could not be generated: {exc}")

    try:
        out_dir = _save_timing_artifacts(
            profiler, timing_fig, node, pair_label, opx_fig=opx_timing_fig
        )
        node.log(f"  Timing report + figure saved to: {out_dir}")
    except Exception as exc:
        node.log(f"  WARNING: timing artifacts could not be saved: {exc}")

    node.namespace["optimization_results"] = optimization_results
    node.namespace["measurement_streams"] = measurement_streams
    node.results["orbit_depth"] = node.parameters.orbit_depth
    node.results["optimization_results"] = {
        name: result.to_dict() for name, result in optimization_results.items()
    }

    def _serialize_value(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {kk: _serialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], np.ndarray):
            return [arr.tolist() for arr in v]
        return v

    node.results["measurement_streams"] = {
        name: _serialize_value(streams)
        for name, streams in measurement_streams.items()
    }


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Load a previously saved optimisation result."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    node.namespace["orbit_depth"] = node.parameters.orbit_depth
    node.namespace["optimization_results"] = {
        name: OptimizationResult.from_dict(d)
        for name, d in node.results["optimization_results"].items()
    }

    def _deserialize_value(v):
        if isinstance(v, dict):
            return {kk: _deserialize_value(vv) for kk, vv in v.items()}
        if isinstance(v, list) and len(v) > 0 and isinstance(v[0], list):
            return [np.array(arr) for arr in v]
        return v

    raw_streams = node.results.get("measurement_streams", {})
    node.namespace["measurement_streams"] = {
        name: _deserialize_value(streams)
        for name, streams in raw_streams.items()
    }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Summarise the CMA-ES optimisation outcome per qubit pair."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    fit_results = analyse_optimization(
        opt_results, success_threshold=node.parameters.success_threshold
    )

    for qp_name, summary in fit_results.items():
        streams = measurement_streams.get(qp_name, {})
        orbit_hist = streams.get("orbit_history", {})
        if orbit_hist:
            summary["best_separation_target"] = max(orbit_hist["separation_target"])
            summary["best_separation_control"] = max(orbit_hist["separation_control"])
            summary["best_score_average"] = max(orbit_hist["score_average"])

    node.results["fit_results"] = fit_results

    log_optimization_results(opt_results, log_callable=node.log)

    for qp in node.namespace["qubit_pairs"]:
        summary = fit_results.get(qp.name)
        if summary is None:
            continue
        sep_t = summary.get("best_separation_target")
        sep_c = summary.get("best_separation_control")
        score_avg = summary.get("best_score_average")
        if sep_t is not None:
            node.log(
                f"  [{qp.name}] Best orbit separations: "
                f"{qp.qubit_target.name}={sep_t:.4f}, "
                f"{qp.qubit_control.name}={sep_c:.4f}, "
                f"average={score_avg:.4f} "
                f"(at depth {node.parameters.orbit_depth})"
            )

    node.outcomes = {
        qp_name: (Outcome.SUCCESSFUL if summary["success"] else Outcome.FAILED)
        for qp_name, summary in fit_results.items()
    }


# %% {Plot_data}
def _plot_orbit_separation_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot orbit separation for the best candidate of each generation."""
    if not orbit_history or not orbit_history.get("score_average"):
        ax.set_title(f"No orbit data — {pair_name}")
        return

    n_gen = len(orbit_history["score_average"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["separation_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name}",
    )
    ax.plot(
        generations, orbit_history["separation_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name}",
    )
    ax.plot(
        generations, orbit_history["score_average"], "D-",
        color="C2", markersize=5, linewidth=2, label="Average",
    )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Orbit separation (P_normal − P_pi)")
    title = (
        f"Best-candidate orbit separation — {pair_name}" if pair_name
        else "Best-candidate orbit separation"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_survival_variants_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Plot P_normal and P_pi for the best candidate of each generation."""
    if not orbit_history or not orbit_history.get("p_normal_target"):
        ax.set_title(f"No survival data — {pair_name}")
        return

    n_gen = len(orbit_history["p_normal_target"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["p_normal_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name} normal",
    )
    ax.plot(
        generations, orbit_history["p_pi_target"], "o--",
        color="C0", markersize=4, alpha=0.6, label=f"{qubit_target_name} π-init",
    )
    ax.plot(
        generations, orbit_history["p_normal_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name} normal",
    )
    ax.plot(
        generations, orbit_history["p_pi_control"], "s--",
        color="C1", markersize=4, alpha=0.6, label=f"{qubit_control_name} π-init",
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival probability")
    title = (
        f"Best-candidate survival — {pair_name}" if pair_name
        else "Best-candidate survival"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_individual_scores_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    pair_name: str = "",
) -> None:
    """Scatter all candidate scores per generation with running best overlay."""
    if not orbit_history or not orbit_history.get("all_score_average"):
        ax.set_title(f"No individual score data — {pair_name}")
        return

    n_gen = len(orbit_history["all_score_average"])
    for gen_idx in range(n_gen):
        gen_num = gen_idx + 1
        avg_vals = np.asarray(orbit_history["all_score_average"][gen_idx])
        ax.scatter(
            np.full_like(avg_vals, gen_num, dtype=float), avg_vals,
            s=10, alpha=0.3, color="C7", zorder=1,
        )

    generations = np.arange(1, n_gen + 1)
    if orbit_history.get("running_best_score"):
        ax.plot(
            generations, orbit_history["running_best_score"],
            "D-", color="C2", markersize=5, linewidth=2,
            label="Running best (avg)", zorder=3,
        )
    if orbit_history.get("running_best_sep_target"):
        ax.plot(
            generations, orbit_history["running_best_sep_target"],
            "o--", color="C0", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_target_name})", zorder=2,
        )
    if orbit_history.get("running_best_sep_control"):
        ax.plot(
            generations, orbit_history["running_best_sep_control"],
            "s--", color="C1", markersize=4, linewidth=1.5,
            label=f"Running best ({qubit_control_name})", zorder=2,
        )

    ax.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Orbit score")
    title = (
        f"Individual scores & running best — {pair_name}" if pair_name
        else "Individual scores & running best"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def _plot_running_best_survival_on_ax(
    ax: plt.Axes,
    orbit_history: dict,
    qubit_target_name: str,
    qubit_control_name: str,
    orbit_depth: int,
    pair_name: str = "",
) -> None:
    """Plot the running-best candidate's P_normal and P_pi over generations."""
    if not orbit_history or not orbit_history.get("running_best_p_normal_target"):
        ax.set_title(f"No running-best data — {pair_name}")
        return

    n_gen = len(orbit_history["running_best_p_normal_target"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(
        generations, orbit_history["running_best_p_normal_target"], "o-",
        color="C0", markersize=4, label=f"{qubit_target_name} normal",
    )
    ax.plot(
        generations, orbit_history["running_best_p_pi_target"], "o--",
        color="C0", markersize=4, alpha=0.6, label=f"{qubit_target_name} π-init",
    )
    ax.plot(
        generations, orbit_history["running_best_p_normal_control"], "s-",
        color="C1", markersize=4, label=f"{qubit_control_name} normal",
    )
    ax.plot(
        generations, orbit_history["running_best_p_pi_control"], "s--",
        color="C1", markersize=4, alpha=0.6, label=f"{qubit_control_name} π-init",
    )

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)

    final_score = orbit_history.get("running_best_score", [])
    if final_score:
        ax.text(
            0.98, 0.02,
            f"Best score: {final_score[-1]:.4f} (depth={orbit_depth})",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize="small",
            bbox=dict(boxstyle="round,pad=0.3", fc="wheat", alpha=0.7),
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Survival probability")
    title = (
        f"Running-best survival — {pair_name}" if pair_name
        else "Running-best survival"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Generate convergence, separation, and parameter-evolution plots."""
    opt_results = node.namespace["optimization_results"]
    measurement_streams = node.namespace.get("measurement_streams", {})

    pair_names = list(opt_results.keys())
    n_pairs = max(len(pair_names), 1)
    orbit_depth = node.parameters.orbit_depth

    fig_combined, axes = plt.subplots(
        5, n_pairs, figsize=(9 * n_pairs, 22), squeeze=False,
    )

    for col, pname in enumerate(pair_names):
        streams = measurement_streams.get(pname, {})
        orbit_hist = streams.get("orbit_history", {})

        qp = node.namespace["qubit_pairs"][col]

        _plot_orbit_separation_on_ax(
            axes[0, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )
        _plot_survival_variants_on_ax(
            axes[1, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )
        _plot_running_best_survival_on_ax(
            axes[2, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name,
            orbit_depth, pname,
        )
        plot_score_convergence_on_ax(axes[3, col], opt_results[pname], pname)
        _plot_individual_scores_on_ax(
            axes[4, col], orbit_hist,
            qp.qubit_target.name, qp.qubit_control.name, pname,
        )

    fig_combined.tight_layout()

    fig_params = plot_parameter_evolution(opt_results)
    plt.show()

    # Merge rather than overwrite so the timing figure produced by
    # run_cmaes_loop is preserved alongside the analysis figures.
    node.results.setdefault("figures", {})
    node.results["figures"]["orbit_and_convergence"] = fig_combined
    node.results["figures"]["parameter_evolution"] = fig_params
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Apply optimal per-qubit x90 amplitude and duration."""
    fit_results = node.results["fit_results"]

    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            pair_summary = fit_results.get(qp.name)
            if pair_summary is None:
                continue
            if not pair_summary["success"]:
                node.log(
                    f"  {qp.name}: optimisation did not succeed — skipping."
                )
                continue

            best = pair_summary["best_params"]
            sep_t = pair_summary.get("best_separation_target", pair_summary["best_score"])
            sep_c = pair_summary.get("best_separation_control", pair_summary["best_score"])

            qubit_params = [
                (qp.qubit_target, best["amplitude_scale_target"], best["duration_offset_target"], best["freq_detuning_target"], sep_t),
                (qp.qubit_control, best["amplitude_scale_control"], best["duration_offset_control"], best["freq_detuning_control"], sep_c),
            ]

            for qubit, opt_amp_scale, opt_dur_offset, opt_freq_det, separation in qubit_params:
                xy_macro = qubit.macros["x90"]

                current_x90_amp = xy_macro.pulse.amplitude
                current_pi_amp = xy_macro.pi_pulse.amplitude
                current_duration = xy_macro.pulse.length
                current_larmor = qubit.larmor_frequency

                new_duration = _quantize_duration(current_duration + opt_dur_offset)
                # Larmor flows into the (integer) drive IF = larmor − LO, so keep it int.
                new_larmor = int(round(current_larmor + opt_freq_det))

                xy_macro.update(
                    amplitude_scale=opt_amp_scale,
                    duration=new_duration,
                )
                qubit.larmor_frequency = new_larmor

                node.log(
                    f"  {qp.name}/{qubit.name}: gate params updated — "
                    f"x90_amp: {current_x90_amp:.6g} → {current_x90_amp * opt_amp_scale:.6g} V, "
                    f"x180_amp: {current_pi_amp:.6g} → {current_pi_amp * opt_amp_scale:.6g} V, "
                    f"duration: {current_duration} → {new_duration} ns, "
                    f"freq_detuning: {opt_freq_det:+.0f} Hz, "
                    f"larmor: {current_larmor:.0f} → {new_larmor:.0f} Hz, "
                    f"orbit_separation={separation:.4f}"
                )

            node.log(
                f"  {qp.name}: average orbit score = {pair_summary['best_score']:.4f} "
                f"(depth={node.parameters.orbit_depth})"
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[CMAESOrbitParameters, Quam]):
    """Persist all results, figures, and parameters to disk."""
    node.save()
