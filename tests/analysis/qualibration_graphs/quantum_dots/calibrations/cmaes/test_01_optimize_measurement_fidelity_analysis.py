"""Analysis test for ``01_optimize_measurement_fidelity`` (CMA-ES).

Constructs a synthetic ``OptimizationResult`` (as if the CMA-ES loop had
already executed), then runs the node's ``analyse_data``, ``plot_data``, and
``update_state`` actions.  The synthetic result models a landscape where
fidelity peaks at known (detuning, ramp_duration) values so we can verify that
the best-parameter extraction and state update work correctly.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from .conftest import (
    ARTIFACTS_BASE,
    CALIBRATION_LIBRARY_ROOT,
)

NODE_NAME = "01_optimize_measurement_fidelity"
PAIR_NAME = "q1_q2"


# ── Synthetic OptimizationResult builder ───────────────────────────────


def _build_synthetic_optimization_result(
    optimal_detuning: float = 0.03,
    optimal_ramp_ns: float = 200.0,
    n_generations: int = 10,
    population_size: int = 6,
    seed: int = 42,
):
    """Build a realistic ``OptimizationResult`` that converges to known values.

    Simulates a fidelity landscape:
        F(d, r) = F_max - a*(d - d_opt)^2 - b*(r - r_opt)^2 + noise
    where CMA-ES gradually narrows in on (d_opt, r_opt).
    """
    from calibration_utils.cmaes.optimization import OptimizationResult

    rng = np.random.default_rng(seed)
    param_names = ["detuning", "ramp_duration"]
    x0 = np.array([0.0, 100.0])

    def _fidelity(d, r):
        return (
            0.95
            - 50.0 * (d - optimal_detuning) ** 2
            - 1e-5 * (r - optimal_ramp_ns) ** 2
        )

    param_history = []
    score_history = []
    all_candidates = []
    all_scores = []
    best_score = -np.inf
    best_params = x0.copy()

    mean = x0.copy()
    sigma = np.array([0.05, 200.0])

    for gen in range(n_generations):
        # Shrink sigma each generation (mimicking CMA-ES convergence)
        scale = max(0.1, 1.0 - gen / n_generations)
        candidates = mean + rng.normal(0, sigma * scale, size=(population_size, 2))

        # Enforce bounds
        candidates[:, 0] = np.clip(candidates[:, 0], -0.1, 0.1)
        candidates[:, 1] = np.clip(candidates[:, 1], 16, 2000)

        scores = np.array([
            _fidelity(c[0], c[1]) + rng.normal(0, 0.01) for c in candidates
        ])
        scores = np.clip(scores, 0.0, 1.0)

        gen_best_idx = int(np.argmax(scores))
        if scores[gen_best_idx] > best_score:
            best_score = float(scores[gen_best_idx])
            best_params = candidates[gen_best_idx].copy()

        # Move mean toward best candidates
        sorted_idx = np.argsort(-scores)
        top_half = candidates[sorted_idx[: population_size // 2]]
        mean = np.mean(top_half, axis=0)

        param_history.append(mean.copy())
        score_history.append(best_score)
        all_candidates.append(candidates.copy())
        all_scores.append(scores.copy())

    return OptimizationResult(
        best_params=best_params,
        best_score=best_score,
        param_names=param_names,
        param_history=param_history,
        score_history=score_history,
        all_candidates=all_candidates,
        all_scores=all_scores,
        n_generations=n_generations,
        converged=True,
        stop_reason="tolfun: 1e-06",
    )


# ── Bespoke runner ─────────────────────────────────────────────────────


def _run_01_analysis(
    *,
    machine,
    optimization_result,
    param_overrides: Dict[str, Any],
    artifacts_subdir: str,
) -> Any:
    from shared_fixtures import (
        apply_param_overrides,
        call_node_action,
        ensure_quam_config_stub,
        get_parameters_dict,
        load_library_node,
        make_save_analysis_plot,
        patch_action_manager_register_only,
        reimport_node_to_register_actions,
    )

    ensure_quam_config_stub(machine)
    from quam_config import Quam

    with (
        patch.object(Quam, "load", return_value=machine),
        patch_action_manager_register_only(),
    ):
        node = reimport_node_to_register_actions(NODE_NAME, CALIBRATION_LIBRARY_ROOT)
        if node is None:
            node = load_library_node(NODE_NAME, CALIBRATION_LIBRARY_ROOT)

    node.machine = machine
    apply_param_overrides(node, {"simulate": False, **param_overrides})

    if node.parameters.qubit_pairs not in (None, ""):
        node.namespace["qubit_pairs"] = [
            machine.qubit_pairs[name] for name in node.parameters.qubit_pairs
        ]
    else:
        node.namespace["qubit_pairs"] = list(machine.qubit_pairs.values())

    # Per-pair optimization results dict.
    pair_names = (
        node.parameters.qubit_pairs
        if node.parameters.qubit_pairs not in (None, "")
        else [qp.name for qp in node.namespace["qubit_pairs"]]
    )
    node.results["optimization_results"] = {
        name: optimization_result for name in pair_names
    }

    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    if "fit_results" in node.results:
        call_node_action(node, "update_state")

    artifacts_dir = ARTIFACTS_BASE / artifacts_subdir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save = make_save_analysis_plot()
    figures = node.results.get("figures", {})
    for fig_name, fig in figures.items():
        save(fig, artifacts_dir, f"{fig_name}.png")

    md = [
        f"# {NODE_NAME}",
        "",
        "## Description",
        "",
        str(getattr(node, "description", "") or "").strip(),
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in sorted(get_parameters_dict(node).items()):
        md.append(f"| `{k}` | `{v}` |")

    fit_results = node.results.get("fit_results", {})
    if fit_results:
        md += ["", "## Optimisation Results"]
        for pair_name, pair_summary in fit_results.items():
            md += [
                "",
                f"### {pair_name}",
                "",
                f"- **Converged**: {pair_summary.get('converged', 'N/A')}",
                f"- **Generations**: {pair_summary.get('n_generations', 'N/A')}",
                f"- **Best score (fidelity)**: {pair_summary.get('best_score', 'N/A'):.6f}",
                "",
                "| Parameter | Value |",
                "|-----------|-------|",
            ]
            for pname, pval in pair_summary.get("best_params", {}).items():
                md.append(f"| `{pname}` | `{pval:.6g}` |")

    md += [
        "",
        "## Analysis Output",
        "",
        "![Convergence](convergence.png)",
        "",
        "![Parameter Evolution](parameter_evolution.png)",
        "",
    ]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


# ── Tests ─────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_01_cmaes_converged(minimal_quam_factory):
    """Verify analysis of a converged CMA-ES optimisation."""
    machine = minimal_quam_factory()
    assert PAIR_NAME in machine.qubit_pairs, (
        f"Test factory missing expected pair '{PAIR_NAME}'; "
        f"got {list(machine.qubit_pairs)}"
    )

    optimal_detuning = 0.03
    optimal_ramp_ns = 200.0

    opt_result = _build_synthetic_optimization_result(
        optimal_detuning=optimal_detuning,
        optimal_ramp_ns=optimal_ramp_ns,
        n_generations=15,
        population_size=8,
    )

    node = _run_01_analysis(
        machine=machine,
        optimization_result=opt_result,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 200,
            "population_size": 8,
            "max_generations": 15,
        },
        artifacts_subdir=NODE_NAME,
    )

    assert "fit_results" in node.results
    fit_results = node.results["fit_results"]
    assert PAIR_NAME in fit_results
    summary = fit_results[PAIR_NAME]
    assert summary["success"]
    assert summary["converged"]
    assert summary["best_score"] > 0.5

    best_params = summary["best_params"]
    assert abs(best_params["detuning"] - optimal_detuning) < 0.05
    assert abs(best_params["ramp_duration"] - optimal_ramp_ns) < 500

    assert "figures" in node.results
    figures = node.results["figures"]
    assert "convergence" in figures
    assert "parameter_evolution" in figures

    conv_fig = figures["convergence"]
    visible_axes = [ax for ax in conv_fig.axes if ax.get_visible()]
    assert len(visible_axes) >= 1
    assert visible_axes[0].get_xlabel() == "Generation"

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    assert (artifacts_dir / "README.md").exists()


@pytest.mark.analysis
def test_01_cmaes_low_fidelity(minimal_quam_factory):
    """Verify analysis marks failure when best fidelity is below threshold."""
    machine = minimal_quam_factory()

    from calibration_utils.cmaes.optimization import OptimizationResult

    # Build a result where fidelity never exceeds 0.5
    opt_result = OptimizationResult(
        best_params=np.array([0.0, 100.0]),
        best_score=0.45,
        param_names=["detuning", "ramp_duration"],
        param_history=[np.array([0.0, 100.0])],
        score_history=[0.45],
        all_candidates=[np.array([[0.0, 100.0], [0.01, 120.0]])],
        all_scores=[np.array([0.45, 0.42])],
        n_generations=1,
        converged=False,
        stop_reason="maxiter: 1",
    )

    node = _run_01_analysis(
        machine=machine,
        optimization_result=opt_result,
        param_overrides={
            "qubit_pairs": [PAIR_NAME],
            "num_shots": 100,
            "population_size": 2,
            "max_generations": 1,
        },
        artifacts_subdir=f"{NODE_NAME}_low_fidelity",
    )

    fit_results = node.results["fit_results"]
    assert PAIR_NAME in fit_results
    summary = fit_results[PAIR_NAME]
    assert not summary["success"]
    assert not summary["converged"]
