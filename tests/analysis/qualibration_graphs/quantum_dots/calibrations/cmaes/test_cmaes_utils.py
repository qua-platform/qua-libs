"""Unit tests for ``calibration_utils.cmaes`` utilities.

These tests exercise the generic CMA-ES optimisation loop, analysis, and
plotting functions without any QUAM or QUA dependencies.
"""
from __future__ import annotations

import numpy as np
import pytest

from calibration_utils.cmaes.optimization import OptimizationResult, run_cmaes_optimization
from calibration_utils.cmaes.analysis import analyse_single_result, analyse_optimization
from calibration_utils.cmaes.plotting import plot_convergence, plot_parameter_evolution


# ── Helpers ────────────────────────────────────────────────────────────


def _quadratic_objective(candidates: np.ndarray) -> np.ndarray:
    """Deterministic 2D objective: peak of 1.0 at (0.5, 100.0)."""
    d = candidates[:, 0] - 0.5
    r = candidates[:, 1] - 100.0
    return 1.0 - 10.0 * d**2 - 1e-4 * r**2


def _make_result(
    best_score: float = 0.9,
    n_generations: int = 5,
    pop_size: int = 4,
    converged: bool = True,
) -> OptimizationResult:
    """Build a minimal deterministic OptimizationResult for testing."""
    rng = np.random.default_rng(0)
    param_names = ["x", "y"]
    param_history = [rng.uniform(size=2) for _ in range(n_generations)]
    score_history = [0.5 + i * (best_score - 0.5) / n_generations for i in range(1, n_generations + 1)]
    all_candidates = [rng.uniform(size=(pop_size, 2)) for _ in range(n_generations)]
    all_scores = [rng.uniform(0.3, 0.9, size=pop_size) for _ in range(n_generations)]

    return OptimizationResult(
        best_params=np.array([0.5, 100.0]),
        best_score=best_score,
        param_names=param_names,
        param_history=param_history,
        score_history=score_history,
        all_candidates=all_candidates,
        all_scores=all_scores,
        n_generations=n_generations,
        converged=converged,
        stop_reason="tolfun: 1e-06" if converged else "maxiter: 5",
    )


# ── run_cmaes_optimization tests ──────────────────────────────────────


@pytest.mark.analysis
def test_optimization_finds_maximum():
    """CMA-ES should converge close to (0.5, 100.0) on a simple quadratic."""
    result = run_cmaes_optimization(
        evaluate_fn=_quadratic_objective,
        param_names=["d", "r"],
        x0=np.array([0.0, 50.0]),
        sigma0=0.5,
        bounds=[(-1.0, 2.0), (0.0, 200.0)],
        population_size=8,
        max_generations=40,
        log_each_generation=False,
    )

    assert result.best_score > 0.9
    assert abs(result.best_params[0] - 0.5) < 0.15
    assert abs(result.best_params[1] - 100.0) < 30.0


@pytest.mark.analysis
def test_optimization_result_shapes():
    """Verify trajectory arrays have consistent shapes."""
    result = run_cmaes_optimization(
        evaluate_fn=_quadratic_objective,
        param_names=["d", "r"],
        x0=np.array([0.0, 50.0]),
        sigma0=0.5,
        bounds=[(-1.0, 2.0), (0.0, 200.0)],
        population_size=6,
        max_generations=5,
        log_each_generation=False,
    )

    assert result.n_generations >= 1
    assert len(result.param_history) == result.n_generations
    assert len(result.score_history) == result.n_generations
    assert len(result.all_candidates) == result.n_generations
    assert len(result.all_scores) == result.n_generations

    for params in result.param_history:
        assert params.shape == (2,)
    for candidates in result.all_candidates:
        assert candidates.shape[1] == 2
    for scores in result.all_scores:
        assert scores.ndim == 1

    assert result.best_params.shape == (2,)
    assert isinstance(result.best_score, float)


@pytest.mark.analysis
def test_optimization_validates_score_shape():
    """evaluate_fn returning wrong shape should raise ValueError."""

    def bad_fn(_candidates):
        return np.array([1.0])  # wrong shape if pop_size > 1

    with pytest.raises(ValueError, match="evaluate_fn must return shape"):
        run_cmaes_optimization(
            evaluate_fn=bad_fn,
            param_names=["x"],
            x0=np.array([0.0]),
            sigma0=1.0,
            bounds=[(-5.0, 5.0)],
            population_size=4,
            max_generations=2,
            log_each_generation=False,
        )


@pytest.mark.analysis
def test_optimization_param_names_mismatch():
    """Mismatched param_names length should raise ValueError."""
    with pytest.raises(ValueError, match="param_names length"):
        run_cmaes_optimization(
            evaluate_fn=_quadratic_objective,
            param_names=["only_one"],
            x0=np.array([0.0, 50.0]),
            sigma0=0.5,
            bounds=[(-1.0, 2.0), (0.0, 200.0)],
        )


# ── analyse tests ─────────────────────────────────────────────────────


@pytest.mark.analysis
def test_analyse_single_result_success():
    result = _make_result(best_score=0.85)
    summary = analyse_single_result(result)

    assert summary["success"] is True
    assert summary["converged"] is True
    assert summary["best_score"] == pytest.approx(0.85)
    assert "x" in summary["best_params"]
    assert "y" in summary["best_params"]
    assert summary["improvement"] != 0.0


@pytest.mark.analysis
def test_analyse_single_result_custom_threshold():
    result = _make_result(best_score=0.6)
    summary = analyse_single_result(result, success_threshold=0.7)
    assert summary["success"] is False

    summary2 = analyse_single_result(result, success_threshold=0.5)
    assert summary2["success"] is True


@pytest.mark.analysis
def test_analyse_optimization_single_wraps_in_dict():
    """Single OptimizationResult should be wrapped as {'': summary}."""
    result = _make_result()
    summaries = analyse_optimization(result)
    assert "" in summaries
    assert summaries[""]["best_score"] == pytest.approx(result.best_score)


@pytest.mark.analysis
def test_analyse_optimization_multi_pair():
    """Dict of results should produce per-pair summaries."""
    r1 = _make_result(best_score=0.9)
    r2 = _make_result(best_score=0.4, converged=False)
    summaries = analyse_optimization({"q1_q2": r1, "q3_q4": r2})

    assert "q1_q2" in summaries
    assert "q3_q4" in summaries
    assert summaries["q1_q2"]["success"] is True
    assert summaries["q3_q4"]["success"] is False


@pytest.mark.analysis
def test_analyse_optimization_multi_pair_custom_threshold():
    r1 = _make_result(best_score=0.7)
    r2 = _make_result(best_score=0.65)
    summaries = analyse_optimization(
        {"a": r1, "b": r2}, success_threshold=0.68
    )
    assert summaries["a"]["success"] is True
    assert summaries["b"]["success"] is False


# ── plotting tests ────────────────────────────────────────────────────


@pytest.mark.analysis
def test_plot_convergence_single():
    result = _make_result()
    fig = plot_convergence(result)
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) >= 1
    assert axes[0].get_xlabel() == "Generation"


@pytest.mark.analysis
def test_plot_convergence_multi_pair():
    r1 = _make_result()
    r2 = _make_result()
    fig = plot_convergence({"pair_A": r1, "pair_B": r2})
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 2
    assert "pair_A" in axes[0].get_title()
    assert "pair_B" in axes[1].get_title()


@pytest.mark.analysis
def test_plot_convergence_empty():
    fig = plot_convergence({})
    assert fig is not None


@pytest.mark.analysis
def test_plot_parameter_evolution_single():
    result = _make_result()
    fig = plot_parameter_evolution(result)
    assert fig is not None
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) >= 2  # 2 params


@pytest.mark.analysis
def test_plot_parameter_evolution_multi_pair():
    r1 = _make_result()
    r2 = _make_result()
    fig = plot_parameter_evolution({"pair_A": r1, "pair_B": r2})
    assert fig is not None
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 4  # 2 params * 2 pairs


@pytest.mark.analysis
def test_plot_parameter_evolution_empty():
    fig = plot_parameter_evolution({})
    assert fig is not None
