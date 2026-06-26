from __future__ import annotations

from typing import Dict, Union

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .optimization import OptimizationResult


def plot_score_convergence_on_ax(
    ax: plt.Axes,
    result: OptimizationResult,
    pair_name: str = "",
) -> None:
    """Plot score mean±std and best-so-far on a given axes.

    This is the standard CMA-ES convergence view usable both standalone
    (via ``plot_convergence``) and as a subplot in node-specific figures.
    """
    generations = np.arange(1, result.n_generations + 1)

    means = np.array([np.mean(s) for s in result.all_scores])
    stds = np.array([np.std(s) for s in result.all_scores])

    ax.fill_between(
        generations, means - stds, means + stds,
        alpha=0.2, color="C0", label="mean ± std",
    )
    ax.plot(generations, means, "o-", color="C0", markersize=4, label="mean score")
    ax.plot(
        generations, result.score_history, "s-", color="C1", markersize=4,
        label="best so far",
    )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    title = f"Score convergence — {pair_name}" if pair_name else "Score convergence"
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def plot_convergence(
    results: Union[OptimizationResult, Dict[str, OptimizationResult]],
) -> Figure:
    """Plot score mean±std and best-so-far vs. generation number.

    Accepts either a single ``OptimizationResult`` or a dict keyed by qubit
    pair name.  When multiple pairs are given each gets its own subplot column.
    """
    if isinstance(results, OptimizationResult):
        results = {"": results}

    if not results:
        fig, ax = plt.subplots()
        ax.set_title("CMA-ES convergence (no results)")
        return fig

    n_pairs = len(results)
    fig, axes = plt.subplots(
        1, n_pairs, figsize=(7 * n_pairs, 4), squeeze=False, sharey=True
    )
    axes = axes[0]

    for ax, (pair_name, result) in zip(axes, results.items()):
        plot_score_convergence_on_ax(ax, result, pair_name)

    fig.tight_layout()
    return fig


def plot_parameter_evolution(
    results: Union[OptimizationResult, Dict[str, OptimizationResult]],
) -> Figure:
    """Plot each optimised parameter vs. generation.

    Accepts either a single ``OptimizationResult`` or a dict keyed by qubit
    pair name.  When multiple pairs are given each gets its own subplot column.
    """
    if isinstance(results, OptimizationResult):
        results = {"": results}

    if not results:
        fig, ax = plt.subplots()
        ax.set_title("CMA-ES parameter evolution (no results)")
        return fig

    pair_names = list(results.keys())
    n_pairs = len(pair_names)
    n_params = len(next(iter(results.values())).param_names)

    fig, axes = plt.subplots(
        n_params, n_pairs, figsize=(7 * n_pairs, 3 * n_params),
        squeeze=False, sharex="col",
    )

    for col, pair_name in enumerate(pair_names):
        result = results[pair_name]
        generations = np.arange(1, result.n_generations + 1)

        for row, name in enumerate(result.param_names):
            ax = axes[row, col]
            centroid_vals = np.array([h[row] for h in result.param_history])
            ax.plot(
                generations, centroid_vals, "o-", markersize=4, color="C1",
                label="centroid",
            )

            for gi, candidates in enumerate(result.all_candidates):
                ax.scatter(
                    np.full(len(candidates), gi + 1),
                    candidates[:, row],
                    s=10, alpha=0.25, color="C0",
                )

            ax.axhline(
                result.best_params[row], color="C3", linestyle="--", alpha=0.6,
                label=f"best = {result.best_params[row]:.6g}",
            )

            if col == 0:
                ax.set_ylabel(name)
            ax.legend(loc="upper right", fontsize="small")
            ax.grid(True, alpha=0.3)

            if row == 0:
                title = pair_name if pair_name else "CMA-ES"
                ax.set_title(title)

        axes[-1, col].set_xlabel("Generation")

    fig.suptitle("CMA-ES parameter evolution")
    fig.tight_layout()
    return fig
