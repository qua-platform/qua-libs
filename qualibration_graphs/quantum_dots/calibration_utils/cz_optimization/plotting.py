"""Plotting utilities for the CZ gate optimisation node."""

import matplotlib.pyplot as plt
import numpy as np


def plot_circuit_probabilities_on_ax(
    ax: plt.Axes,
    circuit_history: dict,
    include_quadrature: bool,
    pair_name: str = "",
) -> None:
    """Plot best-candidate circuit probabilities vs generation."""
    if not circuit_history or not circuit_history.get("circuit_1"):
        ax.set_title(f"No circuit data — {pair_name}")
        return

    n_gen = len(circuit_history["circuit_1"])
    generations = np.arange(1, n_gen + 1)

    ax.plot(generations, circuit_history["circuit_1"], "o-",
            color="C0", markersize=4, label="C1: X(T)/2-CZ-X(T)/2")
    ax.plot(generations, circuit_history["circuit_2"], "s-",
            color="C1", markersize=4, label="C2: X(C)/2-CZ-X(C)/2")
    ax.plot(generations, circuit_history["circuit_3"], "D-",
            color="C2", markersize=4, label="C3: CZ only")

    if include_quadrature:
        if circuit_history.get("circuit_4"):
            ax.plot(generations, circuit_history["circuit_4"], "^--",
                    color="C3", markersize=4, label="C4: Y(T)/2-CZ-X(T)/2")
        if circuit_history.get("circuit_5"):
            ax.plot(generations, circuit_history["circuit_5"], "v--",
                    color="C4", markersize=4, label="C5: Y(C)/2-CZ-X(C)/2")

    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Probability")
    title = (
        f"Best-candidate circuit probabilities — {pair_name}" if pair_name
        else "Best-candidate circuit probabilities"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)


def plot_individual_scores_on_ax(
    ax: plt.Axes,
    circuit_history: dict,
    pair_name: str = "",
) -> None:
    """Scatter all candidate scores per generation with running best."""
    if not circuit_history or not circuit_history.get("all_scores"):
        ax.set_title(f"No score data — {pair_name}")
        return

    n_gen = len(circuit_history["all_scores"])
    for gen_idx in range(n_gen):
        gen_num = gen_idx + 1
        scores = np.asarray(circuit_history["all_scores"][gen_idx])
        ax.scatter(
            np.full_like(scores, gen_num, dtype=float), scores,
            s=10, alpha=0.3, color="C7", zorder=1,
        )

    generations = np.arange(1, n_gen + 1)
    if circuit_history.get("running_best_score"):
        ax.plot(
            generations, circuit_history["running_best_score"],
            "D-", color="C2", markersize=5, linewidth=2,
            label="Running best", zorder=3,
        )

    ax.set_xlabel("Generation")
    ax.set_ylabel("Score")
    title = (
        f"Individual scores & running best — {pair_name}" if pair_name
        else "Individual scores & running best"
    )
    ax.set_title(title)
    ax.legend(fontsize="small")
    ax.grid(True, alpha=0.3)
