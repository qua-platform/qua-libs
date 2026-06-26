from __future__ import annotations

from typing import Callable, Dict, Union

from .optimization import OptimizationResult


def analyse_single_result(
    result: OptimizationResult,
    *,
    success_threshold: float = 0.5,
) -> Dict:
    """Summarise the outcome of a single CMA-ES optimisation run.

    Parameters
    ----------
    result :
        The optimisation result to summarise.
    success_threshold :
        Minimum ``best_score`` to consider the run successful.
        Default is 0.5 (better than random binary guess).

    Returns a plain dict suitable for serialisation into ``node.results``.
    """
    summary: Dict = {
        "success": bool(result.best_score > success_threshold),
        "converged": bool(result.converged),
        "n_generations": int(result.n_generations),
        "stop_reason": result.stop_reason,
        "best_score": float(result.best_score),
        "best_params": {
            name: float(val)
            for name, val in zip(result.param_names, result.best_params)
        },
    }

    if len(result.score_history) >= 2:
        summary["initial_score"] = float(result.score_history[0])
        summary["improvement"] = float(
            result.best_score - result.score_history[0]
        )
    else:
        summary["initial_score"] = float(result.best_score)
        summary["improvement"] = 0.0

    return summary


def analyse_optimization(
    results: Union[OptimizationResult, Dict[str, OptimizationResult]],
    *,
    success_threshold: float = 0.5,
) -> Dict[str, Dict]:
    """Summarise the outcome of one or more CMA-ES optimisation runs.

    Parameters
    ----------
    results :
        A single ``OptimizationResult`` or a dict keyed by qubit pair name.
    success_threshold :
        Minimum ``best_score`` to consider the run successful.

    Returns
    -------
    Dict[str, Dict]
        Always a dict keyed by pair name (single results are keyed ``""``),
        each value being the per-pair summary dict.
    """
    if isinstance(results, OptimizationResult):
        results = {"": results}

    return {
        pair_name: analyse_single_result(result, success_threshold=success_threshold)
        for pair_name, result in results.items()
    }


def log_optimization_results(
    results: Union[OptimizationResult, Dict[str, OptimizationResult]],
    log_callable: Callable = print,
) -> None:
    """Log a human-readable summary of the optimisation(s)."""
    if isinstance(results, OptimizationResult):
        results = {"": results}

    for pair_name, result in results.items():
        summary = analyse_single_result(result)
        prefix = f"  [{pair_name}] " if pair_name else "  "

        log_callable(
            f"{prefix}CMA-ES "
            f"{'converged' if summary['converged'] else 'did not converge'} "
            f"after {summary['n_generations']} generations."
        )
        log_callable(f"{prefix}Best score: {summary['best_score']:.6f}")
        if summary["improvement"] != 0.0:
            log_callable(
                f"{prefix}Improvement over first generation: "
                f"{summary['improvement']:+.6f}"
            )
        for name, val in summary["best_params"].items():
            log_callable(f"{prefix}  {name} = {val:.6g}")
