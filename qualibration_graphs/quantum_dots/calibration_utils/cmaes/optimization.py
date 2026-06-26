from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np


@dataclass
class OptimizationResult:
    """Stores the full trajectory and outcome of a CMA-ES run."""

    best_params: np.ndarray
    """Best parameter vector found."""
    best_score: float
    """Best (highest) score achieved."""
    param_names: List[str]
    """Human-readable names for each parameter dimension."""
    param_history: List[np.ndarray]
    """Population centroids per generation, shape list of (n_params,)."""
    score_history: List[float]
    """Best score seen so far at each generation."""
    all_candidates: List[np.ndarray]
    """All evaluated candidates per generation, list of (pop_size, n_params)."""
    all_scores: List[np.ndarray]
    """All scores per generation, list of (pop_size,)."""
    n_generations: int = 0
    """Total number of generations executed."""
    converged: bool = False
    """Whether CMA-ES declared convergence before hitting max_generations."""
    stop_reason: str = ""
    """Human-readable reason for stopping."""

    def to_dict(self) -> dict:
        return {
            "best_params": self.best_params.tolist(),
            "best_score": float(self.best_score),
            "param_names": list(self.param_names),
            "param_history": [h.tolist() for h in self.param_history],
            "score_history": [float(s) for s in self.score_history],
            "all_candidates": [c.tolist() for c in self.all_candidates],
            "all_scores": [s.tolist() for s in self.all_scores],
            "n_generations": self.n_generations,
            "converged": bool(self.converged),
            "stop_reason": self.stop_reason,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OptimizationResult":
        return cls(
            best_params=np.array(d["best_params"]),
            best_score=float(d["best_score"]),
            param_names=d["param_names"],
            param_history=[np.array(h) for h in d["param_history"]],
            score_history=[float(s) for s in d["score_history"]],
            all_candidates=[np.array(c) for c in d["all_candidates"]],
            all_scores=[np.array(s) for s in d["all_scores"]],
            n_generations=int(d["n_generations"]),
            converged=bool(d["converged"]),
            stop_reason=d["stop_reason"],
        )


def run_cmaes_optimization(
    evaluate_fn: Callable[[np.ndarray], np.ndarray],
    param_names: List[str],
    x0: np.ndarray,
    sigma0: float,
    bounds: List[Tuple[float, float]],
    population_size: int = 10,
    max_generations: int = 50,
    tolx: float = 1e-6,
    tolfun: float = 1e-6,
    log_callable: Callable = print,
    *,
    progress_prefix: str | None = None,
    log_each_generation: bool = True,
) -> OptimizationResult:
    """Run CMA-ES to maximise an objective function.

    Parameters
    ----------
    evaluate_fn :
        Callable that receives a batch of candidates with shape
        ``(pop_size, n_params)`` and returns an array of scores with shape
        ``(pop_size,)`` where **higher is better**.
    param_names :
        Human-readable name for each optimised parameter.
    x0 :
        Initial guess, shape ``(n_params,)``.
    sigma0 :
        Initial step-size (standard deviation in each coordinate).
    bounds :
        ``[(lo, hi), ...]`` for each parameter.
    population_size :
        CMA-ES lambda (number of candidates per generation).
    max_generations :
        Hard cap on the number of generations.
    tolx, tolfun :
        Convergence tolerances forwarded to CMA-ES.
    log_callable :
        Logger used for per-generation status messages.
    progress_prefix :
        If set, prepended to each progress line (e.g. qubit pair name).
    log_each_generation :
        If True, log after every generation with a ``current / max`` counter.
        If False, only log start and final stop (still returns full trajectory).

    Returns
    -------
    OptimizationResult
        Full optimisation trajectory and best solution found.
    """
    import cma

    warnings.filterwarnings("ignore", category=cma.evolution_strategy.InjectionWarning)

    x0 = np.asarray(x0, dtype=float)
    n_params = len(x0)
    if len(param_names) != n_params:
        raise ValueError(
            f"param_names length ({len(param_names)}) must match x0 "
            f"length ({n_params})"
        )

    lower_bounds = [b[0] for b in bounds]
    upper_bounds = [b[1] for b in bounds]

    opts = cma.CMAOptions()
    opts.set("popsize", population_size)
    opts.set("maxiter", max_generations)
    opts.set("tolx", tolx)
    opts.set("tolfun", tolfun)
    opts.set("bounds", [lower_bounds, upper_bounds])
    opts.set("verbose", -9)  # suppress cma's own printing

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    label = f"[{progress_prefix}] " if progress_prefix else ""
    log_callable(
        f"  {label}CMA-ES start — max {max_generations} generations, "
        f"population_size={population_size}, σ₀={sigma0:g}"
    )

    param_history: List[np.ndarray] = []
    score_history: List[float] = []
    all_candidates: List[np.ndarray] = []
    all_scores: List[np.ndarray] = []
    best_score_so_far = -np.inf
    best_params_so_far = x0.copy()
    gen = 0

    while not es.stop():
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

        # CMA-ES minimises, so we negate (our convention: higher score = better).
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
