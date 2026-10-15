import numpy as np
from scipy.stats import chi2 as chi2_dist

from .traces import STATES


def excess_power_cost(power: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Noise-weighted excess of the residual power over its own minimum, per scan point.

    z = (P - P_min) / sqrt(err^2 + err_min^2); the cost is max(0, z - 1)^2, so points within ~1 sigma of the
    minimum are free. A state that barely reacts to the scanned parameter therefore costs ~0 everywhere and
    does not pull the pair toward its noise minimum, while a state with a real dependence has a steep cost."""
    idx = int(np.argmin(power))
    err = np.maximum(np.nan_to_num(err, nan=0.0), np.finfo(float).tiny)
    z = (power - power[idx]) / np.sqrt(err**2 + err[idx] ** 2)
    return np.maximum(z - 1.0, 0.0) ** 2


def flatness_pvalue(power: np.ndarray, err: np.ndarray) -> float:
    """p-value of the null hypothesis "the residual power is constant over the whole scan".

    chi2 = sum_i (P_i - P_w)^2 / err_i^2 with P_w the inverse-variance weighted mean, df = N - 1. It uses every
    point, so unlike a minimum-versus-rest threshold it is not biased by picking the lowest of N noisy points.
    A small p means the scan responds to the parameter."""
    err = np.maximum(np.nan_to_num(err, nan=0.0), np.finfo(float).tiny)
    weight = 1.0 / err**2
    mean = np.sum(weight * power) / np.sum(weight)
    chi2_stat = np.sum(weight * (power - mean) ** 2)
    return float(chi2_dist.sf(chi2_stat, df=len(power) - 1))


def select_points(
    values: dict[str, np.ndarray],
    power: dict[str, np.ndarray],
    err: dict[str, np.ndarray],
    current: dict[str, float],
    flatness_alpha: float,
) -> dict:
    """Scan-point index of the chosen parameter value per state, with a per-state flatness gate.

    All dicts are keyed by state ("ground", "excited"); the two scans are lockstep (point i is measured with
    both parameters at their i-th grid value).
      - A state whose scan is flat (flatness p-value > flatness_alpha) keeps its current value: its index is the
        point nearest the current value and it does not follow the other state.
      - Only one state responds: that state takes its own lowest-excess-cost point; ties go to the point nearest
        the current value.
      - Both respond: a common index minimising the summed excess cost of the two states, so the written pair is
        a pair that was measured together; ties go to the point nearest the current values.

    Returns {"idx", "p_flat", "responsive", "changed"}, each a dict keyed by state. "changed" is False for a
    state that keeps its current value (flat, or its best point is the one nearest the current value)."""
    p_flat = {s: flatness_pvalue(power[s], err[s]) for s in STATES}
    responsive = {s: p_flat[s] <= flatness_alpha for s in STATES}
    distance = {s: np.abs(values[s] - current[s]) for s in STATES}
    nearest = {s: int(np.argmin(distance[s])) for s in STATES}
    cost = {s: excess_power_cost(power[s], err[s]) for s in STATES}

    idx = dict(nearest)
    if all(responsive.values()):
        # lexsort: last key is primary.
        order = np.lexsort((distance["ground"] + distance["excited"], cost["ground"] + cost["excited"]))[0]
        idx = {s: int(order) for s in STATES}
    else:
        for s in STATES:
            if responsive[s]:
                idx[s] = int(np.lexsort((distance[s], cost[s]))[0])

    changed = {s: responsive[s] and idx[s] != nearest[s] for s in STATES}
    return {"idx": idx, "p_flat": p_flat, "responsive": responsive, "changed": changed}
