import numpy as np


def excess_power_cost(power: np.ndarray, err: np.ndarray) -> np.ndarray:
    """Noise-weighted excess of the residual power over its own minimum, per scan point.

    z = (P - P_min) / sqrt(err^2 + err_min^2); the cost is max(0, z - 1)^2, so points within ~1 sigma of the
    minimum are free. A state that barely reacts to the scanned parameter therefore costs ~0 everywhere and
    does not pull the pair toward its noise minimum, while a state with a real dependence has a steep cost."""
    idx = int(np.argmin(power))
    err = np.maximum(np.nan_to_num(err, nan=0.0), np.finfo(float).tiny)
    z = (power - power[idx]) / np.sqrt(err**2 + err[idx] ** 2)
    return np.maximum(z - 1.0, 0.0) ** 2


def select_parameter_pair(
    values_g: np.ndarray,
    power_g: np.ndarray,
    err_g: np.ndarray,
    values_e: np.ndarray,
    power_e: np.ndarray,
    err_e: np.ndarray,
    max_difference_hz: float | None,
    geometric: bool = False,
) -> tuple[int, int] | None:
    """Scan-point indices (ground, excited) of the best parameter pair with |value_g - value_e| <= limit.

    Minimises the summed excess-power cost of the two states; ties (e.g. a flat state) go to the smaller
    difference, then to the pair closest to the current values (the centre of each grid: geometric for the
    geometric kappa scan, arithmetic for the linear zeta scan). Returns None when no pair satisfies the limit."""
    diff = np.abs(values_g[:, None] - values_e[None, :])
    cost = excess_power_cost(power_g, err_g)[:, None] + excess_power_cost(power_e, err_e)[None, :]
    feasible = np.ones_like(diff, dtype=bool) if max_difference_hz is None else diff <= max_difference_hz
    if not feasible.any():
        return None
    if geometric:
        from_current = (
            np.abs(np.log(values_g / np.sqrt(values_g[0] * values_g[-1])))[:, None]
            + np.abs(np.log(values_e / np.sqrt(values_e[0] * values_e[-1])))[None, :]
        )
    else:
        from_current = (
            np.abs(values_g - 0.5 * (values_g[0] + values_g[-1]))[:, None]
            + np.abs(values_e - 0.5 * (values_e[0] + values_e[-1]))[None, :]
        )
    ig, ie = np.nonzero(feasible)
    # lexsort: last key is primary.
    order = np.lexsort((from_current[ig, ie], diff[ig, ie], cost[ig, ie]))[0]
    return int(ig[order]), int(ie[order])
