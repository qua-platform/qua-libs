"""Shared quality grading for optimize-by-argmax curves (GEF readout opt).

The GEF frequency and power nodes pick the argmax of a smoothed pairwise-distance
curve. Reporting ``success=True`` unconditionally wrote a dead readout's noise
argmax to state, and an optimum sitting at the sweep edge was indistinguishable
from a real interior peak. This helper grades the curve; both nodes share it so
the gates cannot drift apart.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class ArgmaxQuality:
    idx: int = -1
    x_opt: float = float("nan")
    prominence_snr: float = 0.0  # (peak - median baseline) / curve noise sigma
    interior: bool = False  # argmax not at the sweep edge
    at_edge: bool = False  # true optimum likely outside the sweep (widen + re-run)
    success: bool = False
    note: str = ""


def argmax_with_quality(x, y, *, min_snr: float = 5.0, edge_points: int = 1) -> ArgmaxQuality:
    """Grade the argmax of an optimize-me curve ``y(x)``.

    Noise sigma comes from the second difference (median |y[i-1]-2y[i]+y[i+1]|
    * 1.4826 / sqrt(6)) — a first-difference estimate counts the slope of a
    smooth, coarsely-sampled curve as noise and mis-fails clean interior peaks.

    An argmax at the sweep edge still succeeds, with a "widen the sweep" note —
    the top-of-sweep point is the best measured operating point. Only a curve
    whose peak is statistically indistinguishable from its own baseline fails.
    """
    out = ArgmaxQuality()
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 5:
        out.note = "curve too short"
        return out
    xg, yg = x[m], y[m]
    n = yg.size
    idx = int(np.argmax(yg))
    out.idx = idx
    out.x_opt = float(xg[idx])
    d2 = yg[:-2] - 2 * yg[1:-1] + yg[2:]
    med_d2 = float(np.median(np.abs(d2)))
    if med_d2 == 0.0:
        out.interior = edge_points <= idx <= n - 1 - edge_points
        out.at_edge = not out.interior
        out.note = (
            "curve is piecewise-constant (quantized/stuck readout) — " "noise sigma undefined, optimum not trustworthy"
        )
        return out
    sigma = 1.4826 * med_d2 / np.sqrt(6.0)
    baseline = float(np.median(yg))
    out.prominence_snr = (float(yg[idx]) - baseline) / sigma
    out.interior = edge_points <= idx <= n - 1 - edge_points
    out.at_edge = not out.interior
    significant = out.prominence_snr >= min_snr
    if not significant:
        out.note = (
            f"optimum not significant (prominence {out.prominence_snr:.1f}σ "
            f"< {min_snr}σ) — readout separation flat/dead over this sweep"
        )
    elif out.at_edge:
        out.note = "optimum at the sweep edge — using the top of the sweep; widen the sweep to confirm the true optimum"
    out.success = bool(significant)
    return out
