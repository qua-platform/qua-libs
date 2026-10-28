"""Two-qubit RB implied-CZ cross-check.

The alpha -> fidelity conversion and the fidelity checks are shared with the single-qubit RB
(``calibration_utils.common_utils.rb``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from calibration_utils.common_utils.rb import (
    IRBFidelity,
    SRBFidelity,
    as_ufloat,
    clifford_fidelity_from_alpha,
    compute_irb_fidelity,
    compute_srb_fidelity,
    unwrap,
)


@dataclass
class ImpliedCZResult:
    """Result of inferring per-CZ RB decay / fidelity from 1Q + 2Q RB data
    (McKay et al. Eq. 10, https://arxiv.org/abs/1712.06550v2) — a cross-check available from Standard RB alone."""

    alpha_0: float | None
    alpha_1: float | None
    alpha_01: float | None
    alpha_01_stderr: float | None
    epg_cz: float | None
    epg_cz_stderr: float | None


def _qubit_rb_alpha(qubit) -> float | None:
    """1Q RB decay ``alpha`` for a qubit, derived from its stored gate fidelity."""
    gf = getattr(qubit, "gate_fidelity", None) or {}
    if "averaged" not in gf:
        return None
    f = float(gf["averaged"])
    if not np.isfinite(f):
        return None
    return 2.0 * f - 1.0


def _alpha_01_from_inputs(alpha_2q: Any, alpha_0: float, alpha_1: float, n_1: float, n_2: float) -> Any | None:
    """Infer per-CZ RB decay ``alpha_01`` from 2Q alpha and fixed 1Q reference alphas."""
    if n_2 <= 0 or not np.isfinite(n_1) or n_1 < 0:
        return None
    if not np.isfinite(alpha_0) or alpha_0 <= 0 or not np.isfinite(alpha_1) or alpha_1 <= 0:
        return None

    half = n_1 / 2.0
    a0, a1 = alpha_0**half, alpha_1**half
    factor_1q = (a0 + a1 + 3.0 * a0 * a1) / 5.0
    if factor_1q <= 0:
        return None

    ratio = alpha_2q / factor_1q
    if not hasattr(ratio, "s") and (not np.isfinite(ratio) or ratio <= 0):
        return None
    if hasattr(ratio, "n") and float(ratio.n) <= 0:
        return None

    return ratio ** (1.0 / n_2)


def infer_alpha_01(
    alpha_2q: float,
    qubit_control,
    qubit_target,
    n_1: float,
    n_2: float,
) -> tuple[float | None, float | None, float | None]:
    """Infer per-CZ RB decay alpha_01 by inverting Eq. (10) of supplemental material of McKay et al. (https://arxiv.org/abs/1712.06550v2)."""
    alpha_0 = _qubit_rb_alpha(qubit_control)
    alpha_1 = _qubit_rb_alpha(qubit_target)
    if alpha_0 is None or alpha_1 is None:
        return alpha_0, alpha_1, None
    if not np.isfinite(alpha_2q) or alpha_2q <= 0:
        return alpha_0, alpha_1, None

    alpha_01 = _alpha_01_from_inputs(alpha_2q, alpha_0, alpha_1, n_1, n_2)
    if alpha_01 is None:
        return alpha_0, alpha_1, None
    return alpha_0, alpha_1, unwrap(alpha_01)[0]


def compute_implied_cz(
    alpha_2q: float,
    alpha_2q_stderr: float | None,
    qubit_control,
    qubit_target,
    n_1: float | None,
    n_2: float | None,
) -> ImpliedCZResult | None:
    """Full implied-CZ pipeline: infer alpha_01, propagate error, convert to EPG."""
    if n_1 is None or n_2 is None:
        return None

    alpha_0, alpha_1, alpha_01 = infer_alpha_01(alpha_2q, qubit_control, qubit_target, n_1, n_2)

    alpha_01_stderr = None
    epg_cz = None
    epg_cz_stderr = None
    if alpha_01 is not None and alpha_0 is not None and alpha_1 is not None:
        alpha_2q_u = as_ufloat(alpha_2q, alpha_2q_stderr)
        alpha_01_u = _alpha_01_from_inputs(alpha_2q_u, alpha_0, alpha_1, n_1, n_2)
        if alpha_01_u is not None:
            _, alpha_01_stderr = unwrap(alpha_01_u)
            epg_cz, epg_cz_stderr = unwrap(1 - clifford_fidelity_from_alpha(alpha_01_u, n_qubits=2))

    return ImpliedCZResult(
        alpha_0=alpha_0,
        alpha_1=alpha_1,
        alpha_01=alpha_01,
        alpha_01_stderr=alpha_01_stderr,
        epg_cz=epg_cz,
        epg_cz_stderr=epg_cz_stderr,
    )
