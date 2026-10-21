"""T1/T2 coherence-limited error per gate for two-qubit RB reporting."""

from __future__ import annotations

import numpy as np


def coherence_limit(T1_list, T2_list, gatelen: float) -> float:
    """2Q error per gate (1 - average gate fidelity) from the T1/T2 limit.

    See qiskit-ignis randomized_benchmarking rb_utils.
    ``gatelen`` and ``T1``/``T2`` must use the same time unit (seconds).
    """
    T1 = np.array(T1_list)
    T2 = np.array(T2_list)

    if len(T1) != 2 or len(T2) != 2:
        raise ValueError("T1 and T2 must each have length 2")

    T1factor = 0.0
    T2factor = 0.0
    for i in range(2):
        T1factor += 1.0 / 15.0 * np.exp(-gatelen / T1[i])
        T2factor += 2.0 / 15.0 * (np.exp(-gatelen / T2[i]) + np.exp(-gatelen * (1.0 / T2[i] + 1.0 / T1[1 - i])))
    T1factor += 1.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T1))
    T2factor += 4.0 / 15.0 * np.exp(-gatelen * np.sum(1 / T2))
    return float(0.75 * (1.0 - T1factor - T2factor))


def cz_gate_duration_s(qp, operation: str) -> float | None:
    """CZ macro duration in seconds (pulse length in ns)."""
    try:
        macro = qp.macros[operation]
    except (KeyError, TypeError):
        return None

    duration = getattr(macro, "duration", None)
    if isinstance(duration, (int, float)) and duration > 0:
        return float(duration) * 1e-9

    flux_pulse = getattr(macro, "flux_pulse_qubit", None)
    if flux_pulse is not None:
        length = getattr(flux_pulse, "length", None)
        if isinstance(length, (int, float)) and length > 0:
            return float(length) * 1e-9

    return None


def try_coherence_limit_epg(qp, operation: str) -> float | None:
    """Coherence-limited EPG for a qubit pair, or None if QUAM data is missing."""
    qc = qp.qubit_control
    qt = qp.qubit_target
    T1_list = [qc.T1, qt.T1]
    T2_list = [qc.T2echo, qt.T2echo]
    gate_len = cz_gate_duration_s(qp, operation)

    if gate_len is None or None in T1_list or None in T2_list:
        return None
    if gate_len <= 0 or any(t is not None and t <= 0 for t in T1_list + T2_list):
        return None

    return coherence_limit(T1_list, T2_list, gate_len)
