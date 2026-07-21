"""Ramsey idle-time grid from frequency span and Nyquist sampling on the delay axis."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


def frequency_grid_mhz(detuning_mhz: float, f_span_mhz: float, f_step_mhz: float) -> np.ndarray:
    """Hypothesis frequencies [detuning - span, detuning + span] in MHz (inclusive of endpoints)."""
    if f_step_mhz <= 0:
        raise ValueError("f_step_in_MHz must be positive.")
    f_lo = detuning_mhz - f_span_mhz
    f_hi = detuning_mhz + f_span_mhz
    return np.arange(f_lo, f_hi + 0.5 * f_step_mhz, f_step_mhz)


def bandwidth_mhz(v_f: np.ndarray) -> float:
    """Largest |f| on the grid (MHz); used so delay sampling stays below Nyquist for cos(2π f τ)."""
    if v_f.size == 0:
        raise ValueError("Frequency grid is empty.")
    return float(np.max(np.abs(v_f)))


def _quantize_to_ns_grid(value_ns: float, quantum_ns: int = 4) -> int:
    q = int(quantum_ns)
    return max(q, int(value_ns // q) * q)


def step_ns_from_nyquist(
    bandwidth_mhz: float,
    nyquist_margin: float,
    *,
    min_step_ns: int = 4,
) -> int:
    """Uniform delay step Δτ (ns) so f_Nyquist ≥ nyquist_margin × B.

    For uniform samples τ_n = n Δτ (τ in seconds), f_Nyquist = 1/(2Δτ) [Hz]
    ⇒ f_Nyquist_MHz = 10⁹/(2 Δτ_ns) × 10⁻⁶ = 500 / Δτ_ns.
    Require 500/Δτ_ns ≥ nyquist_margin × B_MHz  ⇒  Δτ_ns ≤ 500/(nyquist_margin × B).
    """
    if bandwidth_mhz <= 0:
        raise ValueError("Bandwidth (max |f| on grid) must be positive.")
    if nyquist_margin <= 0:
        raise ValueError("nyquist_margin must be positive.")
    dt_ideal_ns = 500.0 / (nyquist_margin * bandwidth_mhz)
    dt_ns = _quantize_to_ns_grid(dt_ideal_ns, 4)
    return max(int(min_step_ns), dt_ns)


def max_idle_ns_from_frequency_step(f_step_mhz: float, *, oversampling: float = 0.5) -> float:
    """Upper bound on Ramsey delay (ns) so adjacent f bins differ by O(1) turn at τ_max.

    Uses τ_us ≈ oversampling / f_step_mhz (τ in µs), i.e. τ_ns = 1000 × oversampling / f_step_mhz.
    """
    if f_step_mhz <= 0:
        raise ValueError("f_step_in_MHz must be positive.")
    tau_us = oversampling / f_step_mhz
    return float(tau_us * 1000.0)


def build_idle_tau_ns_and_frequency_grid(
    detuning_mhz: float,
    f_span_mhz: float,
    f_step_mhz: float,
    *,
    nyquist_margin: float = 2.0,
    min_wait_time_ns: int = 16,
    max_wait_time_ns: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return ``(v_f_mhz, tau_ns, meta)`` with τ on a 4-ns grid satisfying the Nyquist margin on Δτ.

    ``tau_max`` is ``min(max_idle_ns_from_frequency_step(f_step_mhz), max_wait_time_ns)`` when
    ``max_wait_time_ns`` is given; a cap below the frequency-step bound shortens the sweep and
    weakens separation between adjacent hypothesis bins at the longest delays.
    """
    v_f = frequency_grid_mhz(detuning_mhz, f_span_mhz, f_step_mhz)
    B_mhz = bandwidth_mhz(v_f)
    step_ns = step_ns_from_nyquist(B_mhz, nyquist_margin, min_step_ns=4)
    step_ns = max(step_ns, _quantize_to_ns_grid(float(min_wait_time_ns), 4))

    tau_max_ns = max_idle_ns_from_frequency_step(f_step_mhz)
    if max_wait_time_ns is not None:
        tau_max_ns = min(tau_max_ns, float(max_wait_time_ns))

    tau_min_ns = _quantize_to_ns_grid(float(min_wait_time_ns), 4)
    tau_max_ns = max(float(tau_min_ns), tau_max_ns)
    tau_max_ns = float(_quantize_to_ns_grid(tau_max_ns, 4))

    tau_ns = np.arange(tau_min_ns, tau_max_ns + 0.5 * step_ns, step_ns, dtype=float)
    if tau_ns.size == 0:
        tau_ns = np.array([float(tau_min_ns)], dtype=float)

    meta = {
        "bandwidth_mhz": B_mhz,
        "idle_step_ns": int(step_ns),
        "tau_max_ns": float(tau_ns[-1]),
        "tau_min_ns": float(tau_ns[0]),
        "nyquist_limit_mhz": 500.0 / step_ns,
        "nyquist_margin": float(nyquist_margin),
    }
    return v_f, tau_ns, meta


def idle_clock_cycles_from_tau_ns(tau_ns: np.ndarray) -> np.ndarray:
    """OPX clock cycles (4 ns each) for ``from_array``."""
    return (tau_ns // 4).astype(int)


def sweep_from_parameters(p: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build ``(v_f_mhz, tau_ns, idle_clock_cycles, meta)`` from node parameters.

    With ``derive_idle_times`` True, ``max_wait_time_in_ns`` caps ``tau_max`` after
    ``max_idle_ns_from_frequency_step``; keep the cap at or above that bound unless a shorter sweep
    is intentional.
    """
    if p.derive_idle_times:
        v_f, tau_ns, meta = build_idle_tau_ns_and_frequency_grid(
            float(p.detuning),
            float(p.f_span_in_MHz),
            float(p.f_step_in_MHz),
            nyquist_margin=float(p.nyquist_margin),
            min_wait_time_ns=int(p.min_wait_time_in_ns),
            max_wait_time_ns=int(p.max_wait_time_in_ns),
        )
        meta["derive_idle_times"] = True
    else:
        v_f = frequency_grid_mhz(
            float(p.detuning),
            float(p.f_span_in_MHz),
            float(p.f_step_in_MHz),
        )
        idle = np.arange(
            int(p.min_wait_time_in_ns) // 4,
            int(p.max_wait_time_in_ns) // 4,
            int(p.wait_time_step_in_ns) // 4,
        )
        tau_ns = idle.astype(float) * 4.0
        if tau_ns.size == 0:
            raise ValueError(
                "Empty manual idle sweep: check min_wait_time_in_ns, max_wait_time_in_ns, wait_time_step_in_ns."
            )
        meta = {
            "derive_idle_times": False,
            "idle_step_ns": int(p.wait_time_step_in_ns),
            "bandwidth_mhz": bandwidth_mhz(v_f),
        }
    clocks = idle_clock_cycles_from_tau_ns(tau_ns)
    return v_f, tau_ns, clocks, meta
