"""Plan MW-FEM LO shifts so a qubit frequency sweep stays within IF reach.

Shifting the LO moves the whole emitted spectrum: the swept spectroscopy tone can
compensate via ``update_frequency(... - if_update)``, but active-reset x180 does not,
so a successful LO shift forces thermal reset. Triggering at ``IF_MAX_HZ`` (usable
reach) rather than ``IF_WARN_HZ`` (spec) lets windows that only need the 400–500 MHz
margin keep both LO and active reset.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence

import numpy as np
from numpy.typing import NDArray
from qualibration_libs.core import tracked_updates
from quam_builder.architecture.superconducting.qubit import AnyTransmon

LogCallable = Callable[[str], None]

# Spec edge (±400 MHz) and practical usable ceiling (±500 MHz) for MW-FEM IF.
IF_WARN_HZ = 400e6
IF_MAX_HZ = 500e6

# MW-FEM band lower edges for upconverter frequency (Hz).
_BAND_FLOOR_HZ = {1: 0.05e9, 2: 4.5e9, 3: 6.5e9}


def _upconverter_frequency(channel: Any) -> Optional[float]:
    """LO for an MW channel, or ``None`` if it cannot be resolved."""
    try:
        port = getattr(channel, "opx_output", None)
        if port is None:
            return None
        lo = getattr(port, "upconverter_frequency", None)
        if lo is not None:
            return float(lo)
        upconverters = getattr(port, "upconverters", None)
        if not upconverters:
            return None
        entry = upconverters.get(getattr(channel, "upconverter", None) or 1)
        if entry is None:
            return None
        frequency = entry.get("frequency") if hasattr(entry, "get") else getattr(entry, "frequency", None)
        return None if frequency is None else float(frequency)
    except Exception:
        return None


@dataclass
class LoShiftPlan:
    """Per-qubit LO-shift outcome for a detuning sweep ``dfs``."""

    if_update: List[int] = field(default_factory=list)
    """Hz offset to subtract in ``update_frequency`` (0 = no LO shift)."""

    tracked_qubits: List[AnyTransmon] = field(default_factory=list)
    """QUAM objects from ``tracked_updates``; revert in ``save_results``."""

    force_thermal_reset: bool = False
    """True if any LO was shifted (active reset would miss the qubit)."""


def plan_lo_shift_for_frequency_window(
    qubits: Sequence[AnyTransmon],
    dfs: NDArray[np.integer] | NDArray[np.floating],
    *,
    log_callable: Optional[LogCallable] = None,
) -> LoShiftPlan:
    """Decide LO shifts so ``intermediate_frequency + dfs`` stays in usable IF reach.

    Parameters
    ----------
    qubits
        Qubit-like objects with ``xy`` (and ``name``).
    dfs
        Relative frequency sweep axis in Hz (same array used in QUA ``from_array``).

    Returns
    -------
    LoShiftPlan
        ``if_update`` aligns 1:1 with ``qubits``. Mutates LO / RF via ``tracked_updates``
        when a shift is applied; caller should set ``reset_type="thermal"`` when
        ``force_thermal_reset`` is True and revert ``tracked_qubits`` after the run.
    """
    dfs = np.asarray(dfs)
    dfs_mid = int((dfs.min() + dfs.max()) / 2)
    plan = LoShiftPlan()

    for q in qubits:
        if_lo = q.xy.intermediate_frequency
        if_low = if_lo + int(dfs.min())
        if_high = if_lo + int(dfs.max())

        if if_low < -IF_MAX_HZ or if_high > IF_MAX_HZ:
            lo_now = _upconverter_frequency(q.xy)
            band = getattr(q.xy.opx_output, "band", None)
            band_floor = _BAND_FLOOR_HZ.get(band)
            lo_frequency = None if lo_now is None else lo_now + dfs_mid

            if lo_frequency is None:
                warnings.warn(
                    f"{q.name}: cannot resolve the current upconverter frequency, so the LO "
                    f"cannot be shifted. Proceeding without an LO shift — the frequency "
                    f"window will be clipped."
                )
                plan.if_update.append(0)
            elif band_floor is not None and lo_frequency < band_floor:
                warnings.warn(
                    f"{q.name}: the required LO {lo_frequency / 1e9:.3f} GHz falls below the "
                    f"band-{band} floor {band_floor / 1e9:.1f} GHz, so the upconverter cannot be "
                    f"shifted. Proceeding without an LO shift — the window will be clipped. "
                    f"Reduce frequency span or detuning."
                )
                plan.if_update.append(0)
            else:
                plan.force_thermal_reset = True
                warnings.warn(
                    "Qubit LO has been changed to reach desired detuning, "
                    "active reset will not work. Reset type changed to thermal."
                )
                plan.if_update.append(dfs_mid)
                with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                    if log_callable is not None:
                        log_callable(f"Updating {q_upd.name} LO to {lo_frequency}")
                    q_upd.xy.opx_output.upconverter_frequency = lo_frequency
                    q_upd.xy.RF_frequency += dfs_mid
                    plan.tracked_qubits.append(q_upd)
        else:
            edge = max(abs(if_low), abs(if_high))
            if edge > IF_WARN_HZ:
                warnings.warn(
                    f"{q.name}: window reaches {edge / 1e6:.0f} MHz IF, beyond the "
                    f"±{IF_WARN_HZ / 1e6:.0f} MHz specification but within the usable "
                    f"±{IF_MAX_HZ / 1e6:.0f} MHz. Measuring without an LO shift, so active reset "
                    f"stays available; converter gain is not guaranteed at the window edges."
                )
            plan.if_update.append(0)

    return plan
