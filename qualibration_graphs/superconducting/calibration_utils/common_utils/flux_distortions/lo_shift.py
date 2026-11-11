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

from calibration_utils.common_utils.if_limits import upconverter_frequency

LogCallable = Callable[[str], None]

# Spec edge (±400 MHz) and practical usable ceiling (±500 MHz) for MW-FEM IF.
IF_WARN_HZ = 400e6
IF_MAX_HZ = 500e6

# MW-FEM band lower edges for upconverter frequency (Hz).
_BAND_FLOOR_HZ = {1: 0.05e9, 2: 4.5e9, 3: 6.5e9}


def _set_upconverter_frequency(qubit: AnyTransmon, tracked_qubit: Any, lo_frequency: float) -> None:
    """Move the LO of ``qubit.xy`` to ``lo_frequency`` through ``tracked_qubit``, so the change can be reverted.

    A port with a single ``upconverter_frequency`` gets the new value there. A port with an ``upconverters``
    mapping (an f-LRU pump on upconverter 2, for instance) gets the new value in the drive's own entry only:
    writing ``upconverter_frequency`` there would put both keys in the port config and drag the other
    channel along. Only the ``frequency`` entry is written, since ``tracked_updates`` cannot snapshot the
    whole mapping (deep-copying a ``QuamDict`` recurses) and reverts through ``setattr``, which a
    ``QuamDict`` maps to its string keys but not to the integer upconverter keys.
    """
    port = qubit.xy.opx_output
    if port.upconverter_frequency is not None or not port.upconverters:
        tracked_qubit.xy.opx_output.upconverter_frequency = lo_frequency
        return
    upconverter = getattr(qubit.xy, "upconverter", None) or 1
    tracked_qubit.xy.opx_output.upconverters[upconverter]["frequency"] = lo_frequency


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
    if_base_hz: Optional[Sequence[float]] = None,
    log_callable: Optional[LogCallable] = None,
) -> LoShiftPlan:
    """Decide LO shifts so ``if_base + dfs`` stays in usable IF reach.

    Parameters
    ----------
    qubits
        Qubit-like objects with ``xy`` (and ``name``). LO shifts apply to these channels.
    dfs
        Relative frequency sweep axis in Hz (same array used in QUA ``from_array``).
    if_base_hz
        Optional per-qubit IF centre for the sweep (same length as ``qubits``).
        Defaults to each qubit's ``xy.intermediate_frequency``. Use this for three-tone
        spectroscopy where the drive IF is ``coupler_RF - LO`` rather than the qubit idle IF.

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

    for i, q in enumerate(qubits):
        if_lo = int(if_base_hz[i]) if if_base_hz is not None else q.xy.intermediate_frequency
        if_low = if_lo + int(dfs.min())
        if_high = if_lo + int(dfs.max())

        if if_low < -IF_MAX_HZ or if_high > IF_MAX_HZ:
            lo_now = upconverter_frequency(q.xy)
            band = getattr(q.xy.opx_output, "band", None)
            band_floor = _BAND_FLOOR_HZ.get(band)
            # Move the LO to the sweep centre so residual IF is ``dfs - dfs_mid`` (near 0).
            shift = int(if_lo) + dfs_mid
            lo_frequency = None if lo_now is None else lo_now + shift

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
                plan.if_update.append(shift)
                with tracked_updates(q, auto_revert=False, dont_assign_to_none=False) as q_upd:
                    if log_callable is not None:
                        log_callable(f"Updating {q_upd.name} LO to {lo_frequency}")
                    _set_upconverter_frequency(q, q_upd, lo_frequency)
                    q_upd.xy.RF_frequency += shift
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
