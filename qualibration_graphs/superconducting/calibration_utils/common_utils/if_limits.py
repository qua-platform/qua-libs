"""How far an MW-FEM channel's intermediate frequency may sit from its upconverter.

Each MW-FEM output port carries up to two digital upconverters (DUCs), and a DUC spans a
sub-band of about 800 MHz around its centre frequency -- so the QM specification puts the
usable intermediate frequency at +-400 MHz. Past that edge the converter gain is no longer
specified, but it does not fall off a cliff: in practice an IF out to ~500 MHz still
produces usable spectroscopy and readout.

That is the whole policy encoded here:

* ``|IF| <= IF_WARN_HZ``  -- within specification, silent.
* ``IF_WARN_HZ < |IF| <= IF_MAX_HZ`` -- outside specification but usable; say so and carry on.
* ``|IF| > IF_MAX_HZ``    -- say so more loudly, and *still* carry on.

Nothing in this module raises. A calibration that has just fitted a frequency should always
be able to write it and offer the state update; whether the resulting IF is comfortable is
information for the operator, not a gate. Refusing the write would only hide a real chip
frequency behind a hardware-reach complaint, leaving the state pointing at the stale value.

The readout side is the reason this matters in practice. An analog *input* port defines a
single ``downconverter_frequency``, so one input port acquires one ~800 MHz window no matter
how many upconverters the output has; resonators spread wider than that must be split across
input ports. Warning at the write site is what makes that visible.
"""

from __future__ import annotations

import math
from typing import Any, Optional

#: Edge of the MW-FEM specification -- half of a DUC's ~800 MHz sub-band.
IF_WARN_HZ = 400e6

#: Practical ceiling. Beyond the specification the gain is unspecified rather than absent,
#: and out to here the signal is still usable, so this bounds the "usable" message rather
#: than the code's behaviour: nothing here refuses an IF above it.
IF_MAX_HZ = 500e6


def upconverter_frequency(channel: Any) -> Optional[float]:
    """The LO an MW channel plays against, or ``None`` if it cannot be determined.

    A port carries either a single ``upconverter_frequency`` or an ``upconverters`` mapping
    that the channel indexes through its ``upconverter`` field (defaulting to 1).

    Returns ``None`` rather than propagating anything: quam attribute access can raise
    ``KeyError`` as readily as ``AttributeError`` (references, ``QuamDict``), and callers use
    this to decide whether to *report* something, never to decide whether to run.
    """
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


def check_if_reach(rf_hz: Any, lo_hz: Any, label: str = "") -> Optional[str]:
    """Message describing an out-of-reach intermediate frequency, or ``None`` when it is fine.

    Never raises: non-finite or missing frequencies simply yield ``None``, since a fit that
    produced no usable number is the fitter's business to report, not this check's.
    """
    try:
        intermediate = float(rf_hz) - float(lo_hz)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(intermediate):
        return None

    magnitude = abs(intermediate)
    if magnitude <= IF_WARN_HZ:
        return None

    prefix = f"{label}: " if label else ""
    detail = (
        f"intermediate frequency {intermediate / 1e6:+.1f} MHz "
        f"(RF {float(rf_hz) / 1e9:.4f} GHz - LO {float(lo_hz) / 1e9:.4f} GHz)"
    )
    if magnitude <= IF_MAX_HZ:
        return (
            f"{prefix}{detail} is beyond the +-{IF_WARN_HZ / 1e6:.0f} MHz MW-FEM specification. "
            "Still usable; converter gain and SFDR are no longer guaranteed. Re-centre the "
            "upconverter (and the matching downconverter_frequency) to bring it back in."
        )
    return (
        f"{prefix}{detail} is past the +-{IF_MAX_HZ / 1e6:.0f} MHz usable reach. "
        "Expect noticeably reduced signal. Re-centre the upconverter (and the matching "
        "downconverter_frequency), or move this channel to its own LO group."
    )


def check_channel_if_reach(channel: Any, label: str = "", rf_hz: Any = None) -> Optional[str]:
    """:func:`check_if_reach` for a quam MW channel, reading its RF and upconverter frequency.

    ``rf_hz`` overrides the channel's own ``RF_frequency`` -- used where the tone that is
    actually emitted sits somewhere else, as with the GEF readout's shifted frequency.
    """
    try:
        if rf_hz is None:
            rf_hz = getattr(channel, "RF_frequency", None)
        return check_if_reach(rf_hz, upconverter_frequency(channel), label)
    except Exception:
        return None


def warn_if_reach(node: Any, channel: Any, label: str = "", rf_hz: Any = None) -> Optional[str]:
    """Log an out-of-reach IF through ``node`` and return the message (``None`` when in reach).

    Purely advisory: the caller keeps whatever it just wrote, ``node.outcomes`` is untouched,
    and nothing raises. A calibration that has measured a frequency must never be brought
    down by the commentary about it -- so if the check or the logging itself fails, that is
    swallowed too, and the run continues with one fewer log line.
    """
    try:
        message = check_channel_if_reach(channel, label, rf_hz)
        if message is not None:
            node.log(message)
        return message
    except Exception:
        return None
