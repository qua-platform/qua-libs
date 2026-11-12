"""In-memory f-LRU element for the calibration node, kept in the QUAM state only on request.

``add_lru_element`` gives a qubit without one an ``lru`` channel on its XY port's second
upconverter. That takes three changes, none of which ``record_state_updates`` can offer as a
proposal, since it only records simple values:

* the XY port's single ``upconverter_frequency`` becomes an ``upconverters`` mapping: upconverter 1
  keeps the qubit drive LO, upconverter 2 is set ``LRU_NOMINAL_IF`` below the expected pump, or at
  the nearest edge of the port's band when that falls outside it (``lru_upconverter_frequency``);
* ``qubit.xy`` becomes ``MultiUpconverterXYDriveMW``, so it keeps reading upconverter 1;
* a qubit loaded from a state saved before the LRU classes became the default becomes
  ``LruFluxTunableTransmon`` or ``LruFixedFrequencyTransmon``, which have the ``lru`` field
  (``upgrade_to_lru_class``).

``node.save()`` writes whatever is in memory, so the node undoes these changes with
``LruAddition.remove`` unless the user asked to keep the element.
"""

from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from quam.components.pulses import SquarePulse
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from calibration_utils.common_utils.if_limits import IF_MAX_HZ
from quam_config.lru_components import (
    LRU_OPERATION,
    LRU_UPCONVERTER,
    MultiUpconverterXYDriveMW,
    f0g1_frequency,
    lru_reference_frequencies,
    upgrade_to_lru_class,
)

#: LRU intermediate frequency at the expected pump frequency; keeps the default 200 MHz span around it
#: within the MW-FEM's +-400 MHz IF specification. A wider span set by the user gets the IF-reach warning.
LRU_NOMINAL_IF = 100e6
#: Upconverter frequency range of each MW-FEM band, in Hz. Every upconverter of a port must lie inside
#: the port's band, or the QOP refuses the config: a qubit drive at 6 GHz keeps its port in band 2, so the
#: pump's upconverter cannot sit below 4.5 GHz even when the pump itself does.
MW_FEM_BANDS = {1: (50e6, 5.5e9), 2: (4.5e9, 7.5e9), 3: (6.5e9, 10.5e9)}
#: f_12 and f_01 count as spurious lines when they lie within this margin of the swept window, in Hz.
SPURIOUS_LINE_MARGIN = 150e6
#: Initial LRU pulse: a square pulse whose duration the calibration node sweeps in real time.
LRU_PULSE_LENGTH = 200
LRU_PULSE_AMPLITUDE = 0.5


@dataclass
class LruAddition:
    """What ``add_lru_element`` changed on one qubit, so it can be undone."""

    qubit: Any
    upconverter_frequency: Optional[float]
    upconverters: Optional[dict]
    xy_class: type
    qubit_class: type
    band_note: Optional[str] = None
    """Set when upconverter 2 was moved to the edge of the port's band, away from ``LRU_NOMINAL_IF``."""

    def remove(self) -> None:
        """Restore the qubit, its XY drive and its XY port as they were before ``add_lru_element``."""
        port = self.qubit.xy.opx_output
        self.qubit.lru = None
        self.qubit.__class__ = self.qubit_class
        self.qubit.xy.__class__ = self.xy_class
        port.upconverters = self.upconverters
        port.upconverter_frequency = self.upconverter_frequency


def lru_upconverter_frequency(f_lru: float, band: Optional[int]) -> float:
    """Upconverter 2 frequency for a pump at ``f_lru`` Hz on a port in MW-FEM ``band``.

    ``LRU_NOMINAL_IF`` below the pump, rounded to 1 MHz, when that lies inside the band; otherwise the
    nearest band edge, so the pump is reached with a larger IF. An unknown band leaves the nominal value.
    """
    nominal = float(round((f_lru - LRU_NOMINAL_IF) / 1e6) * 1e6)
    if band not in MW_FEM_BANDS:
        return nominal
    low, high = MW_FEM_BANDS[band]
    return min(max(nominal, low), high)


def add_lru_element(qubit, operation: str = LRU_OPERATION) -> Optional[LruAddition]:
    """Give ``qubit`` an ``lru`` channel on its XY port's second upconverter, in memory only.

    The channel is set to the expected |f0> -> |g1> frequency with a square ``operation`` pulse.

    Returns:
        The changes made, to undo with ``LruAddition.remove``; None if the qubit already has an
        ``lru`` channel, in which case nothing is changed.

    Raises:
        ValueError: when the pump lies more than ``IF_MAX_HZ`` outside the XY port's band, so no
            upconverter frequency allowed on that port reaches it.
    """
    if getattr(qubit, "lru", None) is not None:
        return None
    if type(qubit.xy) not in (XYDriveMW, MultiUpconverterXYDriveMW):
        raise TypeError(f"{qubit.name}: the LRU element needs an MW-FEM XY drive, got {type(qubit.xy).__name__}.")

    port = qubit.xy.opx_output
    f_lru = f0g1_frequency(qubit)
    band = getattr(port, "band", None)
    lo_lru = lru_upconverter_frequency(f_lru, band)
    if abs(f_lru - lo_lru) > IF_MAX_HZ:
        low, high = MW_FEM_BANDS[band]
        raise ValueError(
            f"{qubit.name}: the f0-g1 pump at {f_lru / 1e9:.4f} GHz is out of reach of the XY port, which is in "
            f"band {band} ({low / 1e9:.1f}-{high / 1e9:.1f} GHz); the nearest upconverter frequency leaves an IF of "
            f"{(f_lru - lo_lru) / 1e6:+.0f} MHz, past the {IF_MAX_HZ / 1e6:.0f} MHz usable reach."
        )
    # Raises TypeError for an unsupported qubit class before anything is changed
    qubit_class = upgrade_to_lru_class(qubit)
    addition = LruAddition(
        qubit=qubit,
        upconverter_frequency=port.upconverter_frequency,
        upconverters=None if port.upconverters is None else {k: dict(v) for k, v in port.upconverters.items()},
        xy_class=type(qubit.xy),
        qubit_class=qubit_class,
    )

    upconverters = dict(addition.upconverters or {})
    if port.upconverter_frequency is not None:
        upconverters[1] = {"frequency": port.upconverter_frequency}
    if LRU_UPCONVERTER not in upconverters:
        upconverters[LRU_UPCONVERTER] = {"frequency": int(lo_lru)}
        if abs(f_lru - LRU_NOMINAL_IF - lo_lru) > 0.5e6:
            addition.band_note = (
                f"{qubit.name}: the f0-g1 pump at {f_lru / 1e9:.4f} GHz is outside the XY port's band {band}, so "
                f"upconverter {LRU_UPCONVERTER} sits at the band edge {lo_lru / 1e9:.4f} GHz and the pump plays at "
                f"an IF of {(f_lru - lo_lru) / 1e6:+.1f} MHz; the output power there may be below nominal."
            )
    port.upconverter_frequency = None
    port.upconverters = upconverters

    qubit.xy.__class__ = MultiUpconverterXYDriveMW
    qubit.lru = MultiUpconverterXYDriveMW(
        opx_output=qubit.xy.get_raw_value("opx_output"),
        upconverter=LRU_UPCONVERTER,
        RF_frequency=f_lru,
        # No shared core: on QOP 3.7.3 a frequency update on one element of a core cancels the other's pulses
        core=None,
        operations={operation: SquarePulse(length=LRU_PULSE_LENGTH, amplitude=LRU_PULSE_AMPLITUDE, axis_angle=0)},
    )
    return addition


def spurious_lines(qubit, f_min: float, f_max: float) -> List[Tuple[str, float]]:
    """Lines other than f0-g1 whose P(f) dip can capture the fit of a pump swept over [f_min, f_max] Hz.

    * f_r/2: the pump's second harmonic fills the resonator, which Stark-shifts and dephases the qubit.
    * f_02/3: three pump photons drive g-f directly, giving a P(f) dip with a P(g) rise like f0-g1.
    * f_12 and f_01: the pump drives e-f or g-e directly; these count within ``SPURIOUS_LINE_MARGIN``
      of the window, since a strong pump broadens them.

    Returns:
        (name, frequency in Hz) of each line found, in the order above.
    """
    ref = lru_reference_frequencies(qubit)
    lines = []
    for name, frequency, margin in (
        ("f_r/2", ref.f_r / 2, 0.0),
        ("f_02/3", ref.f_02 / 3, 0.0),
        ("f_12", ref.f_12, SPURIOUS_LINE_MARGIN),
        ("f_01", ref.f_01, SPURIOUS_LINE_MARGIN),
    ):
        if f_min - margin <= frequency <= f_max + margin:
            lines.append((name, frequency))
    return lines


def prediction_outside_window(qubit, f_min: float, f_max: float) -> Optional[str]:
    """A warning when the f0-g1 prediction lies outside the swept window [f_min, f_max] Hz, else None.

    Meant for a ``qubit.lru`` element that existed before the run, whose stored frequency may come from
    an older target.
    """
    prediction = f0g1_frequency(qubit)
    if f_min <= prediction <= f_max:
        return None
    return (
        f"{qubit.name}: the existing lru element sits at {qubit.lru.RF_frequency / 1e9:.4f} GHz, so the sweep "
        f"{f_min / 1e9:.4f}-{f_max / 1e9:.4f} GHz misses the f0-g1 prediction {prediction / 1e9:.4f} GHz; "
        "remove qubit.lru or move the sweep with frequency_detuning_in_mhz."
    )
