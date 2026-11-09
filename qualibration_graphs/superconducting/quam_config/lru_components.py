"""QUAM components and transition frequencies for the f-LRU (leakage removal unit).

An f-LRU moves a transmon out of its second excited level |f> by pumping a transition that also
puts one photon in the readout resonator (states are written |transmon level, photon number>). The
photon then leaks out of the resonator at rate kappa, so the leakage in |f> is returned to the
computational subspace.

Conventions, all in Hz:

* f_01 = ``qubit.xy.RF_frequency``, the g-e frequency;
* f_12 = f_01 - |anharmonicity|, the e-f frequency (the convention used in this repo);
* f_02 = f_01 + f_12, the g-f frequency;
* f_r = ``qubit.resonator.RF_frequency``, the readout frequency, a few MHz from the bare resonator.

Two transitions are used, one per drive type:

* ``f0g1``, |f0> -> |g1>, at f_02 - f_r (``f0g1_frequency``). |f0> and |g1> have opposite parity, so
  a charge drive on the qubit XY line couples them at first order through the anharmonicity. Node 26a
  (microwave pump) drives it; the qubit ends in |g0>.
* ``f0e1``, |f0> -> |e1>, at |f_r - f_12| (``f0e1_frequency``). |f0> and |e1> have the same parity,
  so a charge drive cannot couple them, but a flux drive, which keeps parity, can. Node 26b (flux
  tone at half the transition frequency) drives it; the qubit ends in |e0>.

The microwave pump sits at about 3 to 4 GHz for f0-g1 on a typical device, 0.5 to 1.5 GHz below
the qubit drive, on upconverter 2 in the same band. Both tones share the qubit's MW-FEM output port,
each on its own digital upconverter: upconverter 1 for the qubit XY drive, upconverter 2
(``LRU_UPCONVERTER``) for the LRU pump. Node 26a adds the element qubit by qubit
(``calibration_utils.lru_calibration.element``).
"""

import logging
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple

from quam.components.channels import MWChannel
from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon, FluxTunableTransmon

__all__ = [
    "LEGACY_LRU_TRANSITION",
    "LRU_OPERATION",
    "LRU_TRANSITIONS",
    "LRU_UPCONVERTER",
    "LruFixedFrequencyTransmon",
    "LruFluxTunableTransmon",
    "LruReferenceFrequencies",
    "LruTransition",
    "LruTransitionInfo",
    "MultiUpconverterXYDriveMW",
    "f0e1_frequency",
    "f0g1_frequency",
    "lru_reference_frequencies",
]

logger = logging.getLogger(__name__)

#: Upconverter of the qubit drive port that carries the LRU pump.
LRU_UPCONVERTER = 2

#: Name of the LRU pulse on ``qubit.lru``.
LRU_OPERATION = "lru"

#: Transition an f-LRU drives: |f0> -> |g1> (charge drive, node 26a) or |f0> -> |e1> (flux drive, node 26b).
LruTransition = Literal["f0g1", "f0e1"]


@dataclass(frozen=True)
class LruTransitionInfo:
    """Display label of a transition and the population that rises when it is driven."""

    label: str
    population: str
    population_label: str


LRU_TRANSITIONS: Dict[LruTransition, LruTransitionInfo] = {
    "f0g1": LruTransitionInfo(label="|f0> -> |g1>", population="pg", population_label="P(g)"),
    "f0e1": LruTransitionInfo(label="|f0> -> |e1>", population="pe", population_label="P(e)"),
}

#: Transition assumed for a dataset that carries no ``lru_transition`` tag.
LEGACY_LRU_TRANSITION: LruTransition = "f0e1"


@quam_dataclass
class MultiUpconverterXYDriveMW(XYDriveMW):
    """MW drive whose LO is the frequency of its own ``upconverter`` on the output port.

    ``XYDriveMW`` always reads the port's single ``upconverter_frequency``, which is ``None`` once
    the port defines an ``upconverters`` mapping. This falls back to ``MWChannel``'s lookup, which
    handles both layouts.
    """

    @property
    def upconverter_frequency(self) -> float:
        """Returns the up-converter/LO frequency in Hz."""
        return MWChannel.upconverter_frequency.fget(self)


@quam_dataclass
class LruFluxTunableTransmon(FluxTunableTransmon):
    """Flux-tunable transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: MultiUpconverterXYDriveMW = None


@quam_dataclass
class LruFixedFrequencyTransmon(FixedFrequencyTransmon):
    """Fixed-frequency transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: MultiUpconverterXYDriveMW = None


class LruReferenceFrequencies(NamedTuple):
    """Qubit and resonator frequencies the LRU transitions are computed from, in Hz."""

    f_01: float
    f_12: float
    f_02: float
    f_r: float


def lru_reference_frequencies(qubit) -> LruReferenceFrequencies:
    """f_01, f_12, f_02 and f_r of ``qubit``, following the conventions of this module.

    f_12 is computed from ``qubit.anharmonicity``, the value node 26a also uses to prepare |f>. Do not use
    the look-alike fields instead: ``qubit.f_12`` is not set on any qubit, ``qubit.resonator.f_12`` is a
    readout frequency, and the EF_x180 pulse's ``anharmonicity`` is a DRAG parameter. An error in f_01
    enters f0-g1 twice, through f_02.

    Raises:
        ValueError: when the anharmonicity, the qubit frequency or the resonator frequency is not set.
    """
    required = (
        (qubit.anharmonicity, "anharmonicity", "12"),
        (qubit.xy.RF_frequency, "xy.RF_frequency", "03a"),
        (qubit.resonator.RF_frequency, "resonator.RF_frequency", "02a"),
    )
    for value, name, node in required:
        if value is None:
            raise ValueError(f"{qubit.name}: {name} is not set; calibrate it with node {node} first.")
    f_01 = float(qubit.xy.RF_frequency)
    f_12 = f_01 - abs(float(qubit.anharmonicity))
    return LruReferenceFrequencies(f_01=f_01, f_12=f_12, f_02=f_01 + f_12, f_r=float(qubit.resonator.RF_frequency))


def f0e1_frequency(qubit) -> float:
    """Expected |f0> -> |e1> frequency |f_r - f_12| in Hz, at zero drive amplitude (node 26b)."""
    ref = lru_reference_frequencies(qubit)
    return abs(ref.f_r - ref.f_12)


def f0g1_frequency(qubit) -> float:
    """Expected |f0> -> |g1> frequency f_02 - f_r in Hz, at zero drive amplitude (node 26a).

    With the resonator above f_02 the value is negative; its absolute value is returned and a warning
    is logged, since that regime is untested.
    """
    ref = lru_reference_frequencies(qubit)
    frequency = ref.f_02 - ref.f_r
    if frequency <= 0:
        logger.warning(
            f"{qubit.name}: the resonator ({ref.f_r / 1e9:.4f} GHz) lies above f_02 ({ref.f_02 / 1e9:.4f} GHz); "
            "the f0-g1 pump is then untested."
        )
    return abs(frequency)
