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

The pump drives ``f0g1``, |f0> -> |g1>, at f_02 - f_r (``f0g1_frequency``). |f0> and |g1> have
opposite parity, so a charge drive on the qubit XY line couples them at first order through the
anharmonicity. The qubit ends in |g0>.

* ``qubit.lru`` (node 26a): an MW channel playing the f0-g1 pump, in every generated config. The pump
  shifts the qubit frequency while it plays, so the qubit picks up a phase; ``qubit.macros["lru"]``
  (``LruMacro``, node 26c) plays the pump and then cancels that phase with a virtual Z on the qubit drive.

A state names ``FluxTunableTransmon`` or ``FixedFrequencyTransmon``, and QUAM keeps the class a state
names, so node 26a calls ``upgrade_to_lru_class`` before storing an LRU on such a qubit. A state saved
after that names the LRU-capable class and reloads with ``qubit.lru``.

The microwave pump sits at about 3 to 4 GHz for f0-g1 on a typical device, 0.5 to 1.5 GHz below
the qubit drive, on upconverter 2 in the same band. Both tones share the qubit's MW-FEM output port,
each on its own digital upconverter: upconverter 1 for the qubit XY drive, upconverter 2
(``LRU_UPCONVERTER``) for the LRU pump. Node 26a adds the element qubit by qubit
(``calibration_utils.lru_calibration.element``).
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple

from quam.components.channels import MWChannel
from quam.components.macro import QubitMacro
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
    "LruMacro",
    "LruReferenceFrequencies",
    "LruTransition",
    "LruTransitionInfo",
    "MultiUpconverterXYDriveMW",
    "f0g1_frequency",
    "lru_reference_frequencies",
    "play_lru",
    "upgrade_to_lru_class",
    "validate_lru_qubits",
]

logger = logging.getLogger(__name__)

#: Upconverter of the qubit drive port that carries the LRU pump.
LRU_UPCONVERTER = 2

#: Name of the LRU pulse on ``qubit.lru``.
LRU_OPERATION = "lru"

#: Transition an f-LRU drives: |f0> -> |g1> (charge drive, node 26a).
LruTransition = Literal["f0g1"]


@dataclass(frozen=True)
class LruTransitionInfo:
    """Display label of a transition and the population that rises when it is driven."""

    label: str
    population: str
    population_label: str


LRU_TRANSITIONS: Dict[LruTransition, LruTransitionInfo] = {
    "f0g1": LruTransitionInfo(label="|f0> -> |g1>", population="pg", population_label="P(g)"),
}

#: Transition assumed for a dataset that carries no ``lru_transition`` tag.
LEGACY_LRU_TRANSITION: LruTransition = "f0g1"


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


def play_lru(qubit, operation: str = LRU_OPERATION, phase_shift=None, amplitude_scale=None) -> None:
    """Play the microwave f-LRU pump of ``qubit``, then rotate the qubit drive frame by ``phase_shift``.

    The pump is its own element, so it is aligned with the qubit drive before and after the pulse;
    pulses on ``qubit.xy`` then never overlap it.

    Args:
        qubit: A qubit with an ``lru`` channel.
        operation: The pump pulse on ``qubit.lru``.
        phase_shift: Virtual Z on ``qubit.xy`` after the pump, in units of 2 pi; a float or a QUA
            expression. None or 0 plays no frame rotation.
        amplitude_scale: Optional factor on the pump amplitude; 0 plays the slot with no pump (a control).
    """
    from qm.qua import align

    align(qubit.xy.name, qubit.lru.name)
    qubit.lru.play(operation, amplitude_scale=amplitude_scale)
    align(qubit.xy.name, qubit.lru.name)
    _virtual_z(qubit, phase_shift)


def _virtual_z(qubit, phase_shift) -> None:
    """Rotate the frame of ``qubit.xy`` by ``phase_shift`` turns; None or 0 plays nothing."""
    if phase_shift is not None and not (isinstance(phase_shift, (int, float)) and abs(phase_shift) < 1e-6):
        qubit.xy.frame_rotation_2pi(phase_shift)


@quam_dataclass
class LruMacro(QubitMacro):
    """The f-LRU as a gate: the microwave pump on ``qubit.lru`` followed by a virtual Z on ``qubit.xy``.

    Stored as ``qubit.macros["lru"]``. Node 26c calibrates ``phase_shift`` the way node 34a calibrates
    the CZ ``phase_shift_control``: a Ramsey sequence around the LRU with a swept frame rotation. Node 11b
    plays this macro when it interleaves the LRU.
    """

    operation: str = LRU_OPERATION
    """Pump pulse on ``qubit.lru``."""
    phase_shift: float = 0.0
    """Virtual Z applied to the qubit drive after the pump, in units of 2 pi; cancels the qubit phase the
    pump leaves (mostly its AC Stark shift)."""

    def apply(self, phase_shift=None, **kwargs) -> None:
        """Play the pump, then the virtual Z: ``phase_shift`` if given (float or QUA expression), else the stored one."""
        play_lru(self.qubit, self.operation, self.phase_shift if phase_shift is None else phase_shift)

    @property
    def inferred_duration(self) -> float:
        """Pump duration in seconds."""
        return self.qubit.lru.operations[self.operation].length * 1e-9


@quam_dataclass
class LruFluxTunableTransmon(FluxTunableTransmon):
    """Flux-tunable transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: MultiUpconverterXYDriveMW = None


@quam_dataclass
class LruFixedFrequencyTransmon(FixedFrequencyTransmon):
    """Fixed-frequency transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: MultiUpconverterXYDriveMW = None


def upgrade_to_lru_class(qubit) -> type:
    """Give ``qubit`` the LRU-capable class of its kind, in memory, and return the class it had.

    A qubit that already has an ``lru`` field is left alone. Assigning the class back undoes the change.

    Raises:
        TypeError: for a qubit class with no LRU-capable counterpart.
    """
    original = type(qubit)
    if "lru" in {f.name for f in dataclasses.fields(qubit)}:
        return original
    if original is FluxTunableTransmon:
        qubit.__class__ = LruFluxTunableTransmon
    elif original is FixedFrequencyTransmon:
        qubit.__class__ = LruFixedFrequencyTransmon
    else:
        raise TypeError(f"{qubit.name}: no LRU support for qubit class {original.__name__}.")
    return original


def validate_lru_qubits(qubits) -> None:
    """Raise if a qubit has no stored microwave LRU to interleave.

    Raises:
        ValueError: for a qubit with no ``lru`` channel or no ``"lru"`` operation on it.
    """
    missing = [q.name for q in qubits if getattr(q, "lru", None) is None or LRU_OPERATION not in q.lru.operations]
    if missing:
        raise ValueError(
            f"No microwave LRU stored for {', '.join(missing)}: run 26a_lru_calibration on "
            f"{'it' if len(missing) == 1 else 'them'} and keep the result before interleaving the LRU."
        )


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
