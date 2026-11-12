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

Each qubit has one slot per technique, so both can be calibrated on the same qubit:

* ``qubit.lru`` (node 26a): an MW channel playing the f0-g1 pump, in every generated config. The pump
  shifts the qubit frequency while it plays, so the qubit picks up a phase; ``qubit.macros["lru"]``
  (``LruMacro``, node 26c) plays the pump and then cancels that phase with a virtual Z on the qubit drive.
* ``qubit.flux_lru`` (node 26b): a ``FluxLru`` record of the f0-e1 flux tone. It is not a channel: the
  tone needs the flux port at 2 GS/s, so only a program that asks for it gets the element, through
  ``Quam.generate_config(flux_lru_qubits=...)`` (``quam_config.flux_lru_config``). The tone also moves
  the qubit frequency while it plays, so ``qubit.macros["flux_lru"]`` (``FluxLruMacro``, node 26c with
  ``lru_type="flux"``) plays it and then cancels that phase with a virtual Z, as ``LruMacro`` does.

New states are built with the LRU-capable classes (``Quam.qubit_type``). A state saved earlier names
``FluxTunableTransmon`` or ``FixedFrequencyTransmon``, and QUAM keeps the class a state names, so
nodes 26a and 26b call ``upgrade_to_lru_class`` before storing an LRU on such a qubit.

The microwave pump sits at about 3 to 4 GHz for f0-g1 on a typical device, 0.5 to 1.5 GHz below
the qubit drive, on upconverter 2 in the same band. Both tones share the qubit's MW-FEM output port,
each on its own digital upconverter: upconverter 1 for the qubit XY drive, upconverter 2
(``LRU_UPCONVERTER``) for the LRU pump. Node 26a adds the element qubit by qubit
(``calibration_utils.lru_calibration.element``).
"""

import dataclasses
import logging
from dataclasses import dataclass
from typing import Dict, Literal, NamedTuple, Optional

from quam.components.channels import MWChannel
from quam.components.macro import QubitMacro
from quam.core import QuamComponent, quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon, FluxTunableTransmon

__all__ = [
    "FLUX_LRU_MACRO",
    "FLUX_LRU_OPERATION",
    "FluxLru",
    "FluxLruMacro",
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
    "f0e1_frequency",
    "f0g1_frequency",
    "lru_reference_frequencies",
    "play_flux_lru",
    "play_lru",
    "upgrade_to_lru_class",
    "validate_flux_lru_qubits",
    "validate_lru_qubits",
]

logger = logging.getLogger(__name__)

#: Upconverter of the qubit drive port that carries the LRU pump.
LRU_UPCONVERTER = 2

#: Name of the LRU pulse on ``qubit.lru``.
LRU_OPERATION = "lru"

#: Name of the flux-LRU tone's operation, on the element ``Quam.generate_config(flux_lru_qubits=...)`` adds.
FLUX_LRU_OPERATION = "flux_lru"

#: Key of the flux-LRU gate (``FluxLruMacro``) in ``qubit.macros``; the microwave LRU gate uses ``LRU_OPERATION``.
FLUX_LRU_MACRO = "flux_lru"

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
class FluxLru(QuamComponent):
    """Calibrated flux-activated f-LRU (node 26b): a constant flux tone at half the |f0> -> |e1> frequency.

    Only the values are stored. The element that plays them exists only in a config generated with
    ``Quam.generate_config(flux_lru_qubits=[...])``, which switches the qubit's flux port to 2 GS/s.
    """

    transition_frequency: float
    """|f0> -> |e1> drive frequency in Hz, Stark-shifted at ``amplitude_v``; the tone plays at half of it."""
    amplitude_v: float
    """Tone amplitude at the flux port, in V, on top of the joint flux offset."""
    duration_ns: int
    """Tone duration in ns."""

    @property
    def tone_frequency(self) -> float:
        """Frequency of the flux tone in Hz: half the |f0> -> |e1> frequency."""
        return self.transition_frequency / 2

    @property
    def element_name(self) -> str:
        """Name of the flux-LRU element in a config generated with this qubit in ``flux_lru_qubits``."""
        return f"{self.parent.name}.flux_lru"

    def play(self, amplitude_scale=None, duration=None) -> None:
        """Play the flux tone; the program's config must come from ``generate_config(flux_lru_qubits=...)``.

        Args:
            amplitude_scale: Optional QUA ``amp()`` factor on ``amplitude_v``.
            duration: Optional duration in clock cycles (4 ns), replacing ``duration_ns``.
        """
        from qm.qua import amp, play

        pulse = FLUX_LRU_OPERATION if amplitude_scale is None else FLUX_LRU_OPERATION * amp(amplitude_scale)
        play(pulse, self.element_name, duration=duration)


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


def play_flux_lru(qubit, phase_shift=None, amplitude_scale=None, padding_ns: int = 0) -> None:
    """Play the flux-LRU tone of ``qubit``, then rotate the qubit drive frame by ``phase_shift``.

    The program's config must come from ``flux_lru_config.generate_flux_lru_config``. The tone shares
    the flux port with ``qubit.z``, so it is aligned with the qubit drive and the flux line before and after.

    Args:
        qubit: A qubit with a calibrated ``flux_lru``.
        phase_shift: Virtual Z on ``qubit.xy`` after the tone, in units of 2 pi; a float or a QUA
            expression. None or 0 plays no frame rotation.
        amplitude_scale: Optional factor on the tone amplitude; 0 plays the slot with no tone (a control).
        padding_ns: Idle time on the flux-LRU element before and after the tone, in ns (multiple of 4, 0 or
            >= 16), so a tone that reaches the qubit early or late does not overlap the drive pulses around it.
    """
    from qm.qua import align, wait

    element = qubit.flux_lru.element_name
    elements = (qubit.xy.name, qubit.z.name, element)
    align(*elements)
    if padding_ns:
        wait(int(padding_ns) // 4, element)
    qubit.flux_lru.play(amplitude_scale=amplitude_scale)
    if padding_ns:
        wait(int(padding_ns) // 4, element)
    align(*elements)
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
class FluxLruMacro(QubitMacro):
    """The flux-activated f-LRU as a gate: the flux tone of ``qubit.flux_lru``, then a virtual Z on ``qubit.xy``.

    Stored as ``qubit.macros["flux_lru"]``. The tone modulates the qubit frequency, and since that frequency
    is quadratic in flux near the sweet spot, the mean frequency drops while the tone plays; ``phase_shift``
    cancels the phase this leaves. Node 26c (``lru_type="flux"``) calibrates it, and node 11b plays this macro
    when it interleaves ``flux_lru``. Like the tone, it needs a config from
    ``Quam.generate_config(flux_lru_qubits=[...])``.
    """

    phase_shift: float = 0.0
    """Virtual Z applied to the qubit drive after the tone, in units of 2 pi."""
    padding_ns: int = 0
    """Idle time before and after the tone, in ns (multiple of 4, 0 or >= 16). The tone shifts the qubit by tens
    of MHz, so a few ns of timing mismatch between the flux and drive ports (the flux port runs at 2 GS/s, its
    delay is calibrated at 1 GS/s) would detune the neighbouring drive pulses; the padding keeps them apart."""

    def apply(self, phase_shift=None, amplitude_scale=None, **kwargs) -> None:
        """Play the tone, then the virtual Z: ``phase_shift`` if given (float or QUA expression), else the stored one."""
        play_flux_lru(
            self.qubit, self.phase_shift if phase_shift is None else phase_shift, amplitude_scale, self.padding_ns
        )

    @property
    def inferred_duration(self) -> float:
        """Duration of the gate in seconds: the tone and its padding."""
        return (self.qubit.flux_lru.duration_ns + 2 * self.padding_ns) * 1e-9


@quam_dataclass
class LruFluxTunableTransmon(FluxTunableTransmon):
    """Flux-tunable transmon with an f-LRU pump on its XY port's second upconverter and a flux-LRU tone."""

    lru: MultiUpconverterXYDriveMW = None
    flux_lru: Optional[FluxLru] = None


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


def validate_flux_lru_qubits(qubits) -> None:
    """Raise if a qubit has no calibrated flux LRU (``qubit.flux_lru``, node 26b) to play.

    Raises:
        ValueError: for a qubit with no ``flux_lru``.
    """
    missing = [q.name for q in qubits if getattr(q, "flux_lru", None) is None]
    if missing:
        raise ValueError(
            f"No flux LRU stored for {', '.join(missing)}: run 26b_flux_lru_calibration on "
            f"{'it' if len(missing) == 1 else 'them'} and keep the result first."
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
