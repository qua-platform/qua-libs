"""QUAM components for the f-LRU (leakage removal unit).

An f-LRU moves a transmon out of its second excited level |f> by pumping a transition that also
puts one photon in the readout resonator (states are written |transmon level, photon number>). The
photon then leaks out of the resonator at rate kappa, so the leakage in |f> is returned to the
computational subspace.

The pump drives |f0> -> |g1> at f_02 - f_r, where f_02 is the qubit g-f frequency and f_r the readout
frequency (``calibration_utils.lru_calibration.frequencies.f0g1_frequency``). |f0> and |g1> have
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

from quam.components.macro import QubitMacro
from quam.core import quam_dataclass
from quam_builder.architecture.superconducting.components.xy_drive import XYDriveMW
from quam_builder.architecture.superconducting.qubit import FixedFrequencyTransmon, FluxTunableTransmon

__all__ = [
    "LRU_OPERATION",
    "LRU_UPCONVERTER",
    "LruFixedFrequencyTransmon",
    "LruFluxTunableTransmon",
    "LruMacro",
    "lru_wait_ns",
    "play_lru",
    "upgrade_to_lru_class",
    "validate_lru_qubits",
]

#: Upconverter of the qubit drive port that carries the LRU pump.
LRU_UPCONVERTER = 2

#: Name of the LRU pulse on ``qubit.lru``.
LRU_OPERATION = "lru"

#: Shortest wait QUA plays, in ns (4 clock cycles).
MIN_WAIT_NS = 16


def lru_wait_ns(qubit, wait_ns=None) -> int:
    """Wait after the pump as it is played, in ns: ``wait_ns``, or the resonator ``depletion_time`` when None.

    The wait lets the photon the pump leaves in the resonator decay before the next gate or the readout. It is
    rounded down to the 4 ns clock, and a wait under ``MIN_WAIT_NS`` is not played (0 is returned).
    """
    if wait_ns is None:
        wait_ns = qubit.resonator.depletion_time
    wait_ns = int(wait_ns) // 4 * 4
    return wait_ns if wait_ns >= MIN_WAIT_NS else 0


def play_lru(qubit, operation: str = LRU_OPERATION, phase_shift=None, amplitude_scale=None, wait_ns=None) -> None:
    """Play the microwave f-LRU pump of ``qubit``, wait for the resonator to empty, then rotate the qubit drive
    frame by ``phase_shift``.

    The pump is its own element, so it is aligned with the qubit drive before the pulse and after the wait;
    pulses on ``qubit.xy`` then never overlap it.

    Args:
        qubit: A qubit with an ``lru`` channel.
        operation: The pump pulse on ``qubit.lru``.
        phase_shift: Virtual Z on ``qubit.xy`` after the pump, in units of 2 pi; a float or a QUA
            expression. None or 0 plays no frame rotation.
        amplitude_scale: Optional factor on the pump amplitude; 0 plays the slot with no pump (a control).
        wait_ns: Wait after the pump, in ns; None waits the resonator ``depletion_time`` (see ``lru_wait_ns``).
    """
    from qm.qua import align

    align(qubit.xy.name, qubit.lru.name)
    qubit.lru.play(operation, amplitude_scale=amplitude_scale)
    wait = lru_wait_ns(qubit, wait_ns)
    if wait:
        qubit.lru.wait(wait // 4)  # clock cycles
    align(qubit.xy.name, qubit.lru.name)
    _virtual_z(qubit, phase_shift)


def _virtual_z(qubit, phase_shift) -> None:
    """Rotate the frame of ``qubit.xy`` by ``phase_shift`` turns; None or 0 plays nothing."""
    if phase_shift is not None and not (isinstance(phase_shift, (int, float)) and abs(phase_shift) < 1e-6):
        qubit.xy.frame_rotation_2pi(phase_shift)


@quam_dataclass
class LruMacro(QubitMacro):
    """The f-LRU as a gate: the microwave pump on ``qubit.lru``, a wait for the resonator to empty, then a
    virtual Z on ``qubit.xy``.

    Stored as ``qubit.macros["lru"]``. Node 26c calibrates ``phase_shift`` the way node 34a calibrates
    the CZ ``phase_shift_control``: a Ramsey sequence around the LRU with a swept frame rotation. Node 11b
    plays this macro when it interleaves the LRU.
    """

    operation: str = LRU_OPERATION
    """Pump pulse on ``qubit.lru``."""
    phase_shift: float = 0.0
    """Virtual Z applied to the qubit drive after the pump, in units of 2 pi; cancels the qubit phase the
    pump leaves (mostly its AC Stark shift)."""
    wait_after_ns: int = None
    """Wait after the pump, in ns, so the photon a removed leakage leaves in the resonator does not dephase
    the next gates. None waits the resonator ``depletion_time``; 0 plays no wait."""

    def apply(self, phase_shift=None, **kwargs) -> None:
        """Play the pump and the wait, then the virtual Z: ``phase_shift`` if given (float or QUA expression),
        else the stored one."""
        play_lru(
            self.qubit,
            self.operation,
            self.phase_shift if phase_shift is None else phase_shift,
            wait_ns=self.wait_after_ns,
        )

    @property
    def inferred_duration(self) -> float:
        """Pump duration plus the wait after it, in seconds."""
        pump_ns = self.qubit.lru.operations[self.operation].length
        return (pump_ns + lru_wait_ns(self.qubit, self.wait_after_ns)) * 1e-9


@quam_dataclass
class LruFluxTunableTransmon(FluxTunableTransmon):
    """Flux-tunable transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: XYDriveMW = None


@quam_dataclass
class LruFixedFrequencyTransmon(FixedFrequencyTransmon):
    """Fixed-frequency transmon with an f-LRU pump drive on its XY port's second upconverter."""

    lru: XYDriveMW = None


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
