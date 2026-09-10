"""
Define custom state macros for the quantum-dots QUAM.

This module is the place to implement pair-level custom initialize and
measure behaviour. The macros defined here are later wired onto selected
``QuantumDotPair`` objects by ``populate_macros.py`` and can be controlled
from nodes through the matching parameter mixins in
``my_macro_parameters.py``.

When customizing this file, the important contract is the exported names:
- ``InitializeMacro``
- ``MeasureMacro``

These are the classes that downstream wiring code imports and attaches to the
machine. You can keep the provided examples, adapt them, or replace them with
your own implementations, but the exported names should remain the same.

You can also define your own custom-named macro classes and export those
through ``__all__`` instead. If you do that, make sure
``my_macro_parameters.py`` and ``populate_macros.py`` are updated to match the
new exported macro names and parameter fields.

The examples in this file show two common patterns:
- ``InitializeMacroBase`` implements a voltage-balanced round-trip
  initialization sequence.
- ``InitializeMacro`` builds on that base to demonstrate an active-reset /
  heralded initialization flow.
If you would like to implement your own initialize/measure behaviour, make sure to 
comment these examples out. 

Because these state macros are wired at the ``QuantumDotPair`` level, calls
such as ``qubit.initialize()``, ``qubit_pair.initialize()``, and
``dot_pair.initialize()`` all resolve to the same underlying pair macro.
"""

from typing import Optional, Literal

from quam_builder.architecture.quantum_dots.operations import CustomMacro
from quam.core import quam_dataclass

from qm.qua import align, strict_timing_, assign, declare, if_, while_, Cast

__all__ = ["InitializeMacro", "MeasureMacro"]

# Defaults that you can write yourself.

# NOTE: This script, by default, emits the heralded/active initialisation as an example.
# If this is not your desired behaviour, make sure to edit the examples section below.


@quam_dataclass
class InitializeMacro(CustomMacro):
    @property
    def inferred_duration(self) -> float | None:
        return 0

    def apply(self):
        pass


@quam_dataclass
class MeasureMacro(CustomMacro):
    @property
    def inferred_duration(self) -> float | None:
        return 0

    def apply(self):
        pass


# -------- EXAMPLES ---------

# First we create a BalancedInitializeMacro, which describes a balanced round trip.
# The real macro that we export from here should be names InitializeMacro, which is listed below.


@quam_dataclass
class InitializeMacroBase(CustomMacro):
    """Balanced round-trip: ramp 0 → -V → +V → 0 through a named voltage point.

    Shape (per channel):

        0  ──ramp──▶  -V  ──hold──  -V  ──ramp──▶  +V  ──hold──  +V  ──ramp──▶  0

    Ramp 1 and ramp 3 are mirror triangles of each other; ramp 2 is
    antisymmetric about 0 V and integrates to zero. The two holds are
    equal, so their +V and -V contributions cancel. Net integrated
    voltage: zero on every channel.

    Ramp 2 covers twice the voltage of ramps 1 and 3, so its duration is
    ``2 * ramp_duration`` to preserve the same slope (consistent dV/dt).
    """

    zero_duration: int = 100
    ramp_duration: int = 500
    hold_duration: int = 500
    point_name: str = "initialize"

    @property
    def inferred_duration(self) -> float | None:
        return (4 * self.ramp_duration + 2 * self.hold_duration + self.zero_duration) * 1e-9

    def apply(
        self,
        ramp_duration: int | None = None,
        hold_duration: int | None = None,
        zero_duration: int | None = None,
        point_name: str | None = None,
        **kwargs,
    ):
        owner = self.owner  # The QuantumDotPair object that is the ultimate owner of this macro

        # Check if any arguments have been passed to the macro, and make sure to fall back to the
        # class attribute as a default.
        ramp = self.ramp_duration if ramp_duration is None else ramp_duration
        hold = self.hold_duration if hold_duration is None else hold_duration
        zero = self.zero_duration if zero_duration is None else zero_duration
        point_name = self.point_name if point_name is None else point_name

        # Create dicts of positive and negative voltage points
        positive_voltages = self.point_voltages(point_name)
        negative_voltages = {k: -v for k, v in positive_voltages.items()}
        zero_voltages = {k: 0.0 for k, _ in positive_voltages.items()}

        # This macro operates using the VoltageSequence
        vs = owner.voltage_sequence
        gates = [ch_name for ch_name in vs.gate_set.channels.keys()]

        # Align all the gates before the start of the sequence
        align(*gates)

        with strict_timing_():
            vs.ramp_to_voltages(
                negative_voltages,
                duration=hold,
                ramp_duration=ramp,
                ensure_align=False,
            )
            vs.ramp_to_voltages(
                positive_voltages,
                duration=hold,
                ramp_duration=2 * ramp,
                ensure_align=False,
            )
            vs.ramp_to_voltages(
                zero_voltages,
                duration=zero,
                ramp_duration=ramp,
                ensure_align=False,
            )


@quam_dataclass
class InitializeMacro(InitializeMacroBase):
    """
    An active reset initialize scheme, built on the BalancedInitializeMacro.

    The flow:
    - Initialize using the BalancedInitializeMacro
    - Measure the state
    - If the state is NOT the desired state, drive the specified qubit, and repeat the above
    - If the state is the desired state, exit the loop

    This class also optionally allows one to extract the number of loops performed as a stream
    """

    max_loops: int = 2
    return_n_loops: bool = False
    target_state: Literal[0, 1] = 0
    qubit_role: Literal["target", "control"] = "control"

    @property
    def inferred_duration(self) -> float | None:
        single_initialize_trip = (4 * self.ramp_duration + 2 * self.hold_duration + 16) * 1e-9
        measure_macro_duration = self.owner.macros["measure"].inferred_duration
        if measure_macro_duration is None:
            return None

        max_loops = self.max_loops

        # Change this to match your qubit's drive length in seconds if you want a tighter bound.
        estimated_qubit_drive_duration = None
        length = single_initialize_trip + measure_macro_duration
        if estimated_qubit_drive_duration is not None:
            length += estimated_qubit_drive_duration

        return length * max_loops

    def apply(
        self,
        max_loops: Optional[int] = None,
        target_state: Optional[Literal[0, 1]] = None,
        return_n_loops: bool | None = None,
        operation: str = "x180",
        qubit_role: Optional[Literal["target", "control"]] = None,
        qubit_name: Optional[str] = None,
        meas_ramp_duration: Optional[int] = None,
        meas_buffer_duration: Optional[int] = None,
        **kwargs,
    ):
        owner = self.owner

        if qubit_name is None:
            qubit_role = self.qubit_role if qubit_role is None else qubit_role
            # Extract the qubit pair whose quantum_dot_pair is the owner
            qubit_pair = next(qp for qp in owner.machine.qubit_pairs.values() if qp.quantum_dot_pair is owner)
            qubit_name = getattr(qubit_pair, f"qubit_{qubit_role}", None)
            if qubit_name is None:
                raise ValueError("Failed to resolve qubit")

        target_state = self.target_state if target_state is None else target_state
        max_loops = self.max_loops if max_loops is None else max_loops
        return_n_loops = self.return_n_loops if return_n_loops is None else return_n_loops

        vs = owner.voltage_sequence
        gates = [ch_name for ch_name in vs.gate_set.channels.keys()]
        loop_start_n, loop_start_bool = 0, True

        n_count = declare(int)
        assign(n_count, loop_start_n)

        cond = declare(bool)
        assign(cond, loop_start_bool)

        with while_((cond) & (n_count < max_loops)):

            # First initialise. super() should be BalancedInitializeMacro
            super().apply(**kwargs)

            # Now measure the state
            state = owner.measure(
                return_iq=False,
                ramp_duration=meas_ramp_duration,
                buffer_duration=meas_buffer_duration,
            )

            # As long as the state is in the initial value, the loop will continue until max_loops
            assign(cond, Cast.to_bool(state - target_state))
            assign(n_count, n_count + 1)
            qubit = owner.machine.qubits[qubit_name]
            with if_(cond):
                align(*gates, qubit.xy.name, owner.sensor_dots[0].readout_resonator.id)
                qubit.apply(operation)

        if return_n_loops:
            return n_count
        return None
