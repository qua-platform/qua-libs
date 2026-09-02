"""
In this script, you can set up your own custom Quam macros. QuamMacros form 
an essential part of the spin qubits Quam infrastructure. Quam-builder contains a CustomMacro
base class, which you can subclass to write your own QUA snippet in the apply() function. 

When you create a macro, make sure to emit it as one of the following: 
- InitializeMacro
- MeasureMacro

This script builds 2 macros. First, it builds a BalancedInitializeMacro, which performs
a voltage integral balanced round trip about a DC point. This is particularly useful for
performing balanced initialization sequences, where the high-pass filter compensation
happens before the actual desired initialize pulse, keeping the innermost shot clean. 

As a second example, this script writes a BalancedHeraldedInitializeMacro. This builds on 
the BalancedInitializeMacro, but implements an active-reset type initialization sequence. 
"""
from typing import Optional, Literal

from quam_builder.architecture.quantum_dots.operations import CustomMacro
from quam.core import quam_dataclass

from qm.qua import align, strict_timing_, assign, declare, if_, while_, Cast

__all__ = ["InitializeMacro", "MeasureMacro"]

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



#-------- EXAMPLES ---------

# First we create a BalancedInitializeMacro, which describes a balanced round trip. 
# The real macro that we export from here should be names InitializeMacro, which is listed below. 

@quam_dataclass
class BalancedInitializeMacro(CustomMacro): 
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
        return (2 * self.ramp_duration + 2 * self.hold_duration + 16) * 1e-9

    def apply(
        self,
        ramp_duration: int | None = None,
        hold_duration: int | None = None,
        zero_duration: int | None = None,
        point_name: str | None = None,
        **kwargs,
    ):
        owner = self.owner # The QuantumDotPair object that is the ultimate owner of this macro

        # Check if any arguments have been passed to the macro, and make sure to fall back to the 
        # class attribute as a default. 
        ramp = self.ramp_duration if ramp_duration is None else ramp_duration
        hold = self.hold_duration if hold_duration is None else hold_duration
        zero = self.zero_duration if zero_duration is None else zero_duration
        point_name = self.point_name if point_name is None else point_name

        # Create dicts of positive and negative voltage points
        positive_voltages = self.point_voltages(point_name)
        negative_voltages = {k: -v for k, v in positive_voltages.items()}
        zero = {k: 0.0 for k, _ in positive_voltages.items()}

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
                ramp_duration=2*ramp,
                ensure_align=False,
            )
            vs.ramp_to_voltages(
                zero,
                duration=zero,
                ramp_duration=ramp,
                ensure_align=False,
            )


@quam_dataclass
class BalancedHeraldedInitializeMacro(BalancedInitializeMacro): 
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

    def apply(
        self,
        max_loops: int = 2,
        target_state: Optional[Literal[0, 1]] = None,
        return_n_loops: bool | None = False,
        operation: str = "x180",
        qubit_role: Optional[Literal["target", "control"]] = None,
        qubit_name: Optional[str] = None,
        meas_ramp_duration: Optional[int] = None,
        meas_buffer_duration: Optional[int] = None,
        **kwargs
    ):
        owner = self.owner

        if qubit_name is None:
            qubit_role = self.qubit_role if qubit_role is None else qubit_role
            # Extract the qubit pair whose quantum_dot_pair is the owner
            qubit_pair = next(qp for qp in owner.machine.qubit_pairs.values() if qp.quantum_dot_pair is owner)
            qubit_name = getattr(qubit_pair, f"qubit_{qubit_role}", None)
            if qubit_name is None: 
                raise ValueError("Failed to resolve qubit")
        
        if target_state is None:
            target_state = 0
    
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

            # As long as the state is in the initial value, the loop will continue until max_loops
            assign(cond, Cast.to_bool(state - target_state))
            assign(n_count, n_count + 1)
            qubit = owner.machine.qubits[qubit_name]
            with if_(cond):
                align(*gates, qubit.xy.name, owner.sensor_dots[0].readout_resonator.id)
                qubit.apply(operation)

        if return_n_loops: 
            return n_count
        return None
