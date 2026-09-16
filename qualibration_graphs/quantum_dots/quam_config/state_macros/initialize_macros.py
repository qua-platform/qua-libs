"""
A library of initialize macros. 

For each macro, create an Attributes class and mix it with the CustomMacro class from quam-builder.
"""

from typing import Optional, Literal

from quam_builder.architecture.quantum_dots import CustomMacro
from quam.core import quam_dataclass

from qm.qua import align, strict_timing_, assign, declare, if_, while_, Cast

__all__ = [
    "InitializeMacro",
    "BalancedRoundTripInitializeMacro",
    "HeraldedInitializeMacro",
]


##############################
##### Example Initialize #####
##############################

@quam_dataclass
class InitializeMacroAttributes: 
    point_duration: int = 1000
    """Hold duration of the Initialize voltage point."""

@quam_dataclass
class InitializeMacro(CustomMacro, InitializeMacroAttributes):
    Parameters = InitializeMacroAttributes

    @property
    def inferred_duration(self) -> float | None:
        return 0

    def apply(
        self, 
        **kwargs,
    ):
        owner = self.owner
        params = self.resolve_params(**kwargs)
        point_duration = params["point_duration"]
        pass

##########################################
##### Balanced Round Trip Initialize #####
##########################################

@quam_dataclass
class BalancedRoundTripInitializeMacroAttributes: 
    zero_duration: int = 100
    """The amount of time to idle at zero voltage after the round trip."""
    ramp_duration: int = 500
    """The ramp duration to the initalize voltage coordinate."""
    hold_duration: int = 500
    """The hold duration at the initialize voltage coordinate."""
    point_name: str = "initialize"
    """The voltage point name."""

@quam_dataclass
class BalancedRoundTripInitializeMacro(CustomMacro, BalancedRoundTripInitializeMacroAttributes):
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
    Parameters = BalancedRoundTripInitializeMacroAttributes

    @property
    def inferred_duration(self) -> float | None:
        return (4 * self.ramp_duration + 2 * self.hold_duration + self.zero_duration) * 1e-9

    def apply(
        self,
        **kwargs,
    ):
        owner = self.owner  # The QuantumDotPair object that is the ultimate owner of this macro
        params = self.resolve_params(**kwargs)
        # Check if any arguments have been passed to the macro, and make sure to fall back to the
        # class attribute as a default.
        ramp = params["ramp_duration"]
        hold = params["hold_duration"]
        zero = params["zero_duration"]
        point_name = params["point_name"]

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
            

###############################
##### Heralded Initialize #####
###############################

@quam_dataclass
class HeraldedInitializeAttributes: 
    max_loops: int = 2
    """The maximum number of active reset loops to perform before exiting the loop and continuing with the program."""
    return_n_loops: bool = False
    """A bool option to extract the number of loops the active reset initialize has performed."""
    target_state: Literal[0, 1] = 0
    """The qubit state to try to initialize into."""
    qubit_role: Literal["target", "control"] = "control"
    """For a qubit pair, whether to pulse on the control or the target qubit. """
    operation: str = "x180"
    """The operation to play on the qubit."""
    qubit_name: Optional[str] = None
    """The name of the qubit to drive."""
    meas_ramp_duration: Optional[int] = None
    """The ramp duration to the measure point."""
    meas_buffer_duration: Optional[int] = None
    """The buffer duration in the measure macro."""

@quam_dataclass
class HeraldedInitializeMacro(BalancedRoundTripInitializeMacro, HeraldedInitializeAttributes):
    """
    An active reset initialize scheme, built on the BalancedInitializeMacro.

    The flow:
    - Initialize using the BalancedInitializeMacro
    - Measure the state
    - If the state is NOT the desired state, drive the specified qubit, and repeat the above
    - If the state is the desired state, exit the loop

    This class also optionally allows one to extract the number of loops performed as a stream
    """
    Parameters = HeraldedInitializeAttributes

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
        **kwargs,
    ):
        owner = self.owner
        params = self.resolve_params(**kwargs)
        max_loops = params["max_loops"]
        return_n_loops = params["return_n_loops"]
        target_state = params["target_state"]
        qubit_role = params["qubit_role"]
        operation = params["operation"]
        qubit_name = params["qubit_name"]
        meas_ramp_duration = params["meas_ramp_duration"]
        meas_buffer_duration = params["meas_buffer_duration"]

        if qubit_name is None:
            # Extract the qubit pair whose quantum_dot_pair is the owner
            qubit_pair = next(qp for qp in owner.machine.qubit_pairs.values() if qp.quantum_dot_pair is owner)
            qubit_name = getattr(qubit_pair, f"qubit_{qubit_role}", None)
            if qubit_name is None:
                raise ValueError("Failed to resolve qubit")

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

