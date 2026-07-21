from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitsExperimentNodeParameters

from typing import Literal, Optional

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (frequency detuning, amplitude scale) point."""
    detuning: Optional[float]  = None
    """Detuning value to step to as a part of the initialization pulse."""
    barrier_gate_voltage: Optional[float] = None
    """Barrier gate value to step to as a part of the initialization pulse."""
    ramp_duration: int = 1000
    """Ramp duration for init sequence"""
    buffer_duration: int = 524
    """Wait time before measure"""
    hold_duration: int = 1000
    """Wait time after the pulse"""
    drive_frequency_detuning_span_MHz: float = 5
    """The frequency detuning span relative to the drive frequency, to play at the readout point."""
    drive_frequency_detuning_step_MHz: float = 0.05
    """The frequency detuning step relative to the drive frequency, to play at the readout point."""
    drive_amplitude_scale_span: float = 1
    """The frequency scale span at which to play the drive pulse at the readout point."""
    drive_amplitude_scale_step: float = 0.05
    """The frequency scale step at which to play the drive pulse at the readout point."""
    pulse_duration: int = 1500
    """The duration of the pulse to play at the readout point."""
    operation: Literal["x180", "x90"] = "x180"
    """The pulse to play for state preparation before the first measurement."""
    reset_operation: Literal["x180", "x90"] = "x180"
    """The pulse to play as reset drive after the first measurement."""
    plot_pca_maps: bool = True
    """If True, plot PCA maps (PC1 from I/Q) for pre and post measurements."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Parameter set for 07a_init_2d_calibration."""


from qm.qua import *
from qualibrate import QualibrationNode

from typing import Dict, Any
from quam_builder.architecture.quantum_dots.operations.names import (
    TwoQubitMacroName,
    VoltagePointName,
)
from quam_builder.architecture.quantum_dots.defaults import DEFAULTS

def balanced_initialise_pulse(
    qubit,
    xy_channel,
    node: QualibrationNode, 
    ramp_duration: int = 1000,
    buffer_duration: int = 524,
    hold_duration: int = 1000, 
    point: str | Dict = None, 
    pulse_name: str = "x180",
    amplitude_scale: float = 1.0,
    frequency_detuning_Hz: int = 0,
): 

    dot_pair = node.machine.quantum_dot_pairs[
        node.machine.find_quantum_dot_pair(
            qubit.quantum_dot.name,
            qubit.preferred_readout_quantum_dot,
        )
    ]

    if not dot_pair.sensor_dots: 
        raise ValueError(f"QuantumDotPair '{dot_pair.name}' has no sensor dots for readout")
    
    sensor_dot = dot_pair.sensor_dots[0]
    sensor_macro = sensor_dot.macros[TwoQubitMacroName.MEASURE]
    if hasattr(sensor_macro, "readout_pulse_length_ns_for_pair"):
        readout_len = sensor_macro.readout_pulse_length_ns_for_pair(dot_pair.id)
    else:
        readout_len = sensor_macro.readout_pulse_length_ns
    if readout_len is None:
        raise ValueError(
            "Sensor readout pulse length unknown; balanced measurement "
            "requires a fixed readout duration."
        )

    hold = buffer_duration + readout_len
    wait_cycles = int((ramp_duration + buffer_duration) // 4)

    def _point_voltages(owner: Any, point: str | dict) -> dict[str, float]:
        if isinstance(point, dict):
            return point
        if point is None:
            point = VoltagePointName.MEASURE.value
        full_name = owner._create_point_name(point)
        tuning_point = owner.voltage_sequence.gate_set.macros.get(full_name)
        return dict(tuning_point.voltages)

    positive = _point_voltages(dot_pair, point)
    negative = {k: -v for k, v in positive.items()}
    zero = {k: 0.0 for k, _ in positive.items()}

    vs = dot_pair.voltage_sequence

    gates = [ch_name for ch_name in vs.gate_set.channels.keys()]
    elements_to_align = [sensor_dot.readout_resonator.name, xy_channel.id, *gates]

    align(*elements_to_align)
    op_frequency = xy_channel.intermediate_frequency
    new_freq = op_frequency + frequency_detuning_Hz

    pulse_family = node.machine.pulse_family

    pulse_name_with_family = f"{pulse_family}_{pulse_name}"

    # Update the frequency outside of the strict timing
    xy_channel.update_frequency(new_freq)
    op_length = xy_channel.operations[pulse_name_with_family].length

    with strict_timing_():
        vs.ramp_to_voltages(
            positive,
            duration=hold + op_length + hold_duration,
            ramp_duration=ramp_duration,
            ensure_align=False,
        )

        # On both channels: wait for 2 ramps, buffer, and then hold 
        wait(wait_cycles, xy_channel.id)
        wait(wait_cycles, sensor_dot.readout_resonator.name)

        # Unconditionally play the pulse and then wait for readout
        xy_channel.play(pulse_name_with_family, amplitude_scale = amplitude_scale)
        wait(readout_len//4, xy_channel.id)

        # Sensor: ramp + buffer, then read
        wait(op_length//4, sensor_dot.readout_resonator.name)
        i, q, result = sensor_macro.apply(
            quantum_dot_pair_id=dot_pair.id,
            return_iq=True,
        )
        
        vs.ramp_to_voltages(
            negative,
            duration=hold + op_length + hold_duration,
            ramp_duration=2 * ramp_duration,
            ensure_align=False,
        )
        
        wait(wait_cycles + readout_len//4 + op_length//4 + hold_duration//4 + ramp_duration//4, sensor_dot.readout_resonator.name)
        wait(wait_cycles + readout_len//4 + op_length//4 + hold_duration//4 + ramp_duration//4, xy_channel.id)

        # Both XY and sensor wait for hold duration
        wait(hold_duration//4, xy_channel.id)
        wait(hold_duration//4, sensor_dot.readout_resonator.name)
        
        vs.ramp_to_voltages(
            zero,
            duration=DEFAULTS.state_macro.point_duration,
            ramp_duration=ramp_duration,
            ensure_align=False,
        )

        return i, q, result

