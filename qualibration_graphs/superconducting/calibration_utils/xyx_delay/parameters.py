from typing import List

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    zeros_before_after_pulse: int = 60
    """Number of zeros before and after the flux pulse to see the rising time. Default is 60ns"""
    z_pulse_amplitude: float = 0.1
    """Amplitude of the Z pulse to detune the qubit in frequency. Default is 0.1V"""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def baked_flux_xy_segments(config: dict, waveform: List[float], qb, zeros_each_side: int):
    """Create baked XY+Z (flux) pulse segments for all relative shifts.

    Parameters
    ----------
    config : dict
        Full QUA configuration dict.
    waveform : list[float]
        Flux (Z) pulse samples (without padding) matching x180 length.
    qb : AnyTransmon-like
        Qubit object providing access to xy and z channels.
    zeros_each_side : int
        Number of zeros before and after (total scan range = 2 * zeros_each_side).

    Returns
    -------
    list
        List of baking objects, each representing one relative timing segment.
    """
    pulse_segments = []
    total = 2 * zeros_each_side
    i_key = f"{qb.xy.operations['x180'].name}.wf.I"
    q_key = f"{qb.xy.operations['x180'].name}.wf.Q"
    I_samples = config["waveforms"][i_key]["samples"]
    Q_samples = config["waveforms"][q_key]["samples"]
    for i in range(total):
        with baking(config, padding_method="symmetric_l") as b:
            wf = [0.0] * i + waveform + [0.0] * (total - i)
            zeros = [0.0] * zeros_each_side
            I_wf = zeros + I_samples + zeros
            Q_wf = zeros + Q_samples + zeros
            assert len(wf) == len(I_wf) == len(Q_wf), "Flux and XY padded waveforms must have identical length"
            b.add_op("flux_pulse", qb.z.name, wf)
            b.add_op("x180", qb.xy.name, [I_wf, Q_wf])
            b.play("flux_pulse", qb.z.name)
            b.play("x180", qb.xy.name)
        pulse_segments.append(b)
    return pulse_segments
