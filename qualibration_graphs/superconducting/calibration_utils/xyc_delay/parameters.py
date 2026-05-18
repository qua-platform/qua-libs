"""Parameter definitions for XY-Coupler delay calibration experiment."""

from typing import ClassVar, List, Literal

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """XY-Z delay specific parameters for timing scan configuration."""

    num_shots: int = 50
    """Number of averages to perform. Default is 50."""
    zeros_before_after_pulse: int = 60
    """Number of zeros before and after the flux pulse to see the rising time. Default is 60ns"""
    coupler_pulse_amplitude: float = 0.1
    """Amplitude of the Z pulse to detune the qubit in frequency. Default is 0.1V"""
    reset_coupler_bias: bool = False
    """Whether to reset the coupler bias to 0V before each measurement (True)."""
    measure_qubit: Literal["control", "target"] = "target"
    """Which qubit to measure: 'control' or 'target'. Default is 'target'."""
    flux_point: Literal["joint", "independent"] = "independent"
    """Flux point setting strategy: 'joint' or 'independent'. Default is 'independent'."""
    reset_type: Literal["active", "thermal"] = "thermal"
    """Type of qubit reset to use before each measurement: 'active' or 'thermal'. Default is 'thermal'."""
    use_state_discrimination: bool = True
    """Whether to use state discrimination. Default is True since the analysis function is written for state discrimination."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for XY-Coupler Z delay calibration node."""

    targets_name: ClassVar[str] = "qubit_pairs"


# pylint: disable=too-many-locals
def baked_cplr_flux_xy_segments(config: dict, waveform: List[float], qb, coupler, zeros_each_side: int):
    """Create baked XY+Coupler Z (flux) pulse segments for all relative shifts.

    Parameters
    ----------
    config : dict
        Full QUA configuration dict.
    waveform : list[float]
        Flux (Z) pulse samples (without padding) matching x180 length.
    qb : AnyTransmon-like
        Qubit object providing access to xy channel.
    coupler : AnyFluxTunableLike
        Coupler object providing access to flux channel.
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
            b.add_op("flux_pulse", coupler.name, wf)
            b.add_op("x180", qb.xy.name, [I_wf, Q_wf])
            b.play("flux_pulse", coupler.name)
            b.play("x180", qb.xy.name)
        pulse_segments.append(b)
    return pulse_segments
