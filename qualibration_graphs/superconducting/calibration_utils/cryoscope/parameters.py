from typing import List, Literal, Optional

from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 5000
    """Number of averages to perform. Default is 50."""
    detuning_target_in_MHz: int = 400
    """Target detuning from sweetspot for the cryoscope pulse in MHz. Default is 350."""
    cryoscope_len: int = 240
    """Length of the cryoscope operation in microseconds. Default is 240."""
    num_frames: int = 17
    """Number of frames to use in the cryoscope experiment. Default is 17."""
    number_of_exponents: Literal[1, 2] = 1
    """Number of exponents to use in the cryoscope experiment. One or two, default is 1."""
    exp_1_tau_guess: Optional[float] = None
    """Initial guess for the time constant of the first exponential decay. Default is None."""
    exponential_fit_time_fractions: List[float] = [0.5, 0.01]
    """List of time fractions for the exponential fit. Default is [0.5, 0.01]."""
    update_state_from_GUI: bool = False
    """Whether to update the state from the GUI. Default is False."""
    update_state: bool = False
    """Whether to update the state. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass


def baked_waveform(waveform_amp, qubit, baked_config):
    """Generate baked pulse segments with 1ns granularity up to 16ns.

    Parameters
    ----------
    waveform_amp : float
        Amplitude for each 1ns sample of the waveform
    qubit : object
        Qubit object containing at least 'z.name'
    baked_config : dict
        Configuration object used by the baking context

    Returns
    -------
    list
        List of baking objects, each containing a pulse from 1ns to 16ns
    """
    pulse_segments = []  # Stores the baking objects
    waveform = [waveform_amp] * 16
    for i in range(1, 17):  # from first item up to pulse_duration (16)
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments
