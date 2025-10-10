from typing import ClassVar, List, Literal, Optional

import numpy as np
from qualang_tools.bakery import baking
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a qubit spectroscopy experiment.

    Attributes:
        num_shots (int): Number of averages to perform. Default is 100.
        frequency_span_in_mhz (float): Span of frequencies to sweep in MHz. Default is 100 MHz.
        frequency_step_in_mhz (float): Step size for frequency sweep in MHz. Default is 0.25 MHz.
        operation (str): Type of operation to perform. Default is "saturation".
        operation_amplitude_factor (Optional[float]): Amplitude pre-factor for the operation. Default is 1.0.
        operation_len_in_ns (Optional[int]): Length of the operation in nanoseconds. Default is None.
        target_peak_width (Optional[float]): Target peak width in Hz. Default is 3e6 Hz.
    """

    num_shots: int = 100
    max_time_in_ns: int = 160
    amp_range: float = 0.1
    amp_step: float = 0.003
    use_state_discrimination: bool = True


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"


def baked_waveform(qubit, baked_config, base_level: float = 0.5, max_samples: int = 16):
    """Create truncated baked waveforms for the chevron CZ calibration.

    Generates a list of baking objects, each containing an incrementally longer flux pulse
    (1..max_samples samples) at the specified base_level. Each baked pulse is registered
    as an operation named "flux_pulse{i}" on the provided qubit z line.

    Args:
        qubit: The qubit object whose z line is used.
        baked_config: A mutable QM configuration object to which baked operations are added.
        base_level (float): The constant waveform level used for the short baked segments.
        max_samples (int): The maximum number of samples (and thus baked variants) to generate.

    Returns:
        List of baking objects, index i corresponds to pulse length i+1 samples.
    """
    pulse_segments = []
    waveform = [base_level] * max_samples
    for i in range(1, max_samples + 1):
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        pulse_segments.append(b)
    return pulse_segments
    return pulse_segments
    return pulse_segments
