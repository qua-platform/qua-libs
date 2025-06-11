from typing import List, Union

from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorIQ,
    ReadoutResonatorMW,
)
from qualibration_libs.core import tracked_updates
from calibration_utils.time_of_flight.parameters import Parameters


def patch_readout_pulse_params(
    resonators: List[Union[ReadoutResonatorIQ, ReadoutResonatorMW]],
    node_parameters: Parameters,
):
    patched_resonators = []
    for resonator in resonators:
        # make temporary updates before running the program and revert at the end.
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            resonator.time_of_flight = node_parameters.time_of_flight_in_ns
            resonator.operations["readout"].length = node_parameters.readout_length_in_ns
            resonator.operations["readout"].amplitude = node_parameters.readout_amplitude_in_v
            if node_parameters.intermediate_frequency_in_mhz is not None:
                resonator.intermediate_frequency = int(node_parameters.intermediate_frequency_in_mhz * 1e6)
            patched_resonators.append(resonator)

    return patched_resonators
