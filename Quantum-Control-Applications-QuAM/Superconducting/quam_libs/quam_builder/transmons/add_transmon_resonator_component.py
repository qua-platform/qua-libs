from typing import Dict

from quam_libs.quam_builder.transmons.channel_ports import iq_in_out_channel_ports, mw_in_out_channel_ports
from quam_libs.quam_builder.transmons.get_digital_outputs import get_digital_outputs
from qualang_tools.addons.calibration.calibrations import unit
from quam_libs.components import Transmon, ReadoutResonatorIQ, ReadoutResonatorMW, QuAM

u = unit(coerce_to_integer=True)

def add_transmon_resonator_component(transmon: Transmon, wiring_path: str, ports: Dict[str, str], machine: QuAM):
    digital_outputs = get_digital_outputs(wiring_path, ports)

    intermediate_frequency = -250 * u.MHz
    depletion_time = 1 * u.us
    time_of_flight = 28  # 4ns above default so that it appears in state.json

    if all(key in iq_in_out_channel_ports for key in ports):
        transmon.resonator = ReadoutResonatorIQ(
            opx_output_I=f"{wiring_path}/opx_output_I",
            opx_output_Q=f"{wiring_path}/opx_output_Q",
            opx_input_I=f"{wiring_path}/opx_input_I",
            opx_input_Q=f"{wiring_path}/opx_input_Q",
            opx_input_offset_I=0.0,
            opx_input_offset_Q=0.0,
            frequency_converter_up=f"{wiring_path}/frequency_converter_up",
            frequency_converter_down=f"{wiring_path}/frequency_converter_down",
            digital_outputs=digital_outputs,
            intermediate_frequency=intermediate_frequency,
            depletion_time=depletion_time,
            time_of_flight=time_of_flight
        )

        RF_output_resonator = transmon.resonator.frequency_converter_up
        RF_output_resonator.channel = transmon.resonator.get_reference()
        RF_output_resonator.output_mode = "always_on"
        RF_output_resonator.LO_frequency = 6.2 * u.GHz

        RF_input_resonator = transmon.resonator.frequency_converter_down
        RF_input_resonator.channel = transmon.resonator.get_reference()
        RF_input_resonator.LO_frequency = 6.2 * u.GHz

    elif all(key in mw_in_out_channel_ports for key in ports):
        transmon.resonator = ReadoutResonatorMW(
            opx_output=f"{wiring_path}/opx_output",
            opx_input=f"{wiring_path}/opx_input",
            digital_outputs=digital_outputs,
            depletion_time=depletion_time,
            intermediate_frequency=intermediate_frequency,
            time_of_flight=time_of_flight
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")
