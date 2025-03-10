from typing import Dict
from quam_builder.builder.qop_connectivity.channel_ports import (
    iq_in_out_channel_ports,
    mw_in_out_channel_ports,
)
from quam_builder.builder.qop_connectivity.get_digital_outputs import (
    get_digital_outputs,
)
from qualang_tools.addons.calibration.calibrations import unit
from quam_builder.architecture.superconducting.components.readout_resonator import (
    ReadoutResonatorIQ,
    ReadoutResonatorMW,
)
from quam_builder.architecture.superconducting.qubit import AnyTransmon

u = unit(coerce_to_integer=True)


def add_transmon_resonator_component(transmon: AnyTransmon, wiring_path: str, ports: Dict[str, str]):
    """
    Adds a resonator component to a transmon qubit based on the provided wiring path and ports.

    Parameters:
    transmon (AnyTransmon): The transmon qubit to which the resonator component will be added.
    wiring_path (str): The path to the wiring configuration.
    ports (Dict[str, str]): A dictionary mapping port names to their respective configurations.

    Raises:
    ValueError: If the port keys do not match any implemented mapping.
    """
    digital_outputs = get_digital_outputs(wiring_path, ports)

    intermediate_frequency = -250 * u.MHz
    depletion_time = 1 * u.us
    time_of_flight = 28  # 4ns above default so that it appears in state.json

    if all(key in ports for key in iq_in_out_channel_ports):
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
            time_of_flight=time_of_flight,
        )

        RF_output_resonator = transmon.resonator.frequency_converter_up
        RF_output_resonator.channel = transmon.resonator.get_reference()
        RF_output_resonator.output_mode = "always_on"
        RF_output_resonator.LO_frequency = 6.2 * u.GHz

        RF_input_resonator = transmon.resonator.frequency_converter_down
        RF_input_resonator.channel = transmon.resonator.get_reference()
        RF_input_resonator.LO_frequency = 6.2 * u.GHz

    elif all(key in ports for key in mw_in_out_channel_ports):
        transmon.resonator = ReadoutResonatorMW(
            opx_output=f"{wiring_path}/opx_output",
            opx_input=f"{wiring_path}/opx_input",
            digital_outputs=digital_outputs,
            depletion_time=depletion_time,
            intermediate_frequency=intermediate_frequency,
            time_of_flight=time_of_flight,
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")
