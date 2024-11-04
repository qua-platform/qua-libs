from typing import Dict
from quam.components.channels import IQChannel, MWChannel

from quam_libs.quam_builder.transmons.channel_ports import iq_out_channel_ports, mw_out_channel_ports
from quam_libs.quam_builder.transmons.get_digital_outputs import get_digital_outputs
from qualang_tools.addons.calibration.calibrations import unit
from quam_libs.components import Transmon

u = unit(coerce_to_integer=True)

def add_transmon_drive_component(transmon: Transmon, wiring_path: str, ports: Dict[str, str]):
    digital_outputs = get_digital_outputs(wiring_path, ports)

    if all(key in ports for key in iq_out_channel_ports):
        # LF-FEM & Octave or OPX+ & Octave
        transmon.xy = IQChannel(
            opx_output_I=f"{wiring_path}/opx_output_I",
            opx_output_Q=f"{wiring_path}/opx_output_Q",
            frequency_converter_up=f"{wiring_path}/frequency_converter_up",
            intermediate_frequency=-200 * u.MHz,
            digital_outputs=digital_outputs
        )

        RF_output = transmon.xy.frequency_converter_up
        RF_output.channel = transmon.xy.get_reference()
        RF_output.output_mode = "always_on"  # "triggered"
        RF_output.LO_frequency = 4.7 * u.GHz

        # LF-FEM & Octave or OPX+ & Octave
        transmon.xy_detuned = IQChannel(
            opx_output_I=f"{wiring_path}/opx_output_I",
            opx_output_Q=f"{wiring_path}/opx_output_Q",
            frequency_converter_up=f"{wiring_path}/frequency_converter_up",
            intermediate_frequency=-250 * u.MHz,
            digital_outputs=digital_outputs
        )

        RF_output = transmon.xy_detuned.frequency_converter_up
        RF_output.channel = transmon.xy.get_reference()
        RF_output.output_mode = "always_on"  # "triggered"
        RF_output.LO_frequency = 4.7 * u.GHz

    elif all(key in ports for key in mw_out_channel_ports):
        # MW-FEM single channel
        transmon.xy = MWChannel(
            opx_output=f"{wiring_path}/opx_output",
            digital_outputs=digital_outputs,
            intermediate_frequency=-200 * u.MHz
        )
        # MW-FEM single channel
        transmon.xy_detuned = MWChannel(
            opx_output=f"{wiring_path}/opx_output",
            digital_outputs=digital_outputs,
            intermediate_frequency=-250 * u.MHz
        )

    else:
        raise ValueError(f"Unimplemented mapping of port keys to channel for ports: {ports}")
