from dataclasses import field
from typing import Optional

from quam.components.pulses import SquarePulse

from quam_libs.components import QuAM

from quam.components import MWChannel
from quam.components.ports import MWFEMAnalogOutputPort

from quam_libs.lib.power_tools import set_output_power_mw_channel

twpa_controller = "con1"
twpa_fem_index = 6
twpa_port_index = 8
twpa_frequency = 4.5e9
twpa_intermediate_frequency = 50e6
twpa_frequency_band = 1  # 1 [0.05, 5.5]; 2 [4.5, 7.5]; 3 [6.5, 10.5] GHz
twpa_power_in_dBm = -12
twpa_pulse_name = "const"
twpa_thread = "twpa"

machine = QuAM.load()

ports_fem = machine.ports.mw_outputs[twpa_controller][twpa_fem_index]

if ports_fem.get(twpa_port_index) is not None:
    raise KeyError(f"Port {twpa_port_index} already exists, which means it's likely being "
                   f"used by a different element. For continuous TWPA pulsing, the port "
                   f"needs to be unused by other elements.")

else:
    ports_fem[twpa_port_index] = MWFEMAnalogOutputPort(
        controller_id=twpa_controller, fem_id=twpa_fem_index, port_id=twpa_port_index,
        upconverter_frequency=twpa_frequency - twpa_intermediate_frequency, band=twpa_frequency_band
    )

machine.wiring.twpa = f"#/ports/mw_outputs/{twpa_controller}/{twpa_fem_index}/{twpa_port_index}"

machine.twpa = MWChannel(
    opx_output="#/wiring/twpa", id="twpa", thread=twpa_thread,
    intermediate_frequency=twpa_intermediate_frequency
)

temporary_amplitude = 1
machine.twpa.operations[twpa_pulse_name] = SquarePulse(amplitude=temporary_amplitude, length=1000)

set_output_power_mw_channel(machine.twpa, power_in_dbm=twpa_power_in_dBm, operation=twpa_pulse_name)

machine.save(
    content_mapping={
        "wiring.json": ["network", "wiring"],
    }
)