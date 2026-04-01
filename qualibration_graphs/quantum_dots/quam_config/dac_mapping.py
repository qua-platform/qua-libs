"""
This script should do the following:

1. Connect to your external DAC.
2. Set up your channel mapping.
"""

from typing import Dict


def channel_mapping(
    machine=None,
    quam_path: str = None,
) -> Dict:

    # Load the Quam first.
    if machine is None:
        try:
            from quam_config import Quam

            if quam_path is None:
                machine = Quam.load()
            else:
                machine = Quam.load(quam_path)
        except:
            from quam_config import QubitQuam as Quam

            if quam_path is None:
                machine = Quam.load()
            else:
                machine = Quam.load(quam_path)

    # This example uses QCodes QDAC2. Replace with your own DAC driver.
    from qcodes_contrib_drivers.drivers.QDevil.QDAC2 import QDac2

    dac = QDac2("DAC", visalib="@py", address=f"TCPIP::172.16.33.101::5025::SOCKET")

    # Use your own mapping here
    channel_mapping = {
        machine.quantum_dots["virtual_dot_1"].physical_channel: dac.ch01.dc_constant_V,
        machine.quantum_dots["virtual_dot_2"].physical_channel: dac.ch02.dc_constant_V,
        machine.quantum_dots["virtual_dot_3"].physical_channel: dac.ch07.dc_constant_V,
        machine.quantum_dots["virtual_dot_4"].physical_channel: dac.ch08.dc_constant_V,
        machine.sensor_dots["virtual_sensor_1"].physical_channel: dac.ch03.dc_constant_V,
        machine.barrier_gates["virtual_barrier_1"].physical_channel: dac.ch05.dc_constant_V,
        machine.barrier_gates["virtual_barrier_2"].physical_channel: dac.ch06.dc_constant_V,
        machine.barrier_gates["virtual_barrier_3"].physical_channel: dac.ch09.dc_constant_V,
    }

    return channel_mapping
