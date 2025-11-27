from qualibrate import QualibrationNode


def external_source_setup(node: QualibrationNode, reset_voltages: bool = True):
    """
    Fill this function with the connection to your external voltage source, and provide a mapping of channel : external voltage. 

    An example has been provided for use with a QDAC.
    """
    qdac_ip = "172.16.33.101"
    name="QDAC"
    from qcodes import Instrument
    from qcodes_contrib_drivers.drivers.QDevil import QDAC2
    
    try:
        qdac = Instrument.find_instrument(name)
    except KeyError:
        qdac = QDAC2.QDac2(name, visalib='@py', address=f'TCPIP::{qdac_ip}::5025::SOCKET')
    
    external_voltage_mapping = {
        node.machine.quantum_dots["virtual_dot_1"].physical_channel: qdac.ch01.dc_constant_V, 
        node.machine.quantum_dots["virtual_dot_2"].physical_channel: qdac.ch02.dc_constant_V, 
        node.machine.quantum_dots["virtual_dot_3"].physical_channel: qdac.ch03.dc_constant_V, 
        node.machine.quantum_dots["virtual_dot_4"].physical_channel: qdac.ch04.dc_constant_V, 
        node.machine.barrier_gates["virtual_barrier_1"].physical_channel: qdac.ch05.dc_constant_V, 
        node.machine.barrier_gates["virtual_barrier_2"].physical_channel: qdac.ch06.dc_constant_V,
        node.machine.barrier_gates["virtual_barrier_3"].physical_channel: qdac.ch07.dc_constant_V, 
        node.machine.sensor_dots["virtual_sensor_1"].physical_channel: qdac.ch08.dc_constant_V
    }
    node.machine.connect_to_external_source(external_voltage_mapping, reset_voltages = reset_voltages)

