import os

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.octave import *
from qm.qua import *
from configuration import *

qop_ip = "xxx.xxx.xxx.xxx"
opx_port = 80
octave_port = 80

octave_config = QmOctaveConfig()
octave_config.add_device_info("octave1", qop_ip, octave_port)

octave_config.set_opx_octave_mapping([("con1", "octave1")])

# If you want to use custom mapping you should use this function instead of the one above
# octave_config.add_opx_octave_port_mapping(portmap)

octave_config.set_calibration_db(os.getcwd())

qmm = QuantumMachinesManager(host=qop_ip, port=opx_port, octave=octave_config)

with program() as prog:
    with infinite_loop_():
        play("readout", "qe1")

qm = qmm.open_qm(config)

# When using internal clock
qm.octave.set_clock("octave1", ClockType.Internal, ClockFrequency.MHZ_10)
# When using external clock. the frequency can be 10, 100 or 1000 MHz
# qm.octave.set_clock("octave1",ClockType.External,ClockFrequency.MHZ_1000)

element = "q1"

qm.octave.set_lo_frequency(element, lo_freq)
qm.octave.set_rf_output_gain(element, -10)
qm.octave.set_rf_output_mode(element, RFOutputMode.on)

qm.octave.set_lo_source(element, OctaveLOSource.LO1)

qm.octave.calibrate_element(element, [(lo_freq, if_freq)])
qm = qmm.open_qm(config)  # Calibration closes the QM so another one should be opened again after calibration is done

qm.octave.set_qua_element_octave_rf_in_port(element, "octave1", 1)
qm.octave.set_downconversion(element)

# This is for the case you want to use an external LO source
# qm.octave.set_downconversion(element, lo_source=RFInputLOSource.Dmd1LO)

qm.execute(prog)
