"""
This file is used to configure the Octave's clock and do the automatic calibration.
"""

from qm import QuantumMachinesManager
from qm.octave import ClockMode
from configuration import *


# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config, log_level="ERROR")
qm = qmm.open_qm(config)

##################
# Clock settings #
##################
qm.octave.set_clock("octave1", clock_mode=ClockMode.Internal)
# If using external LO change this line to one of the following:
#     qm.octave.set_clock("octave1", clock_mode=ClockMode.External_10MHz)
#     qm.octave.set_clock("octave1", clock_mode=ClockMode.External_100MHz)
#     qm.octave.set_clock("octave1", clock_mode=ClockMode.External_1000MHz)

##################
# Calibration #
##################
calibration = True

if calibration:
    elements = ["qubit"]
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element)  # can provide many IFs for specific LO
