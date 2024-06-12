"""
This file is used to configure the Octave ports (gain, switch_mode, down-conversion) and calibrate the up-conversion mixers.
You need to run this file in order to update the Octaves with the new parameters.
"""

from qm import QuantumMachinesManager
from qm.octave import ClockMode

from configuration import *
from quam import QuAM


config = build_config(machine)


# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(
    host=machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config, log_level="ERROR"
)
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
    elements = ["qe1", "qe2", "qe3", "qe4", "qe5"]
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element)
