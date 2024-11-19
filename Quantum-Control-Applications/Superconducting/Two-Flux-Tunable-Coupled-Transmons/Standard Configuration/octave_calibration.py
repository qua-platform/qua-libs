"""
This file is used to perform the Octave automatic calibration.
"""

from qm import QuantumMachinesManager
from configuration import *


# Calibrate the Octave's mixers
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave=octave_config, log_level="ERROR")
qm = qmm.open_qm(config)



##################
# Calibration #
##################
calibration = True

if calibration:
    elements = ["rr1", "rr2", "q1_xy", "q2_xy"]
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element)  # can provide many IFs for specific LO
