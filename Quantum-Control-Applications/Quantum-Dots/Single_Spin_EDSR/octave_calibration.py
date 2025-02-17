"""
This file is used to perform the Octave automatic calibration.
"""

from qm import QuantumMachinesManager
from configuration import *


# Calibrate the Octave's mixers
qmm = QuantumMachinesManager(**qmm_settings)
qm = qmm.open_qm(config)


##################
# Calibration #
##################
calibration = True

if calibration:
    elements = ["qubit"]
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element)  # can provide many IFs for specific LO
