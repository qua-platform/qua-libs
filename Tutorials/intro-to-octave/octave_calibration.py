"""
This file is used to configure the Octave's clock and do the automatic calibration.
"""

import os

from qm import QuantumMachinesManager
from configuration import *


# Configure the Octave according to the elements settings and calibrate
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, octave_calibration_db_path=os.getcwd(), log_level="ERROR")
qm = qmm.open_qm(config)


##################
# Calibration #
##################
calibration = True

if calibration:
    elements = ["qe1", "qe2", "qe3", "qe4", "qe5"]
    for element in elements:
        print("-" * 37 + f" Calibrates {element}")
        qm.calibrate_element(element, {LO: (IF,)})  # can provide many IFs for specific LO
