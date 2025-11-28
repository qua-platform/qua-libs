"""
This file is used to run the Octave automatic mixer calibration.
"""

from qm import QuantumMachinesManager
from configuration import *

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name, log_level="ERROR")
qm = qmm.open_qm(config)

elements = [
    "rr1",
    "rr2",
    "q1_xy",
    "q2_xy",
    "cr_drive_c1t2",
    "cr_drive_c2t1",
    "cr_cancel_c1t2",
    "cr_cancel_c2t1",
]
for element in elements:
    print("-" * 37 + f" Calibrates {element}")
    qm.calibrate_element(element)  # can provide many IFs for specific LO
