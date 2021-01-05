"""
widefield_odmr.py: NV center ODMR measurement with camera triggering
Author: Gal Winer - Quantum Machines
Created: 13/12/2020
Created on QUA version: 0.6.393
"""

from random import randint
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
from configuration import config
from camera_mock_lib import *
import time

QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config)

N = 5
f_start = int(1e6)
f_end = int(10e6)
f_step = int((f_end - f_start) / N)
T1 = 10 * 1e2  # in nS
flash_time = 20 * 1e2

cam = cam_mock() #create the camera object
cam.allocate_buffer(1)  # allocate new space in camera
cam.arm() #trigger arm for the first time

result_array = []
with program() as ODMR:
    freq = declare(int)
    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency('qubit', freq)
        play('readout', 'readout_el', duration=24)  # init
        align('qubit', 'readout_el')
        play('SAT', 'qubit') #MW : TODO put timing in config, not dynamic, wider gaussian
        align('qubit', 'readout_el')
        play('readout', 'readout_el', duration=flash_time) #readout. take first 300 ns as signal and rest as ref.
        pause()

job = QM1.execute(ODMR)
# job = QM1.simulate(ODMR, SimulationConfig(10000))

for ind in range(0, N):
    while not job.is_paused():
        time.sleep(0.001)

    result_array.append(cam.get_image())  # collect previous image
    cam.allocate_buffer(1)  # allocate new space in camera
    cam.arm() #trigger arm
    job.resume()
#
# res=job.result_handles
# samples = job.get_simulated_samples()
# samples.con1.plot()
