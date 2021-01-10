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

N = 101
f_start = int(-100e6)
f_end = int(100e6)
f_step = int((f_end - f_start) / N)

simulate = True
case = 1  # 1 for fast, 2 for normal, 3 for external

# Phantom camera has 1MHz FPS, can trigger for a 10us readout, no need for reference.
with program() as FAST_ODMR:
    freq = declare(int)
    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency('NV', freq)
        play('init', 'laser', duration=10e3//4)  # init
        play('CW', 'NV', duration=10e3//4)  # MW
        play('trigger', 'camera', duration=10e3//4)  # camera
        wait(1000//4, 'NV')


# Readout is being done for 10ms with MW and 10ms without MW. Camera is programmed in the beginning so can run without
# pausing.
with program() as ODMR:
    freq = declare(int)
    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency('NV', freq)
        play('init', 'laser', duration=10e4//4)  # init
        play('CW', 'NV', duration=10e4//4)  # MW
        play('trigger', 'camera', duration=10e3//4)  # camera
        wait(10000//4, 'laser')
        align('laser', 'camera')
        play('init', 'laser', duration=10e4//4)  # init
        play('trigger', 'camera', duration=10e3//4)  # camera
        wait(10000//4, 'laser')


# Readout is being done for 10ms with MW and 10ms without MW. Camera output is extracted after each freq.
with program() as ODMR_EXT:
    freq = declare(int)
    with for_(freq, f_start, freq <= f_end, freq + f_step):
        update_frequency('NV', freq)
        play('init', 'laser', duration=10e6//4)  # init
        play('CW', 'NV', duration=10e6//4)  # MW
        play('trigger', 'camera', duration=10e3//4)  # camera
        wait(10000//4, 'laser')
        align('laser', 'camera')
        play('init', 'laser', duration=10e6//4)  # init
        play('trigger', 'camera', duration=10e3//4)  # camera
        wait(10000//4, 'laser')
        pause()

if simulate:
    job = QM1.simulate(FAST_ODMR, SimulationConfig(60000))
    res = job.result_handles
    samples = job.get_simulated_samples()
    samples.con1.plot()
else:
    if case == 1:
        job = QM1.execute(FAST_ODMR)
        cam = cam_mock()  # create the camera object
        cam.allocate_buffer(N)  # allocate new space in camera
        cam.arm()  # trigger arm for the first time

        result_array = []

        for ind in range(0, N):
            while not job.is_paused():
                time.sleep(0.001)

            result_array.append(cam.get_image())  # collect previous image
            cam.allocate_buffer(1)  # allocate new space in camera
            cam.arm()  # trigger arm
            job.resume()
    elif case == 3:
        
    elif case == 3:
        job = QM1.execute(FAST_ODMR)
        cam = cam_mock()  # create the camera object
        cam.allocate_buffer(2)   # allocate new space in camera
        cam.arm()  # trigger arm for the first time

        result_array = []

        for ind in range(0, N):
            while not job.is_paused():
                time.sleep(0.001)

            result_array.append(cam.get_image())  # collect previous image
            cam.allocate_buffer(1)  # allocate new space in camera
            cam.arm() #trigger arm
            job.resume()

