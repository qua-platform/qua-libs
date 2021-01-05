"""
g2-with-stage.py: Single NVs confocal microscopy: Intensity autocorrelation g2 as a function of stage position
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
from stage_mock_lib import *
import time

QMm = QuantumMachinesManager()
QM1 = QMm.open_qm(config)

x_start = 0
x_end = 20
y_start = 0
y_end = 20
Nx = 10
Ny = 10
x_step = int((x_end - x_start) / Nx)
y_step = int((y_end - y_start) / Ny)

T1 = 10 * 1e3  # in nS
flash_time = 20 * 1e3
N_avg = 2

stage = stage_mock()
correlation_width = 200
meas_len = 2000
result_array = []


with program() as confocal_g2:
    x = declare(int)
    y = declare(int)
    n = declare(int)
    n_avg = declare(int)
    k = declare(int)
    p = declare(int)
    g2 = declare(int, size=int(2 * correlation_width))
    diff = declare(int)
    resultLen1 = declare(int)
    resultLen2 = declare(int)
    result1 = declare(int, size=int(meas_len / 500))
    result2 = declare(int, size=int(meas_len / 500))
    g2_stream=declare_stream()
    with for_(x, x_start, x <= x_end, x + x_step):
        with for_(y, y_start, y <= y_end, y + y_step):
            assign(IO1, x)
            assign(IO2, y)
            pause()
            with for_(n_avg, 0, n_avg <= N_avg, n_avg + 1):
                play('SAT', 'qubit', duration=10 * T1)

                measure("readout", "readout_el1", None,
                        time_tagging.raw(result1, meas_len, targetLen=resultLen1))  # 1ns
                measure("readout", "readout_el2", None,
                        time_tagging.raw(result2, meas_len, targetLen=resultLen2))  # 1ns
                assign(n, 0)
                with for_(k, 0, k < resultLen2, k + 1):
                    with if_(n < resultLen1):
                        assign(diff, result1[n] - result2[k])

                        with if_((diff < correlation_width) & (diff > -correlation_width)):
                            assign(diff, diff + correlation_width)
                            assign(g2[diff], g2[diff] + 1)
                            assign(n, n + 1)

        with for_(p, 0, p < g2.length(), p + 1):
            save(g2[p], g2_stream)

    with stream_processing():
        g2_stream.buffer(2 * correlation_width).average().save_all('g2') #take g2 vector per position. (buffer shape is wrong)

job = QM1.execute(confocal_g2)

res = job.result_handles
for ind in range(0, Nx * Ny):
    while not job.is_paused():
        time.sleep(0.001)
    (x_pos, y_pos) = QM1.get_io_values()
    stage.go_to(pos=(x_pos['int_value'], y_pos['int_value']))
    job.resume()

# res=job.result_handles
# samples = job.get_simulated_samples()
# samples.con1.plot()
