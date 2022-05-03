"""
g2-with-stage.py: Single NVs confocal microscopy: Intensity autocorrelation g2 as a function of stage position
Author: Gal Winer - Quantum Machines
Created: 13/12/2020
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import numpy as np
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

stage = stage_mock()
correlation_width = 200
meas_len = 2000  # 2us
N_avg = 1e6  # total measurement time will be: meas_len*N_avg
result_array = []


###############
# confocal g2 #
###############
with program() as confocal_g2:

    i = declare(int)
    n = declare(int)
    n_avg = declare(int)
    k = declare(int)
    p = declare(int)
    diff = declare(int)
    diff_index = declare(int)
    resultLen1 = declare(int)
    resultLen2 = declare(int)
    # For a 2us readout, this limits the number of photons at 4, which corresponds to 2MHz counts.
    # This can be increased if needed.
    result1 = declare(int, size=int(meas_len / 500))
    result2 = declare(int, size=int(meas_len / 500))
    g2 = declare(int, size=int(2 * correlation_width))
    g2_stream = declare_stream()
    counts = declare(int)

    #########################
    # looping over position #
    #########################
    with for_(i, 0, i < Nx * Ny, i + 1):

        pause()

        ########################
        # initialize g2 vector #
        ########################
        with for_(p, 0, p < g2.length(), p + 1):
            assign(g2[p], 0)

        # Total number of counts
        assign(counts, 0)

        ##########################
        # perform g2 measurement #
        ##########################
        with for_(n_avg, 0, n_avg <= N_avg, n_avg + 1):
            play("readout", "AOM")
            measure(
                "readout",
                "spcm1",
                None,
                time_tagging.raw(result1, meas_len, resultLen1),
            )
            measure(
                "readout",
                "spcm2",
                None,
                time_tagging.raw(result2, meas_len, resultLen2),
            )
            assign(n, 0)
            assign(k, 0)
            assign(counts, counts + resultLen2 + resultLen1)  # Total number of counts.
            with while_(k < resultLen2):
                with while_(n < resultLen1):
                    assign(diff, result1[n] - result2[k])

                    with if_((diff < correlation_width) & (diff > -correlation_width)):
                        assign(diff_index, diff + correlation_width)
                        assign(g2[diff_index], g2[diff_index] + 1)
                        assign(n, n + 1)
                        assign(k, k + 1)

                    with if_(diff >= correlation_width):
                        pass

                    with if_(diff <= correlation_width):
                        assign(n, n + 1)

                assign(k, k + 1)

        ####################
        # stream g2 vector #
        ####################
        with for_(p, 0, p < g2.length(), p + 1):
            save(g2[p], g2_stream)
        save(counts, "counts")

    with stream_processing():
        g2_stream.buffer(2 * correlation_width).save_all("g2")  # Take g2 vector per position.


job = QM1.execute(confocal_g2)
res = job.result_handles
x_vec = np.arange(x_start, x_end, x_step)
y_vec = np.arange(y_start, y_end, y_step)
g2_data = np.zeros([len(x_vec), len(y_vec)], correlation_width)
counts_data = np.zeros([len(x_vec), len(y_vec)])


##############
# move stage #
##############
for x_i in range(len(x_vec)):
    for y_i in range(len(x_vec)):
        stage.go_to(pos=(x_vec[x_i], y_vec[y_i]))
        job.resume()
        while not job.is_paused():
            time.sleep(0.001)
        g2_data[x_i, y_i] = res.g2_stream.fetch_all()
        counts_data = res.counts.fetch_all()["value"]
