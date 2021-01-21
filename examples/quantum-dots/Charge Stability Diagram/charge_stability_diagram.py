from configuration import config

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager, SimulationConfig, LoopbackInterface
import numpy as np

qmm = QuantumMachinesManager()

v1_start = -0.2
v1_end = 0
v2_start = -0.15
v2_end = 0.10
step = 0.01
n_avg = 100


def scan2d(v1_start, v1_end, v2_start, v2_end, step, n_avg):
    n_v1 = int((v1_end - v1_start) / step)
    n_v2 = int((v2_end - v2_start) / step)
    with program() as prog:
        v1 = declare(fixed)
        v2 = declare(fixed)
        I = declare(fixed)
        n = declare(int)
        I_avg = declare_stream()
        with for_(n, 0, n < n_avg, n + 1):
            with for_(v1, v1_start, v1 < v1_end, v1 + step):
                with for_(v2, v2_start, v2 < v2_end, v2 + step):
                    align("PG1", "PG2", "QPC")
                    play("playOp" * amp(v1), "PG1")
                    play("playOp" * amp(v2), "PG2")
                    measure("readout", "QPC", None, integration.full("integW", I, "out1"))
                    save(I, I_avg)
        with stream_processing():
            I_avg.buffer(n_v1, n_v2).average().save("current")
    return prog


def charge_stability_patch(v1_start, v1_end, v2_start, v2_end, step, n_avg, qm):
    job = qm.execute(scan2d(v1_start, v1_end, v2_start, v2_end, step, n_avg))
    job.result_handles.wait_for_all_values()
    current = job.result_handles.current.fetch_all()
    # calculate the derivative of the current for the charge stability diagram
    charge_stability = (np.gradient(current, step)[0] + np.gradient(current, step)[1]) / 2
    return charge_stability


qm = QuantumMachinesManager().open_qm(config)
a = charge_stability_patch(v1_start, v1_end, v2_start, v2_end, step, n_avg, qm)