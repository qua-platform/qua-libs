"""
multiplexed-readout.py: A template for FDM
Author: Ilan Mitnikov - Quantum Machines
Created: 10/11/2020
Created on QUA version: 0.5.170
"""
from configuration import config

from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt

simulation_config = SimulationConfig(
    duration=int(1e4),
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        latency=230,
        noisePower=0.05**2,
    ),
)
rr_num = 2  # currently works only for up to 2, due to 10 parallel demodulation limit (each rr requires 4 real demods)

wait_time = 10
with program() as multi_read:
    n = declare(int)
    I = declare(fixed, size=rr_num)
    Q = declare(fixed, size=rr_num)
    I1 = declare(fixed, size=rr_num)
    Q1 = declare(fixed, size=rr_num)
    I2 = declare(fixed, size=rr_num)
    Q2 = declare(fixed, size=rr_num)
    with for_(n, 1, n < 500, n + 1):
        align(*["rr" + str(i) for i in range(1, rr_num + 1)])
        for i in range(rr_num):
            measure(
                "readout_pulse",
                "rr" + str(i + 1),
                "adc",
                demod.full("integW_cos", I1[i], "out1"),
                demod.full("integW_sin", Q1[i], "out1"),
                demod.full("integW_cos", I2[i], "out2"),
                demod.full("integW_sin", Q2[i], "out2"),
            )

        wait(wait_time, *["rr" + str(i) for i in range(1, rr_num + 1)])
        for i in range(rr_num):
            assign(I[i], I1[i] + Q2[i])
            assign(Q[i], -Q1[i] + I2[i])
            save(I[i], "I" + str(i + 1))
            save(Q[i], "Q" + str(i + 1))

qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(multi_read, simulation_config)
for i in range(1, rr_num + 1):
    I_res = job.result_handles.get("I" + str(i)).fetch_all()["value"]
    Q_res = job.result_handles.get("Q" + str(i)).fetch_all()["value"]
    plt.plot(I_res, Q_res, ".")
