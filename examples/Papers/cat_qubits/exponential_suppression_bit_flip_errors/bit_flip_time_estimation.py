"""
bit_flip_time_estimation.py: Script showcasing how the OPX can be used to generate Fig 3
Author: Arthur Strauss - Quantum Machines
Created: 24/03/2021
Created on QUA version: 0.80.769
"""

from configuration import *
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import numpy as np


simulation_config = SimulationConfig(
    duration=int(2e5),  # need to run the simulation for long enough to get all points
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        latency=200,
        noisePower=0.05 ** 2,
    ),
)

revival_time = int(np.pi / chi_qa / 4) * 4  # get revival time in multiples of 4 ns

amp_array = [0.05 * i for i in range(8)]  # We assume that each amplitude corresponds to a specific photon number
N_a = len(amp_array)
points = 10
shots = 1
t_min = 1e6
t_max = 2e6 // 4
dt = 2e3 // 4
t_vec = np.arange(t_min, t_max, dt)
N_t = len(t_vec)

with program() as bit_flip_time_estimation:
    th = declare(fixed, value=1.0)  # threshold for state discrimination
    n = declare(int)
    a = declare(fixed)
    blob = declare(int)
    t = declare(int)
    I, I1, I2, Q = declare(fixed), declare(fixed), declare(fixed), declare(fixed)
    Q1, Q2 = declare(fixed), declare(fixed)
    state = declare(bool)
    I_stream, Q_stream, state_stream = declare_stream(), declare_stream(), declare_stream()

    with for_(n, 0, n < shots, n + 1):  # shots for averaging
        with for_each_(a, amp_array):
            with for_(t, 0, t < t_max, t + dt):  # Buffer drive duration
                with for_(blob, -1, blob <= 1, blob + 2):
                    play("Pump_Op", "ATS", duration=t)
                    play("drive" * amp(a), "buffer", duration=t)
                    align()

                    # Measurement

                    play("Displace_Op" * amp(blob), "storage")  # Displace towards the two blobs |alpha> & |- alpha>
                    play("X90", "transmon")
                    wait(revival_time, "transmon")
                    play("X90", "transmon")

                    align()
                    measure(
                        "Readout_Op",
                        "RR",
                        "raw",
                        demod.full("integW_cos", I1, "out1"),
                        demod.full("integW_sin", Q1, "out1"),
                        demod.full("integW_cos", I2, "out2"),
                        demod.full("integW_sin", Q2, "out2"),
                    )
                    assign(I, I1 + Q2)
                    assign(Q, -Q1 + I2)
                    assign(state, I > th)
                    save(I, I_stream)
                    save(Q, Q_stream)
                    save(state, state_stream)
                    wait(10, "storage", "transmon", "RR")  # wait and let all elements relax

    with stream_processing():
        state_stream.boolean_to_int().buffer(N_a, N_t, 2).average().save("parity")


qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
job = qm.simulate(bit_flip_time_estimation, simulation_config)
results = job.result_handles
# job.get_simulated_samples().con1.plot()  # to see the output pulses
results.wait_for_all_values()

P_excited = results.parity.fetch_all()["value"]
Wigner = np.zeros_like(P_excited)
for i in range(N_a):
    for j in range(N_t):
        for k in range(2):
            Wigner[i][j][k] = (4 * P_excited[i][j][k] - 1) / np.pi
