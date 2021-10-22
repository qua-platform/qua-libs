"""
pump_frequency_calibration.py: Script showcasing how the OPX can be used to generate Fig S8 a
Author: Arthur Strauss - Quantum Machines
Created: 24/03/2021
Created on QUA version: 0.80.769
"""

from configuration import *
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

simulation_config = SimulationConfig(
    duration=int(2e4),  # need to run the simulation for long enough to get all points
    simulation_interface=LoopbackInterface(
        [("con1", 1, "con1", 1), ("con1", 2, "con1", 2)],
        latency=200,
        noisePower=0.05 ** 2,
    ),
)

Nshots = 1
eps_max = 0.4
N_eps = eps_max // 0.05
f_d_init = 10e6
f_p_init = 10e6
f_d_final = 50e6
f_p_final = 50e6
df_d = 10e6
df_p = 10e6
N_f_p = f_p_final // df_p
N_f_d = f_d_final // df_d
with program() as freq_sweep:
    n = declare(int)
    f_pump = declare(int)
    f_drive = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)
    power = declare(fixed)
    I_stream = declare_stream()
    Q_stream = declare_stream()
    power_stream = declare_stream()

    with for_(n, 0, n < Nshots, n + 1):  # shots for averaging
        with for_(f_drive, f_d_init, f_drive < f_d_final, f_drive + df_d):
            with for_(f_pump, f_p_init, f_pump < f_p_final, f_pump + df_p):
                update_frequency("ATS", f_pump)
                update_frequency("buffer", f_drive)
                play("pump", "ATS", duration=2e2)
                play("drive", "buffer", duration=2e2)
                align()
                measure(
                    "drive",
                    "buffer",
                    "raw",
                    dual_demod.full("integW_cos", "out3", "integW_sin", "out4", I),
                    dual_demod.full(
                        "integW_minus_sin", "out3", "integW_cos", "out4", Q
                    ),
                )
                assign(power, Math.sqrt(I * I + Q * Q))
                save(power, power_stream)

    with stream_processing():
        power_stream.buffer(N_f_d, N_f_p).average().save_all("pow")

qmm = QuantumMachinesManager()
job = qmm.simulate(config, freq_sweep, simulation_config)
# job.get_simulated_samples().con1.plot()  # to see the output pulses
job.result_handles.wait_for_all_values()
time.sleep(3)
power = np.sqrt(job.result_handles.pow.fetch_all()["value"])

# Create plot for displaying acquired power (= S11) in terms of the frequencies of drive and pump
samples = job.get_simulated_samples()

plt.figure(1)
plt.plot(samples.con1.analog["1"])
plt.plot(samples.con1.analog["2"])
plt.plot(samples.con1.analog["5"] + 1)
plt.plot(samples.con1.analog["3"] + 2)
plt.plot(samples.con1.analog["4"] + 2)
plt.plot(samples.con1.analog["7"] + 3)
plt.plot(samples.con1.analog["8"] + 3)
plt.plot(samples.con1.analog["9"] + 4)
plt.plot(samples.con1.analog["10"] + 4)
plt.plot(samples.con2.analog["1"] + 5)
plt.plot(samples.con2.analog["2"] + 5)
plt.xlabel("Time [ns]")
plt.yticks(
    [0, 1, 2, 3, 4, 5],
    ["Transmon", "flux", "Readout", "storage", "ATS pump", "buffer drive"],
    rotation="vertical",
    va="center",
)
