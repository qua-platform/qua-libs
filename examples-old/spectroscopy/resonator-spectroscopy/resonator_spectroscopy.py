"""
resonator_spectroscopy.py: Resonator spectroscopy to find the resonance frequency
Author: Niv Drucker - Quantum Machines
Created: 8/11/2020
Created on QUA version: 0.6.156
"""

from configuration import config
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from qm import LoopbackInterface
import matplotlib.pyplot as plt

with program() as resonator_spectroscopy:
    n = declare(int)
    f = declare(int)

    I = declare(fixed)
    I_stream = declare_stream()

    Q = declare(fixed)
    Q_stream = declare_stream()

    A = declare_stream()
    with for_(n, 0, n < 1000, n + 1):
        with for_(f, 1e6, f < 100.5e6, f + 1e6):
            wait(100, "rr")  # wait 100 clock cycles (4microS) for letting resonator relax to vacuum

            update_frequency("rr", f)
            measure(
                "long_readout",
                "rr",
                None,
                demod.full("long_integW1", I, "out1"),
                demod.full("long_integW2", Q, "out1"),
            )

            save(I, I_stream)
            save(Q, Q_stream)
    with stream_processing():
        I_stream.buffer(100).average().save("I")
        Q_stream.buffer(100).average().save("Q")
qmm = QuantumMachinesManager()
qm = qmm.open_qm(config)
# need to run the simulation for a sufficiently long time, because the buffer outputs results only when full
job = qm.simulate(
    resonator_spectroscopy,
    SimulationConfig(int(1e6), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)
res_handle = job.result_handles
Q_handle = res_handle.get("Q")
I_handle = res_handle.get("I")
Q = Q_handle.fetch_all()
I = I_handle.fetch_all()
plt.plot(I**2 + Q**2)
