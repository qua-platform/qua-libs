"""transmission_spectrum.py: Wide frequency sweep for metamaterial resonator
Author: Arthur Strauss - Quantum Machines
Created: 08/01/2021
QUA version used : 0.8.439
"""

# QM imports
from configuration import *
from mock_LO_source import mock_LO_source
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
import time
import matplotlib.pyplot as plt

qm1 = QuantumMachinesManager()
QM = qm1.open_qm(config)
freq_range = np.linspace(4e9, 12e9, 16)
with program() as t_spectrum:
    f = declare(int)
    LO_freq = declare(int)
    running = declare(bool, value=True)
    N = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_stream = declare_stream()
    Q_stream = declare_stream()
    f_stream = declare_stream()

    with while_(running):
        pause()
        assign(LO_freq, IO1)
        assign(running, IO2)

        with for_(f, 0, f <= 500e6, f + 1e6):
            update_frequency("RR", f, units="Hz")
            measure(
                "measure",
                "RR",
                None,
                demod.full("integW1", I),
                demod.full("integW2", Q),
            )
            save(I, I_stream)
            save(Q, Q_stream)
            save(f + LO_freq, f_stream)

    with stream_processing():
        I_stream.save_all("I")
        Q_stream.save_all("Q")
        f_stream.save_all("f")


job = QM.execute(t_spectrum)
LO_source = mock_LO_source()

for freq in freq_range:
    while not (job.is_paused()):
        time.sleep(0.001)
    LO_source.set_LO_frequency(freq)
    QM.set_io1_value(freq)
    job.resume()

QM.set_io2_value(False)

results = job.result_handles
I = results.I.fetch_all()["value"]
Q = results.Q.fetch_all()["value"]
f = results.f.fetch_all()["value"]
f_GHZ = f / 1e9
S_21 = np.sqrt(I**2 + Q**2)
plt.figure()
plt.plot(f, S_21)
plt.xlabel("Frequency [GHz]")
plt.ylabel("S_21 [dB]")
