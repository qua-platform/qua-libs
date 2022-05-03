# Importing the necessary from qm
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua import math
from qm import LoopbackInterface
from qm import SimulationConfig
import numpy as np
import matplotlib.pyplot as plt
import time
from configuration import *
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

t_max = 5  # Maximum pulse duration (in clock cycles, 1 clock cycle =4 ns)
dt = 1  # timestep
N_t = int(np.round(t_max / dt))  # Number of timesteps
N_max = 2
a_max = 0.3  # Maximum amplitude
da = 0.05  # amplitude sweeping step
N_a = int(np.round(a_max / da))  # Number of steps

qmManager = QuantumMachinesManager()  # Reach OPX's IP address
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as time_powerRabiProg:  # Mix up of the power and time Rabi QUA program
    I = declare(fixed)  # QUA variables declaration
    Q = declare(fixed)
    t = declare(int)  # Sweeping parameter over the set of durations
    a = declare(fixed)  # Sweeping parameter over the set of amplitudes
    I_stream = declare_stream()  # Declare streams to store I and Q components
    Q_stream = declare_stream()
    time_stream = declare_stream()
    amp_stream = declare_stream()
    Nrep = declare(int)  # Number of repetitions of the experiment

    with for_(Nrep, 0, Nrep < N_max, Nrep + 1):  # Do a 100 times the experiment to obtain statistics
        with for_(a, 0.0, a < a_max, a + da):  # Sweep for varying amplitudes
            with for_(t, 0, t < t_max, t + dt):  # Sweep from 0 to t_max *4 ns the pulse duration

                play("gauss_pulse" * amp(a), "qubit", duration=t)
                align("qubit", "RR")
                measure("meas_pulse", "RR", "samples", ("integW1", I), ("integW2", Q))
                save(I, I_stream)
                save(Q, Q_stream)
                save(t, time_stream)
            save(a, amp_stream)
    with stream_processing():
        I_stream.buffer(N_a, N_t).average().save(
            "I"
        )  # Use stream_processing to retrieve shaped data to perform easy post processing
        Q_stream.buffer(N_a, N_t).average().save("Q")
        amp_stream.buffer(N_a).save("a")
        time_stream.buffer(N_t).save("t")

my_job = my_qm.simulate(
    time_powerRabiProg,
    SimulationConfig(int(50000), simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)])),
)
time.sleep(1.0)
my_timeRabi_results = my_job.result_handles
I1 = my_timeRabi_results.I.fetch_all()  # Retrieve raw data
Q1 = my_timeRabi_results.Q.fetch_all()
t1 = my_timeRabi_results.t.fetch_all()  # Retrieve only once the set of pulse durations
a1 = my_timeRabi_results.a.fetch_all()  # Retrieve set of amplitudes swept
samples = my_job.get_simulated_samples()
# samples.con1.plot()


fig = plt.figure()

# Plot the surface.
plt.pcolormesh(t1, a1, I1, shading="nearest")
plt.xlabel("Pulse duration [ns]")
plt.ylabel("Amplitude [a.u]")
plt.colorbar()
plt.title("I component expressed for varying pulse amplitude and duration")
