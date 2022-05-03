# Importing the necessary from qm
import time

import matplotlib.pyplot as plt
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from configuration import *

resolution = 2  # In ns, can be 1,2 or 4.
# All of the numbers here are multiplied by the resolution above.
# For example, if the resolution is 1ns, then the pulse will go from 4 ns to 10 ns in jumps of 1ns.
# if the resolution is 2ns, then the pulse will go from 8 ns to 20 ns in jumps of 2ns.
t_start = 4  # Minimum pulse duration (in clock cycles, minimum is 4)
t_max = 10  # Maximum pulse duration (in clock cycles)
dt = 1  # timestep
N_t = int(t_max / dt)  # Number of timesteps
n_repeats = 1

qmManager = QuantumMachinesManager()
my_qm = qmManager.open_qm(config)  # Generate a Quantum Machine based on the configuration described above

with program() as timeRabiProg:  # Time Rabi QUA program
    t = declare(int)  # Sweeping parameter over the set of durations
    r = declare(int)  # Number of repetitions of the experiment

    with for_(r, 0, r < n_repeats, r + 1):  # Do a n_repeats times the experiment to obtain statistics
        with for_(t, t_start, t <= t_max, t + dt):  # Sweep the pulse duration from t_start to t_max
            play(f"gauss_pulse_{resolution}ns_res", "qubit", duration=t)


# job = my_qm.execute(timeRabiProg)

my_job = my_qm.simulate(timeRabiProg, SimulationConfig(int(1500)))
time.sleep(1.0)
my_timeRabi_results = my_job.result_handles

samples = my_job.get_simulated_samples()
I = samples.con1.analog.get("1")
Q = samples.con1.analog.get("1")
out = np.sqrt(I**2 + Q**2)
plt.plot(out)
