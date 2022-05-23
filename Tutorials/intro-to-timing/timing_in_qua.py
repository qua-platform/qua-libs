from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

qmm = QuantumMachinesManager()

with program() as example1:
    play("cw1", "qubit")
    play("cw2", "resonator")

with program() as example2:
    play("cw1", "qubit")
    wait(100)
    play("cw2", "resonator")

with program() as example3:
    play("cw1", "qubit")
    align("qubit", "resonator")
    play("cw2", "resonator")

with program() as example4:
    t = Random()
    play("cw1", "qubit", duration=4 * t.rand_int(50))
    align("qubit", "resonator")
    play("cw2", "resonator")

with program() as example5:
    play("cw1", "qubit")
    play("cw2", "qubit")

# Simulate and plot 1st example
job = qmm.simulate(config, example1, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(1)

plt.subplot(2, 1, 1)
samples.con1.plot("1")
plt.xlabel("Time[ns]")
plt.ylabel("qubit")
plt.title("example 1 - Two pulses from different elements")
plt.subplot(2, 1, 2)
samples.con1.plot("2")
plt.xlabel("Time[ns]")
plt.ylabel("resonator")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)


# Simulate and plot 2nd example
job = qmm.simulate(config, example2, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(2)

plt.subplot(2, 1, 1)
samples.con1.plot("1")
plt.xlabel("Time[ns]")
plt.ylabel("qubit")
plt.title("example 2 - Two pulses from different elements with wait command")
plt.subplot(2, 1, 2)
samples.con1.plot("2")
plt.xlabel("Time[ns]")
plt.ylabel("resonator")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Simulate and plot 3rd example
job = qmm.simulate(config, example3, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(3)
samples.con1.plot()
plt.title("example 3 - Two pulses from different elements \n with align command (deterministic case)")

# Simulate and plot 4th example
job = qmm.simulate(config, example4, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(4)
samples.con1.plot()
plt.title("example 4 - from different elements \n with align command (non-deterministic case)")

# Simulate and plot 5th example
job = qmm.simulate(config, example5, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(5)
samples.con1.plot()
plt.title("example 5 - Two pulses from the same element")
