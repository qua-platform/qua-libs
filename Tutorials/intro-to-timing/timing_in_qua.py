from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

QMm = QuantumMachinesManager()

with program() as example1:
    play("const1", "qe1")
    play("const2", "qe2")

with program() as example2:
    play("const1", "qe1")
    wait(100)
    play("const2", "qe2")

with program() as example3:
    play("const1", "qe1")
    align("qe1", "qe2")
    play("const2", "qe2")

with program() as example4:
    t = Random()
    play("const1", "qe1", duration=4 * t.rand_int(50))
    align("qe1", "qe2")
    play("const2", "qe2")

with program() as example5:
    play("const1", "qe1")
    play("const2", "qe1")

# Simulate and plot 1st example
job = QMm.simulate(config, example1, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.subplot(2, 1, 1)
samples.con1.plot("qe1")
plt.xlabel("Time[ns]")
plt.ylabel("qe1")

plt.subplot(2, 1, 2)
samples.con1.plot("qe2")
plt.xlabel("Time[ns]")
plt.ylabel("qe2")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Simulate and plot 2nd example
job = QMm.simulate(config, example2, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.subplot(2, 1, 1)
samples.con1.plot("qe1")
plt.xlabel("Time[ns]")
plt.ylabel("qe1")

plt.subplot(2, 1, 2)
samples.con1.plot("qe2")
plt.xlabel("Time[ns]")
plt.ylabel("qe2")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)

# Simulate and plot 3rd example
job = QMm.simulate(config, example3, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()
samples.con1.plot()

# Simulate and plot 4th example
job = QMm.simulate(config, example4, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()
samples.con1.plot()

# Simulate and plot 5th example
job = QMm.simulate(config, example5, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()
samples.con1.plot()
