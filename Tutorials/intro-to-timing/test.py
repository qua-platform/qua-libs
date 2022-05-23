from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *

qmm = QuantumMachinesManager()

with program() as example1:
    play("cw1", "qubit1")
    play("cw2", "qe2")

# Simulate and plot 1st example
job = qmm.simulate(config, example1, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.figure(1)

plt.subplot(2, 1, 1)
samples.con1.plot("qubit1")
plt.xlabel("Time[ns]")
plt.ylabel("qubit")

plt.subplot(2, 1, 2)
samples.con1.plot("qe2")
plt.xlabel("Time[ns]")
plt.ylabel("qe2")

plt.subplots_adjust(left=0.1, bottom=0.1, right=0.9, top=0.9, wspace=0.4, hspace=0.4)
