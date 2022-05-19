

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from configuration import *
QMm = QuantumMachinesManager()

with program() as prog:
    play('const1', 'qe1')
    wait(100)
    play('const2', 'qe2')


job = QMm.simulate(config, prog, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()

plt.subplot(2, 1, 1)
samples.con1.plot('qe1')
plt.xlabel('Time[ns]')
plt.ylabel('qe1')

plt.subplot(2, 1, 2)
samples.con1.plot('qe2')
plt.xlabel('Time[ns]')
plt.ylabel('qe2')

plt.subplots_adjust(left=0.1,
                    bottom=0.1,
                    right=0.9,
                    top=0.9,
                    wspace=0.4,
                    hspace=0.4)

