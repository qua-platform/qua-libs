
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

from configuration import *
QMm = QuantumMachinesManager()

with program() as prog:
    play('const1', 'qe1')
    play('const2', 'qe1')


job = QMm.simulate(config, prog, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()
samples.con1.plot()