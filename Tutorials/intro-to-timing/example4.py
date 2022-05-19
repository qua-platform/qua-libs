
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
QMm = QuantumMachinesManager()

with program() as prog:
    t = Random()
    play('const1', 'qe1', duration=4 * t.rand_int(50))
    align('qe1', 'qe2')
    play('const2', 'qe2')


job = QMm.simulate(config, prog, SimulationConfig(int(1000)))  # in clock cycles, 4 ns
samples = job.get_simulated_samples()
samples.con1.plot()