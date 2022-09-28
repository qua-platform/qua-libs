import time

from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from quam import QuAM


with program() as hello_QUA:

    for z in range(4):
        play('dissipative_stabilization', f'q{z}_flux')
        wait(16, f'q{z}')
        play('cw', f'q{z}')
        wait(11, f'q{z}')
        play('Excitation', f'q{z}_flux')
        play('x90', f'q{z}')
        play('Free_evolution', f'q{z}_flux')
        play('Jump', f'q{z}_flux')
        align()
        wait(8, f'rr{z}')
        measure('readout', f'rr{z}', None)
        play('Readout', f'q{z}_flux')
        play('flux_balancing', f'q{z}_flux')

        align()
        wait(20)

machine = QuAM("quam_bootstrap_state.json")
config = machine.build_config()

qmm = QuantumMachinesManager(host='172.16.2.103', port='85')
simulation_duration = 2400  # clock cycle units - 4ns
job_sim = qmm.simulate(config, hello_QUA, SimulationConfig(simulation_duration))
job_sim.get_simulated_samples().con1.plot()
job_sim.get_simulated_samples().con2.plot()