"""
intro-to-macros.py: Your first QUA script
Author: Gal Winer - Quantum Machines
Created: 26/12/2020
Created on QUA version: 0.6.393
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import config


QMm = QuantumMachinesManager()

def declare_vars(stream_num=1):
    a = declare(int,value=1)
    b = declare(fixed,value=0.2)
    stream_array = [declare_stream() for num in range(stream_num)]
    return [a,b,stream_array]

def modify_var(b):
    assign(b,b+1)

with program() as prog:
    [a,b,c_streams] = declare_vars()
    save(b,c_streams[0])
    play('playOp' * amp(b), 'qe1', duration=100 * a)
    modify_var(b)
    save(b,c_streams[0])
    play('playOp'*amp(b), 'qe1',duration=100*a)

    with stream_processing():
        c_streams[0].save_all('out_stream')

QM1 = QMm.open_qm(config)
job = QM1.simulate(prog,
                   SimulationConfig(int(1000)))
res = job.result_handles
out_str = res.out_stream.fetch_all()
samples = job.get_simulated_samples()
samples.con1.plot()
