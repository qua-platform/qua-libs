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
    """
    A macro to declare QUA variables. Stream num showcases a way to declare multiple streams in an array
    Note that variables and streams need to be explicitly returned to the QUA function to be in scope
    """
    a = declare(int,value=100)
    b = declare(fixed,value=0.2)
    stream_array = [declare_stream() for num in range(stream_num)]
    return [a,b,stream_array]

def modify_var(addition=1):
    """
    A macro to modify a QUA variable. In this case, the variable does not
    need to be passed.
    """
    assign(b,b+addition)

def qua_function_calls(el):
    """
    A macro that calls QUA play statements
    :param el: The quantum element used by the QUA statments
    :return:
    """
    play('playOp',el,duration=300)
    play('playOp'*amp(b), el,duration=300)

with program() as prog:
    [a,b,c_streams] = declare_vars()
    save(b,c_streams[0])
    play('playOp' * amp(b), 'qe1', duration=a)
    modify_var()
    save(b,c_streams[0])
    play('playOp'*amp(b), 'qe1',duration=a)
    qua_function_calls('qe1')

    with stream_processing():
        c_streams[0].save_all('out_stream')

QM1 = QMm.open_qm(config)
job = QM1.simulate(prog,
                   SimulationConfig(int(4000)))
res = job.result_handles
out_str = res.out_stream.fetch_all()
samples = job.get_simulated_samples()
samples.con1.plot()
