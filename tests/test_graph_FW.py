from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qualibs.graph.program_node import *
from qualibs.graph.program_graph import *
from qualibs.templates import hello_qua
from qualibs.templates.vanilla_config import config


def test_run_pyNode():
    def myfunc(a):
        if a == 1:
            return {'out1': True}
        else:
            return {'out1': False}

    node = PyNode('test', myfunc, {'a': 1}, {'out1'})
    node.run()
    assert node.result['out1']


def test_run_quaNode():
    sim_args = {
        'simulate': SimulationConfig(int(1e3))}
    QMm = QuantumMachinesManager()
    QM = QMm.open_qm(config)

    def qua_prog():
        with program() as prog:
            play('playOp', 'qe1')
        return prog
    node = QuaNode('node_name', qua_prog)
    node.input_vars = {}
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'OUT'} #TODO: can I have a qua node without outputs? 
    node.run()
    assert 1
