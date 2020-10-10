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
            res = declare(int)
            res_str = declare_stream()
            play('playOp', 'qe1')
            assign(res, 1)
            save(res, res_str)
            with stream_processing():
                res_str.save_all('res')
        return prog

    node = QuaNode('node_name', qua_prog)
    node.input_vars = {}
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}  # TODO: can I have a qua node without outputs?
    node.run()
    assert node.result['res'][0]==1

def test_make_graph_with_quaNode():
    sim_args = {
        'simulate': SimulationConfig(int(1e3))}
    QMm = QuantumMachinesManager()
    QM = QMm.open_qm(config)

    def qua_prog():
        with program() as prog:
            res = declare(int)
            res_str = declare_stream()
            play('playOp', 'qe1')
            assign(res, 1)
            save(res, res_str)
            with stream_processing():
                res_str.save_all('res')
        return prog

    node = QuaNode('node_name', qua_prog)
    node.input_vars = {}
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}

    graph = ProgramGraph('test_graph', {'output': node.output('res')})
    graph.add_nodes([node])
    gjob = graph.run()
    assert 1
