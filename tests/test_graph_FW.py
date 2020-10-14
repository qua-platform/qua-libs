from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from qualibs.graph.environment import env_dependency, env_resolve
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
    # node.input_vars = {}
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}  # TODO: can I have a qua node without outputs?
    node.run()
    print(node.result)
    assert node.result['res'][0] == 1


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
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}

    graph = ProgramGraph('test_graph')
    graph.add_nodes([node])
    graph.run()

    assert node.result['res'][0] == 1


def test_save_graph_to_db():
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
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}
    graph_db = GraphDB('my_db.db')
    graph = ProgramGraph('test_graph')
    # graph = ProgramGraph('test_graph',graph_db) #another option
    graph.add_nodes([node])
    job_db = graph.run(graph_db)

    assert node.result['res'][0] == 1


def test_metadata_save():
    sim_args = {
        'simulate': SimulationConfig(int(1e3))}
    QMm = QuantumMachinesManager()
    QM = QMm.open_qm(config)
    envmodule={}
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

    def set_my_device():
        print('done')

    @env_dependency(envmodule)
    def my_device1():
        "returns a list object to emulate a device"
        print("opened device 1")
        return []

    @env_dependency(envmodule)
    def my_device2():
        "returns a list object to emulate a device"
        print("opened device 2")
        return []


    def get_dat_from_device1():
        a=my_device1()
        return 1

    def get_dat_from_device2():
        a=my_device2()
        return 2


    def py_node_script():
        set_my_device()

    dep_list = [get_dat_from_device1,get_dat_from_device2]
    print({dep.__name__:env_resolve(dep, envmodule)() for dep in dep_list})

    pnode = PyNode('py_node', py_node_script)

    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}
    # graph_db = GraphDB('my_db.db')
    graph = ProgramGraph('test_graph')
    # graph = ProgramGraph('test_graph',graph_db) #another option
    graph.add_nodes([node, pnode])
    job_db = graph.run()

    assert node.result['res'][0] == 1
