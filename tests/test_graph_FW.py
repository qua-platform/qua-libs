from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

from qualibs.graph.environment import env_dependency
from qualibs.graph import *
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


def compare_py_nodes(a: PyNode, b: PyNode):
    assert a.input_vars == b.input_vars
    assert a.label == b.label
    assert a.program == b.program
    assert a.output_vars == b.output_vars
    assert a.result == b.result
    assert a.id != b.id
    assert a.metadata_funcs == b.metadata_funcs


def compare_qua_nodes(a: QuaNode, b: QuaNode):
    assert a.input_vars == b.input_vars
    assert a.label == b.label
    assert a.program == b.program
    assert a.output_vars == b.output_vars
    assert a.result == b.result
    assert a.id != b.id
    assert a.quantum_machine == b.quantum_machine
    assert a.simulation_kwargs['simulate'].__dict__ == b.simulation_kwargs['simulate'].__dict__
    if a.execution_kwargs:
        assert a.execution_kwargs.__dict__ == b.execution_kwargs.__dict__


def test_node_deepcopy():
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

    b = node.deepcopy()
    compare_qua_nodes(node, b)


def test_graph_add_remove_nodes():
    a = PyNode('a', lambda x: {'x': x}, {'x': 1}, {'x'})
    b = PyNode('b', lambda x: {'x': x}, {'x': a.output('x')}, {'x'})
    c = PyNode('c', lambda x: {'x': x}, {'x': 1}, {'x'})
    nodes = [a.deepcopy() for i in range(5)]
    g = ProgramGraph('a')
    g.add_nodes(*nodes)
    assert list(g.nodes.values()) == nodes
    assert len(g.nodes.values()) == 5
    g.remove_nodes(nodes[0].id)
    assert list(g.nodes.values()) == nodes[1:]
    assert list(g.nodes.values()) != nodes
    g.add_nodes(b)
    nodes.append(b)
    g.remove_nodes(nodes[1])
    assert list(g.nodes.values()) == nodes[2:]
    assert list(g.nodes.values()) != nodes
    g.remove_nodes('a')
    assert list(g.nodes.values()) == [nodes[-1]]
    assert list(g.nodes.values()) != nodes
    g.add_nodes(c)
    g.remove_nodes('b', c)
    assert list(g.nodes.values()) == []


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
    graph.add_nodes(node)
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
    graph.add_nodes(node)
    job_db = graph.run(graph_db)
    print(node.result['res'])
    assert node.result['res'][0] == 1


def test_metadata_save():
    sim_args = {
        'simulate': SimulationConfig(int(1e3))}
    QMm = QuantumMachinesManager()
    QM = QMm.open_qm(config)
    envmodule = {}

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
        print("opened device 1")
        return []

    @env_dependency(envmodule)
    def my_device2():
        print("opened device 2")
        return []

    def py_node_script():
        set_my_device()

    def globalMeta(my_device2, my_device1):
        print('called global meta')
        res1 = my_device1()
        res2 = my_device2()
        return {'g1': res1}

    def myMeta():
        print('called local meta')
        return {'k1': 'v1'}

    pnode = PyNode('py_node', py_node_script, metadata_funcs=myMeta)
    # metadata func to resolve
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}
    # global metadata func to resolve
    graph_db = GraphDB('my_db.db', global_metadata_funcs=globalMeta, envmodule=envmodule)
    graph = ProgramGraph('test_graph', graph_db, verbose=True)

    graph.add_nodes(node, pnode)
    job_db = graph.run()

    assert node.result['res'][0] == 1


def test_graph_deepcopy():
    a = PyNode('a', lambda x: {'x': 1}, {'x': None}, {'x'})
    b = PyNode('b', lambda x: {'y': x}, {'x': a.output('x')}, {'y'})
    g = ProgramGraph()
    g.add_nodes(a, b)
    g.run()
    assert b.result['y'] == 1
    f = g.deepcopy()
    f.run()
    for v in f.nodes.values():
        if v.label == 'b':
            assert v.result['y'] == 1


def test_graph_deepcopy_2():
    sim_args = {
        'simulate': SimulationConfig(int(1e3))}
    QMm = QuantumMachinesManager()
    QM = QMm.open_qm(config)
    envmodule = {}

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

    node = QuaNode('qua_node', qua_prog)

    def set_my_device():
        print('done')

    @env_dependency(envmodule)
    def my_device1():
        print("opened device 1")
        return []

    @env_dependency(envmodule)
    def my_device2():
        print("opened device 2")
        return []

    def py_node_script():
        set_my_device()
        return {'x': 2, 'y': 3}

    def py2_node_script(x, y, m):
        print(x)
        return {'x': x, 'y': y, 'm': m}

    def globalMeta(my_device2, my_device1):
        print('called global meta')
        res1 = my_device1()
        res2 = my_device2()
        return {'g1': res1}

    def myMeta():
        print('called local meta')
        return {'k1': 'v1'}

    pnode = PyNode('py_node', py_node_script, output_vars={'x', 'y'}, metadata_funcs=myMeta)
    p2node = PyNode('py2_node', py2_node_script, {'x': pnode.output('x'), 'y': pnode.output('y'), 'm': node.qua_job()},
                    {'y', 'm'})
    # metadata func to resolve
    node.quantum_machine = QM
    node.simulation_kwargs = sim_args
    node.output_vars = {'res'}
    # global metadata func to resolve
    graph_db = GraphDB('my_db.db', global_metadata_funcs=globalMeta, envmodule=envmodule)
    graph = ProgramGraph('test_graph', graph_db, verbose=False)

    graph.add_nodes(node, pnode, p2node)

    g_cp = graph.deepcopy()
    graph.run()
    g_cp.run()
    g_py2 = graph.nodes_by_label['py2_node'].pop()
    g_py2_cp = g_cp.nodes_by_label['py2_node'].pop()
    # using qua_node.job() keeps the same QM
    assert g_py2.result['m'].quantum_machine is g_py2_cp.result['m'].quantum_machine
    g_qua = graph.nodes_by_label['qua_node'].pop()
    g_qua_cp = g_cp.nodes_by_label['qua_node'].pop()
    assert g_py2.result['m'].node is g_qua and g_py2_cp.result['m'].node is g_qua_cp
    for a, b in zip(graph.nodes.values(), g_cp.nodes.values()):
        if a.type == 'Py':
            compare_py_nodes(a, b)
        if a.type == 'Qua':
            compare_qua_nodes(a, b)

