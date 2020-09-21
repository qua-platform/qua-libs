from abc import ABC, abstractmethod
from qm import QuantumMachine
from qm import SimulationConfig
import qm
from types import *
from typing import *

from typing import Set


class ProgramNode(ABC):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: dict = None, _output_vars: set = None,
                 _to_run: bool = True):
        """
        Program node contains a program to run and description of input/output variables
        :param _label: label for the node
        :type: _label: str
        :param _program: a python function to run
        :type _program: function
        :param _input: input variables names and values
        :type _input: dict
        :param _output_vars: output variable names
        :type _output_vars: set
        :param _to_run: whether to run the node
        :type _to_run: bool
        """
        self._id: int = id(self)
        self._label: str = None
        self._program: FunctionType = None
        self._input: Dict = None
        self._to_run: bool = None
        self._output_vars: set = None
        self._output: dict = None
        self._timestamp = None

        self.label = _label
        self.program = _program
        self.input = _input
        self.output_vars = _output_vars
        self.to_run = _to_run

    @property
    def id(self):
        return self._id

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, _program):
        if _program is not None:
            assert type(_program) is FunctionType, \
                "TypeError: Expected FunctionType but given <{}>".format(type(_program))
        self._program = _program

    @property
    def input(self):
        return self._input

    @input.setter
    def input(self, _input):
        if _input is not None:
            assert type(_input) is dict, \
                "TypeError: Try a different input. Expected <dict> but given <{}>".format(type(_input))
            self._input = _input

    @property
    def output_vars(self):
        return self._output_vars

    @output_vars.setter
    def output_vars(self, _output_vars):
        if _output_vars is not None:
            assert type(_output_vars) is set, \
                "TypeError: Try a different output_vars. Expected <set> but given <{}>".format(type(_output_vars))
            self._output_vars = _output_vars

    @abstractmethod
    def get_output(self):
        pass
        return self._output

    def output(self, _output_vars=None):
        return OutputNode(self, _output_vars)

    @property
    @abstractmethod
    def timestamp(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @property
    def to_run(self):
        return self._to_run

    @to_run.setter
    def to_run(self, to_run):
        assert type(to_run) is bool, "TypeError: Expected bool but given {}".format(type(to_run))
        self._to_run = to_run


class OutputNode:
    def __init__(self, node, output_vars=None):
        self.node: ProgramNode = node
        if output_vars is not None:
            for var in output_vars:
                assert var in self.node.output_vars, \
                    "KeyError: Output of node <{}> doesn't contain the variable <{}>".format(self.node.label, var)
        self.output_vars: set = output_vars


class QuaNode(ProgramNode, ABC):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: dict = None, _output_vars: set = None,
                 quantum_machine: QuantumMachine = None, _simulate_or_execute: str = 'simulate'):

        super().__init__(_label, _program, _input, _output_vars)
        self._job: qm.QmJob.QmJob = None
        self._qua_program: qm.program._Program = None
        self._quantum_machine: QuantumMachine = None
        self._simulate_or_execute: str = None

        self.quantum_machine = quantum_machine
        self.simulate_or_execute = _simulate_or_execute

    @property
    def quantum_machine(self):
        return self._quantum_machine

    @quantum_machine.setter
    def quantum_machine(self, quantum_machine):
        if quantum_machine is not None:
            assert isinstance(quantum_machine, QuantumMachine), \
                "TypeError: Expected QuantumMachine but given <{}>".format(type(quantum_machine))
        self._quantum_machine = quantum_machine

    @property
    def simulate_or_execute(self):
        return self._simulate_or_execute

    @simulate_or_execute.setter
    def simulate_or_execute(self, s_or_e):
        assert s_or_e == 'simulate' or s_or_e == 'execute', \
            "ValueError: Expected 'simulate' or 'execute' but got {}".format(s_or_e)
        self._simulate_or_execute = s_or_e

    def get_output(self):
        for var in self._output_vars:
            try:
                self._output[var] = self._job.result_handles[var]
            except KeyError:
                print("Couldn't fetch {} from Qua program results".format(var))

        return self._output

    @property
    def timestamp(self):
        pass

    def run(self, **kwargs):

        # Get the Qua program that is wrapped by the python function
        qua_program = self.program(**self._input)
        assert isinstance(qua_program, qm.program._Program), \
            "In node <id:{},label:{}> TypeError: Try a different program. " \
            "Expected <qm.program._Program> but given <{}>".format(self.id, self.label, type(qua_program))
        self._qua_program = qua_program

        if self._simulate_or_execute == 'simulate':
            self.simulate(**kwargs)
        if self._simulate_or_execute == 'execute':
            self.execute(**kwargs)

    def execute(self, **kwargs):
        self._job = self._quantum_machine.execute(self._qua_program, **kwargs)

    def simulate(self, sim_config=SimulationConfig(), **kwargs):
        self._job = self._quantum_machine.simulate(self._qua_program, sim_config, **kwargs)


class PyNode(ProgramNode):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: dict = None, _output_vars: set = None,):
        super().__init__(_label, _program, _input, _output_vars)
        self._job_results = None

    def get_output(self):
        for var in self._output_vars:
            try:
                self._output[var] = self._job_results[var]
            except KeyError:
                print("Couldn't fetch {} from Qua program results".format(var))

        return self._output

    @property
    def timestamp(self):
        pass

    def run(self):
        self._job_results = self.program(**self.input)
        assert type(self._job_results) is dict, \
            "TypeError: Expected <dict> but got <{}> as program results".format(type(self._job_results))


class ProgramGraph:

    def __init__(self, _label: str = None):
        """
        A program graph describes a program flow with input/output dependencies
        :param _label: a label for the graph
        :type _label: str
        """
        self._id: int = id(self)
        self._label: str = None
        self._nodes: Dict[int, ProgramNode] = None
        self._node_counter: int = 0
        self._edges: Dict[int, set] = None
        self._backward_edges: Dict[int, set] = None
        self._timestamp = None
        self._output: dict = None

        self.label = _label

    @property
    def id(self):
        return self._id

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, _label):
        self._label = _label

    @property
    def nodes(self):
        return self._nodes

    def add_nodes(self, new_nodes: List[ProgramNode]):
        """
        Adds the given nodes to the graph
        :param new_nodes: list of node objects
        :type new_nodes: List[ProgramNode]
        :return:
        """
        for node in new_nodes:
            self._nodes[node.id] = node
            self._node_counter += 1
            for var, value in node.input.items():
                if isinstance(value, OutputNode):
                    self.add_edges([(value.node, node)])

    def remove_nodes(self, nodes_to_remove: Set[ProgramNode]):
        """
        Removes the given nodes from the graph
        :param nodes_to_remove: Set of nodes to remove
        :type nodes_to_remove: Set[ProgramNode]
        :return:
        """
        edges_to_remove: Set[Tuple[ProgramNode, ProgramNode]] = set()
        for source_node in nodes_to_remove:
            try:
                ids_to_remove = self._nodes.pop(source_node.id)
                for dest_node_id in ids_to_remove:
                    edges_to_remove.add((source_node, self.nodes[dest_node_id]))
            except KeyError:
                print("KeyError: Tried to remove node <{}>, but was not found".format(source_node.label))
        self.remove_edges(edges_to_remove)

    @property
    def edges(self):
        return self._edges

    def add_edges(self, _edges: Set[Tuple[ProgramNode, ProgramNode]]):
        """
        Add edges between given nodes
        When used outside of add_nodes method, it describes either time order or input/output dependency as usual.
        :param _edges: set of tuples {(source_node, dest_node)...}
        :type _edges: Set[Tuple[ProgramNode, ProgramNode]]
        :return:
        """
        for source, dest in _edges:
            self._edges.setdefault(source.id, set()).add(dest.id)
            self._backward_edges.setdefault(dest.id, set()).add(source.id)

    def remove_edges(self, _edges: Set[Tuple[ProgramNode, ProgramNode]]):
        """
        Remove edges from graph
        :param _edges: set of tuples {(source_node, dest_node)...}
        :type _edges: Set[Tuple[ProgramNode, ProgramNode]]
        :return:
        """
        # need to update backward edges
        for source, dest in _edges:
            try:
                self._edges.get(source.id, set()).remove(dest.id)
                self._backward_edges.get(dest.id, set()).remove(source.id)
                print("Successfully removed edge from <{}> to <{}>".format(source.id, dest.id))
            except KeyError:
                print("KeyError: Tried to remove edge from <{}> to <{}>, "
                      "but it doesn't exist.".format(source.id, dest.id))



    @property
    def backward_edges(self):
        return self._backward_edges

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def output(self):
        return self._output

    def run(self, start_nodes=None):
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs accordingly
        :param start_nodes: list of nodes to start running the graph from
        :type: start_nodes: list
        :return:
        """

    def plot(self, start_nodes=None):
        """
        Plot starting from the given node and in the direction of propagation
        :param start_nodes: list of nodes to start plotting from
        :return:
        """

    def merge(self, graph):
        """
        Merge graph into the current graph
        :param graph:
        :return:
        """
        pass
