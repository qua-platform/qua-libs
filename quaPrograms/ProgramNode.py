from abc import ABC, abstractmethod
from qm import QuantumMachine
from qm import SimulationConfig
import qm
from types import *


class ReferenceNode:
    def __init__(self, node, output_vars=None):
        self.node: ProgramNode = node
        self.output_vars: set = output_vars


class ProgramNode(ABC):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None, _to_run=True):
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
        self._id: int = None
        self._label: str = None
        self._program: FunctionType = None
        self._input: dict = None
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
    @abstractmethod
    def program(self):
        pass

    @program.setter
    @abstractmethod
    def program(self, _program):
        pass

    @property
    @abstractmethod
    def input(self):
        pass

    @input.setter
    @abstractmethod
    def input(self, _input):
        pass

    @property
    def output_vars(self):
        return self._output_vars

    @output_vars.setter
    def output_vars(self, _output_vars):
        pass

    @abstractmethod
    def get_output(self):
        pass
        return self._output

    def output(self, _output_vars=None):
        return ReferenceNode(self, _output_vars)

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


class QuaNode(ProgramNode, ABC):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None,
                 quantum_machine=None, _simulate_or_execute='simulate'):

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
            # TODO: add support for ReferenceNode unpacking
            qua_program = self.program(**_input)
            assert isinstance(qua_program, qm.program._Program), \
                "TypeError: Try a different program. Expected <qm.program._Program> but given <{}>".format(
                    type(qua_program))
            self._qua_program = qua_program

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
        if self._simulate_or_execute == 'simulate':
            self.simulate(**kwargs)
        if self._simulate_or_execute == 'execute':
            self.execute(**kwargs)

    def execute(self, **kwargs):
        self._job = self._quantum_machine.execute(self._qua_program, **kwargs)

    def simulate(self, sim_config=SimulationConfig(), **kwargs):
        self._job = self._quantum_machine.simulate(self._qua_program, sim_config, **kwargs)


class PyNode(ProgramNode):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None):
        super().__init__(_label, _program, _input, _output_vars)

    @property
    def program(self):
        return self._program

    @program.setter
    def program(self, _program):
        if _program is not None:
            assert type(_program) is FunctionType, \
                "TypeError: Expected <FunctionType> but given <{}>".format(type(_program))
        self._program = _program

    @property
    def input(self):
        pass

    @input.setter
    def input(self, _input):
        pass

    def get_output(self):
        pass
        return self._output

    @property
    def timestamp(self):
        pass

    def run(self):
        pass


class ProgramGraph:

    def __init__(self, _label):
        """
        A program graph describes a program flow with input/output dependencies
        :param _label: a label for the graph
        :type _label: str
        """
        self._id: int = id(self)
        self._label: str = None
        self._nodes: dict = None
        self._node_counter: int = 0
        self._edges: dict = None
        self._backward_edges: dict = None
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

    def add_nodes(self, nodes):
        """
        Adds given nodes to the graph
        :param nodes: list of node objects
        :return:
        """
        # update edges
        # update self._node_counter

    def remove_nodes(self, node_ids):
        """
        Removes the nodes with given ids from the graph
        :param node_ids:
        :return:
        """
        # update edges

    @property
    def edges(self):
        return self._edges

    def add_edges(self, _edges):
        """
        Add edges between nodes with given ids.
        This is used to describe time order rather than input/output dependency.
        :param _edges: list of tuples [(source_node_id, dest_node_id)...]
        :type _edges: list
        :return:
        """
        # need to update backward_edges

    def remove_edges(self, _edges):
        """
        Remove edges from graph
        :param _edges: list of tuples [(source_node_id, dest_node_id)...]
        :type _edges: list
        :return:
        """
        # need to update backward edges

    @property
    def backward_edges(self):
        return self._backward_edges

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def output(self):
        return self._output

    def run(self, start_node_ids=None):
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs accordingly
        :param start_node_ids: list of node ids to start running the graph from
        :type: start_nodes_ids: list
        :return:
        """

    def plot(self, start_node_ids=None):
        """
        Plot starting from the given node and in the direction of propagation
        :param start_node_ids: list of node ids to start plotting from
        :return:
        """

    def merge(self, graph):
        """
        Merge graph into the current graph
        :param graph:
        :return:
        """
        pass
