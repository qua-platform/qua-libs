from abc import ABC, abstractmethod
from qm import QuantumMachine


class ReferenceNode:
    def __init__(self, node_id, output_vars):
        self.node_id = node_id
        self.output_vars = output_vars


class ProgramNode(ABC):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None, _to_run=True):
        """

        :param _label: label of the node
        :type: _label: str
        :param _program: a function to run
        :type _program: function
        :param _input: input variable names and values
        :type _input: dict
        :param _output_vars: output variable names
        :type _output_vars: set
        :param _to_run: whether to run the node
        :type _to_run: bool
        """
        self._id = None
        self._label = None
        self._program = None
        self._input = None
        self._to_run = None
        self._output_vars = None
        self._output = None

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

    def output(self, _output_vars):
        return ReferenceNode(self.id, _output_vars)

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


class QuaNode(ProgramNode):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None, quantum_machine=None):
        super().__init__(_label, _program, _input, _output_vars)
        self._quantum_machine = None
        self.quantum_machine = quantum_machine

    @property
    def quantum_machine(self):
        return self._quantum_machine

    @quantum_machine.setter
    def quantum_machine(self, quantum_machine):
        if quantum_machine is not None:
            assert isinstance(quantum_machine, QuantumMachine), \
                "TypeError: Expected QuantumMachine but given {}".format(type(quantum_machine))
        self._quantum_machine = quantum_machine

    @property
    def program(self):
        pass

    @program.setter
    def program(self, _program):
        pass

    @property
    def input(self):
        pass

    @input.setter
    def input(self, _input):
        pass

    def run(self):
        pass


class PyNode(ProgramNode):

    def __init__(self, _label=None, _program=None, _input=None, _output_vars=None):
        super().__init__(_label, _program, _input, _output_vars)

    @property
    def program(self):
        pass

    @program.setter
    def program(self, _program):
        pass

    @property
    def input(self):
        pass

    @input.setter
    def input(self, _input):
        pass

    def run(self):
        pass


class ProgramGraph:

    def __init__(self, _label):
        self._id = id(self)
        self._label = None
        self._nodes = None
        self._node_counter = 0
        self._edges = None
        self._backward_edges = None

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
        # update self._node_counter

    def remove_nodes(self, node_ids):
        """
        Removes the nodes with given ids from the graph
        :param node_ids:
        :return:
        """

    @property
    def edges(self):
        return self._edges

    def add_edges(self, _edges):
        """
        Add edges between given node ids
        :param _edges: list of tuples [(source_node_id, dest_node_id)...]
        :return:
        """
        # need to update backward_edges

    def remove_edges(self, _edges):
        """
        Remove edges from graph
        :param _edges: list of tuples [(source_node_id, dest_node_id)...]
        :return:
        """
        # need to update backward edges

    @property
    def backward_edges(self):
        return self._backward_edges

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
