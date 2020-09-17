from abc import ABC, abstractmethod
from qm import QuantumMachine


class ProgramNode(ABC):

    def __init__(self, _id, _label=None, _program=None, _input=None, _to_run=True):
        self._id = _id
        self._label = None
        self._program = None
        self._input = None
        self._to_run = None
        self._output = None

        self.label = _label
        self.program = _program
        self.input = _input
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
    @abstractmethod
    def output(self):
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


class QuaNode(ProgramNode):

    def __init__(self, _id, _label=None, _program=None, _input=None, _quantum_machine=None):
        super().__init__(_id, _label, _program, _input)
        self._quantum_machine = None
        self.quantum_machine = _quantum_machine

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

    @property
    def output(self):
        pass

    def run(self):
        pass


class PyNode(ProgramNode):

    def __init__(self, _id, _label=None, _program=None, _input=None):
        super().__init__(_id, _label, _program, _input)

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

    @property
    def output(self):
        pass

    def run(self):
        pass
