import qm

from abc import ABC, abstractmethod
from types import FunctionType
from typing import Dict, Set, Any


class LinkNode:
    def __init__(self, node, output_var: str = None):
        """
        Provides a link between different ProgramNodes
        :param node: source ProgramNode
        :param output_var: the desired output variable of the given ProgramNode
        """
        self.node: ProgramNode = node
        if output_var is not None:
            assert output_var in self.node.output_vars, \
                "KeyError: Output of node <{}> doesn't contain the variable <{}>".format(self.node.label, output_var)
        self.output_var: str = output_var


class ProgramNode(ABC):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: Dict[str, Any] = None,
                 _output_vars: Set[str] = None,
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
        self.label = _label
        self.program = _program
        self.input = _input
        self.output_vars = _output_vars
        self.to_run = _to_run

        self._result: Dict[str, Any] = dict()
        self._timestamp = None
        self._type = None

    @property
    def type(self):
        return self._type

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
    def program(self) -> FunctionType:
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
        else:
            self._input = dict()

    @property
    def output_vars(self) -> Set[str]:
        return self._output_vars

    @output_vars.setter
    def output_vars(self, _output_vars):
        if _output_vars is not None:
            assert type(_output_vars) is set, \
                "TypeError: Try a different output_vars. Expected <set> but given <{}>".format(type(_output_vars))
            self._output_vars = _output_vars
        else:
            self._output_vars = set()

    @property
    def result(self) -> Dict[str, Any]:
        return self._result

    @abstractmethod
    def get_result(self) -> None:
        pass

    def output(self, _output_vars=None) -> LinkNode:
        return LinkNode(self, _output_vars)

    @property
    @abstractmethod
    def timestamp(self):
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    @property
    def to_run(self) -> bool:
        return self._to_run

    @to_run.setter
    def to_run(self, to_run):
        assert type(to_run) is bool, "TypeError: Expected bool but given {}".format(type(to_run))
        self._to_run = to_run


class QuaNode(ProgramNode):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: Dict[str, Any] = None,
                 _output_vars: Set[str] = None,
                 _quantum_machine: qm.QuantumMachine = None, _simulation_kwargs: Dict[str, Any] = None,
                 _execution_kwargs: Dict[str, Any] = None, _simulate_or_execute: str = None):

        super().__init__(_label, _program, _input, _output_vars)

        self.quantum_machine = _quantum_machine
        self.execution_kwargs = _execution_kwargs
        self.simulation_kwargs = _simulation_kwargs
        self.simulate_or_execute: str = _simulate_or_execute

        self._type = 'Qua'
        self._job: qm.QmJob.QmJob = None
        self._qua_program: qm.program._Program = None

    @property
    def quantum_machine(self) -> qm.QuantumMachine:
        return self._quantum_machine

    @quantum_machine.setter
    def quantum_machine(self, quantum_machine):
        if quantum_machine is not None:
            assert isinstance(quantum_machine, qm.QuantumMachine), \
                "TypeError: Expected <QuantumMachine> but given {}".format(type(quantum_machine))
        self._quantum_machine = quantum_machine

    @property
    def simulate_or_execute(self) -> str:
        try:
            if self._simulate_or_execute:
                return self._simulate_or_execute
        except AttributeError:
            if self._simulation_kwargs:
                self._simulate_or_execute = 'simulate'
            elif self._execution_kwargs:
                self._simulate_or_execute = 'execute'
            return self._simulate_or_execute

    @simulate_or_execute.setter
    def simulate_or_execute(self, s_or_e):
        if s_or_e is not None:
            assert s_or_e == 'simulate' or s_or_e == 'execute', \
                "ValueError: Expected 'simulate' or 'execute' but got '{}'".format(s_or_e)
            self._simulate_or_execute = s_or_e

    @property
    def execution_kwargs(self) -> dict:
        return self._execution_kwargs

    @execution_kwargs.setter
    def execution_kwargs(self, kwargs):
        if kwargs is not None:
            assert type(kwargs) is dict, \
                "TypeError: Expecting a <dict> of args but got {}.".format(type(kwargs))
        self._execution_kwargs = kwargs

    @property
    def simulation_kwargs(self) -> dict:
        return self._simulation_kwargs

    @simulation_kwargs.setter
    def simulation_kwargs(self, kwargs):
        if kwargs is not None:
            assert type(kwargs) is dict, \
                "TypeError: Expecting a <dict> of args but got {}.".format(type(kwargs))
        self._simulation_kwargs = kwargs

    def get_result(self) -> None:
        assert self.output_vars is not None, \
            "Error: must specify output variables for node <{}>".format(self.label)
        for var in self.output_vars:
            try:
                self._result[var] = getattr(self._job.result_handles, var).fetch_all()['value']
            except AttributeError:
                print("Error: the variable '{}' isn't in the output of node <{}>".format(var, self.label))

    @property
    def timestamp(self):
        pass

    def run(self) -> None:

        # Get the Qua program that is wrapped by the python function
        qua_program = self.program(**self._input)
        assert isinstance(qua_program, qm.program._Program), \
            "In node <id:{},label:{}> TypeError: Expected <qm.program._Program> but given <{}>.\n" \
            "QuaNode program must return a Qua program.".format(self.id, self.label, type(qua_program))
        self._qua_program = qua_program

        assert self.simulate_or_execute is not None, \
            "Error: Either missing parameters or " \
            "didn't specify whether to simulate/execute QuaNode {}".format(self.label)

        if self.simulate_or_execute == 'simulate':
            self.simulate()
        if self.simulate_or_execute == 'execute':
            self.execute()

        self.get_result()

    def execute(self) -> None:
        print("\nEXECUTING QuaNode '{}'...".format(self.label))
        self._job = self._quantum_machine.execute(self._qua_program, **self._execution_kwargs)
        print("DONE")

    def simulate(self) -> None:
        print("\nSIMULATING QuaNode '{}'...".format(self.label))
        self._job = self._quantum_machine.simulate(self._qua_program, **self._simulation_kwargs)
        print("DONE")


class PyNode(ProgramNode):

    def __init__(self, _label: str = None, _program: FunctionType = None, _input: Dict[str, Any] = None,
                 _output_vars: Set[str] = None):
        super().__init__(_label, _program, _input, _output_vars)
        self._job_results = None
        self._type = 'Py'

    def get_result(self):
        if self.output_vars is None:
            print("ATTENTION! No output variables defined for node <{}>".format(self.label))
            return
        for var in self.output_vars:
            try:
                self._result[var] = self._job_results[var]
            except KeyError:
                print("Couldn't fetch '{}' from Qua program results".format(var))

    @property
    def timestamp(self):
        pass

    def run(self):
        print("\nRUNNING PyNode '{}'...".format(self.label))
        self._job_results = self.program(**self.input)
        print("DONE")
        assert type(self._job_results) is dict, \
            "TypeError: Expected <dict> but got <{}> as program results.\n" \
            "PyNode program must return a dictionary.".format(type(self._job_results))
        self.get_result()



