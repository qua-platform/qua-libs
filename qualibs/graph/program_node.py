from qm import QmJob, QuantumMachine
from qm.program import _Program as QuaProgram

from abc import ABC, abstractmethod
from types import FunctionType
from typing import Dict, Set, Any, Union
from collections.abc import Coroutine
from time import time_ns
from inspect import iscoroutinefunction


class LinkNode:
    def __init__(self, node, output_var: str = None):
        """
        Provides a link between different ProgramNodes
        :param node: source ProgramNode
        :param output_var: the desired output variable of the given ProgramNode
        """
        self.node: ProgramNode = node

        if output_var is not None:

            try:
                assert output_var in self.node.output_vars
            except AssertionError:
                raise Exception(f"Output variables of node <{self.node.label}> "
                                f"don't contain the variable '{output_var}'")

        self.output_var: str = output_var

    def get_output(self):
        if self.output_var is not None:
            return self.node.result[self.output_var]
        else:
            return self.node.result


class ProgramNode(ABC):

    def __init__(self, label: str = None, program: Union[FunctionType, Coroutine] = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None,
                 to_run: bool = True):
        """
        Program node contains a program to run and description of input_vars/output variables
        :param label: label for the node
        :type: label: str
        :param program: a python function to run
        :type program: function
        :param input_vars: input_vars variables names and values
        :type input_vars: dict
        :param output_vars: output variable names
        :type output_vars: Set[str]
        :param to_run: whether to run the node
        :type to_run: bool
        """
        self._id: int = id(self)
        self.label = label
        self.program = program
        self.input_vars = input_vars
        self.output_vars: Set[str] = output_vars
        self.to_run = to_run

        self._result: Dict[str, Any] = dict()
        self._timestamp = None  # when last finished running
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
    def program(self, program):
        if program is not None:
            assert type(program) is FunctionType, \
                "TypeError: Expected FunctionType but given <{}>".format(type(program))
        self._program = program

    @property
    def input_vars(self):
        return self._input_vars

    @input_vars.setter
    def input_vars(self, input_vars):
        if input_vars is not None:
            assert type(input_vars) is dict, \
                "TypeError: Try a different input_vars. Expected <dict> but given <{}>".format(type(input_vars))
            self._input_vars = input_vars
        else:
            self._input_vars = dict()

    @property
    def output_vars(self) -> Set[str]:
        return self._output_vars

    @output_vars.setter
    def output_vars(self, output_vars):
        if output_vars is not None:
            assert type(output_vars) is set, \
                "TypeError: Try a different output_vars. Expected <set> but given <{}>".format(type(output_vars))
            self._output_vars = output_vars
        else:
            self._output_vars = set()

    @property
    def result(self) -> Dict[str, Any]:
        return self._result

    @abstractmethod
    def get_result(self) -> None:
        pass

    def output(self, output_vars=None) -> LinkNode:
        return LinkNode(self, output_vars)

    @property
    def timestamp(self):
        return self._timestamp

    @abstractmethod
    async def run(self) -> None:
        pass

    @property
    def to_run(self) -> bool:
        return self._to_run

    @to_run.setter
    def to_run(self, to_run):
        assert type(to_run) is bool, "TypeError: Expected bool but given {}".format(type(to_run))
        self._to_run = to_run


class QuaJob:
    def __init__(self, node):
        """
        Provides a link between a QuaNode job result handle and another node
        :param node: source ProgramNode
        """
        self.node: QuaNode = node

    @property
    def result_handles(self):
        return self.node._job.result_handles

    @property
    def quantum_machine(self):
        return self.node.quantum_machine


class QuaNode(ProgramNode):

    def __init__(self, label: str = None, program: Union[FunctionType, Coroutine] = None, input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None,
                 quantum_machine: QuantumMachine = None, simulation_kwargs: Dict[str, Any] = None,
                 execution_kwargs: Dict[str, Any] = None, simulate_or_execute: str = None):

        super().__init__(label, program, input_vars, output_vars)

        self.quantum_machine = quantum_machine
        self.execution_kwargs = execution_kwargs
        self.simulation_kwargs = simulation_kwargs
        self.simulate_or_execute: str = simulate_or_execute

        self._type = 'Qua'
        self._job: QmJob.QmJob = None
        self._qua_program: QuaProgram = None

    @property
    def quantum_machine(self) -> QuantumMachine:
        return self._quantum_machine

    @quantum_machine.setter
    def quantum_machine(self, quantum_machine):
        if quantum_machine is not None:
            assert isinstance(quantum_machine, QuantumMachine), \
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
        for var in self.output_vars:
            try:
                # TODO: Make sure it works for all ways of saving data in qua
                self._result[var] = getattr(self._job.result_handles, var).fetch_all()['value']
            except AttributeError:
                raise AttributeError(f"The variable '{var}' isn't in the job result of node <{self.label}>")

    def job(self):
        return QuaJob(self)

    async def run(self) -> None:
        if self.to_run:
            # Get the Qua program that is wrapped by the python function
            if iscoroutinefunction(self.program):
                qua_program = await self.program(**self.input_vars)
            else:
                qua_program = self.program(**self.input_vars)
            assert isinstance(qua_program, QuaProgram), \
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
            self._timestamp = time_ns()
            self.get_result()

    def execute(self) -> None:
        print("\nEXECUTING QuaNode <{}>...".format(self.label))
        self._job = self._quantum_machine.execute(self._qua_program, **self._execution_kwargs)
        print(f"DONE running node <{self.label}>")

    def simulate(self) -> None:
        print("\nSIMULATING QuaNode <{}>...".format(self.label))
        self._job = self._quantum_machine.simulate(self._qua_program, **self._simulation_kwargs)
        print(f"DONE running node <{self.label}>")


class PyNode(ProgramNode):

    def __init__(self, label: str = None, program: Union[FunctionType, Coroutine] = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None):
        super().__init__(label, program, input_vars, output_vars)
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
                print(f"Couldn't fetch '{var}' from results of node <{self.label}>")

    async def run(self):
        if self.to_run:
            print("\nRUNNING PyNode <{}>...".format(self.label))
            if iscoroutinefunction(self.program):
                self._job_results = await self.program(**self.input_vars)
            else:
                self._job_results = self.program(**self.input_vars)
            print(f"DONE running node <{self.label}>")
            assert type(self._job_results) is dict, \
                "TypeError: Expected <dict> but got <{}> as program results.\n" \
                "PyNode program must return a dictionary.".format(type(self._job_results))
            self._timestamp = time_ns()
            self.get_result()
