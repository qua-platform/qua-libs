from __future__ import annotations

from qm import QmJob, QuantumMachine
from qm.program import _Program as QuaProgram

from abc import ABC, abstractmethod
from types import FunctionType
from typing import Dict, Set, Any, Union, List, Optional
from collections.abc import Coroutine
from datetime import datetime
from colorama import Fore, Style
from inspect import iscoroutinefunction, isfunction
from copy import deepcopy
import asyncio


def print_red(skk): print(Fore.RED + f"{skk}" + Style.RESET_ALL)


def print_green(skk): print(Fore.GREEN + f"{skk}" + Style.RESET_ALL)


def print_yellow(skk): print(Fore.YELLOW + f"{skk}" + Style.RESET_ALL)


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
                raise ValueError(f"Output variables of node <{self.node.label}> "
                                 f"do not contain the variable '{output_var}'")
        self.output_var: str = output_var

    def get_output(self):
        """
        In case of a specified output variable returns it's value, otherwise returns the full result
        :return:
        """
        if self.output_var is not None:
            try:
                return self.node.result[self.output_var]
            except KeyError:
                raise KeyError(f"The variable '{self.output_var}' is not in the result of node <{self.node.label}>")
        else:
            return self.node.result

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result

    def __eq__(self, other: LinkNode):
        if not self.output_var == other.output_var:
            return False
        if not self.node.output_vars == other.node.output_vars:
            return False
        if not self.node.label == other.node.label:
            return False
        if not self.node.program == other.node.program:
            return False
        if not self.node.input_vars == other.node.input_vars:
            return False
        if not self.node.result == other.node.result:
            return False
        return True

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class InputVars:
    def __init__(self, input_vars):
        if input_vars:
            for var, val in input_vars.items():
                setattr(self, var, val)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, item):
        return getattr(self, item)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return ((k, v) for k, v in self.__dict__.items())

    def __eq__(self, other):
        for k, v in self.__iter__():
            if not type(other[k]) == type(v):
                return False
            else:
                if not v == other[k]:
                    return False
        return True

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    #
    # def copy(self):
    #     """
    #     Implements  shallow copy - copy by reference
    #     :return:
    #     """
    #     return self.__copy__()

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def deepcopy(self):
        """
        Implements deep copy - copy by value
        :return:
        """
        return self.__deepcopy__()


class ProgramNode(ABC):

    def __init__(self, label: str = None,
                 program: Union[FunctionType, Coroutine] = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None,
                 metadata_funcs: Union[FunctionType, List[FunctionType]] = None,
                 to_run: bool = True
                 ):

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
        :param metadata_funcs: a list of functions or a single function that describes the node's metadata
        :param to_run: whether to run the node
        :type to_run: bool
        """
        self._id = id(self)
        self.label = label
        self.program = program
        self.input_vars = input_vars
        self.output_vars: Set[str] = output_vars
        self.to_run = to_run
        self.metadata_funcs = metadata_funcs
        self._result: Dict[str, Any] = dict()
        self._start_time = None  # last time started running
        self._end_time = None  # last time when finished running
        self._type = None
        self.save_result_to_db = True

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    # def copy(self):
    #     """
    #     Implements  shallow copy - copy by reference, provides new id
    #     :return:
    #     """
    #     self_copy = self.__copy__()
    #     self_copy._id = id(self_copy)
    #     return self_copy

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_id':
                setattr(result, k, id(result))
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    def deepcopy(self):
        """
        Implements deep copy - copy by value, provides new id
        :return:
        """
        return self.__deepcopy__()

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
        if _label is not None:
            if type(_label) is not str:
                raise TypeError(f"In node <{self.label}> expected {str} but given {type(_label)}")
        self._label = _label

    @property
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def program(self) -> FunctionType:
        return self._program

    @program.setter
    def program(self, program):
        if program is not None:
            if not isfunction(program):
                raise TypeError(f"In node <{self.label}> expected {FunctionType} but given {type(program)}")
        self._program = program

    @property
    def input_vars(self):
        return self._input_vars

    @input_vars.setter
    def input_vars(self, input_vars):
        if type(input_vars) is dict or input_vars is None:
            self._input_vars = InputVars(input_vars)
        else:
            raise TypeError(f"In node <{self.label}> expected {dict} but given {type(input_vars)}")

    @property
    def output_vars(self) -> Set[str]:
        return self._output_vars

    @output_vars.setter
    def output_vars(self, output_vars):
        if output_vars is not None:
            if type(output_vars) is not set:
                raise TypeError(f"In node <{self.label}> expected {set} but given {type(output_vars)}")

            self._output_vars = output_vars
        else:
            self._output_vars = set()

    @property
    def result(self) -> Dict[str, Any]:
        return self._result

    @abstractmethod
    def _fetch_result(self) -> None:
        pass

    def output(self, output_vars=None) -> LinkNode:
        return LinkNode(self, output_vars)

    @abstractmethod
    async def run_async(self) -> None:
        pass

    def run(self) -> None:
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self.run_async())
        except RuntimeError:
            asyncio.run(self.run_async())

    @property
    def to_run(self) -> bool:
        return self._to_run

    @to_run.setter
    def to_run(self, to_run):
        if type(to_run) is not bool:
            raise TypeError(f"In node <{self.label}> expected {bool} but given {type(to_run)}")
        self._to_run = to_run

    @property
    def metadata_funcs(self):
        return self._metadata_funcs

    @metadata_funcs.setter
    def metadata_funcs(self, node_metadata):
        if node_metadata:
            if isfunction(node_metadata):
                self._metadata_funcs = [node_metadata]
            elif type(node_metadata) is list:
                self._metadata_funcs = node_metadata
            else:
                raise TypeError(f"Metadata parameter must be a {FunctionType} or a {List[FunctionType]}")
        else:
            self._metadata_funcs = list()

    @property
    def save_result_to_db(self):
        return self._save_result_to_db

    @save_result_to_db.setter
    def save_result_to_db(self, t):
        if not type(t) is bool:
            raise TypeError(f"Must be of type {bool}")
        self._save_result_to_db = t


class QuaJobNode:
    def __init__(self, node):
        """
        Provides a link between a QuaNode job result handle and another node
        :param node: source ProgramNode
        """
        self.node: QuaNode = node

    @property
    def job(self):
        return self.node.job

    def resume(self):
        self.job.resume()

    def wait(self, timeout: Optional[int] = None):
        """
        Wait until the job is finished and all the results are available for retrieving
        :argument timeout: The maximum time to wait in seconds
        :type timeout: int
        """
        self.job.wait_for_all_results(timeout)

    def stop(self):
        """
        Stops the job
        :return:
        """
        self.job.halt()

    @property
    def result_handles(self):
        return self.node.job.result_handles

    def get_values(self, var):
        return getattr(self.result_handles, var).fetch_all()['value']

    @property
    def quantum_machine(self):
        return self.node.quantum_machine

    @property
    def IO1(self):
        return self.quantum_machine.get_io1_value()

    @IO1.setter
    def IO1(self, val):
        """
                Sets the value of ``IO1``. Can be used inside qua as IO1 without decleration
                :param val: the value to be placed in ``IO1``
                :type val: float | bool | int
        """
        self.quantum_machine.set_io1_value(val)

    @property
    def IO2(self):
        return self.quantum_machine.get_io2_value()

    @IO2.setter
    def IO2(self, val):
        """
                Sets the value of ``IO2``. Can be used inside qua as IO1 without decleration
                :param val: the value to be placed in ``IO2``
                :type val: float | bool | int
        """
        self.quantum_machine.set_io2_value(val)

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, v)
        return result

    def __eq__(self, other: QuaJobNode):
        if not self.quantum_machine == other.quantum_machine:
            return False
        if not self.node.output_vars == other.node.output_vars:
            return False
        if not self.node.label == other.node.label:
            return False
        if not self.node.program == other.node.program:
            return False
        if not self.node.input_vars == other.node.input_vars:
            return False
        if not self.node.result == other.node.result:
            return False

        return True

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()


class QuaNode(ProgramNode):

    def __init__(self, label: str = None,
                 program: Union[FunctionType, Coroutine] = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None,
                 quantum_machine: QuantumMachine = None,
                 simulation_kwargs: Dict[str, Any] = None,
                 execution_kwargs: Dict[str, Any] = None,
                 simulate_or_execute: str = None,
                 metadata_funcs: Union[FunctionType, List[FunctionType]] = None):

        super().__init__(label, program, input_vars, output_vars, metadata_funcs)

        self.quantum_machine = quantum_machine
        self.execution_kwargs = execution_kwargs
        self.simulation_kwargs = simulation_kwargs
        self.simulate_or_execute: str = simulate_or_execute
        self._type = 'Qua'
        self._job: QmJob.QmJob = None
        self._qua_program: QuaProgram = None

    def __deepcopy__(self, memo=None):
        if memo is None:
            memo = dict()
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_id':
                setattr(result, k, id(result))
            elif k == '_quantum_machine':
                setattr(result, k, v)
            elif k == '_job':
                setattr(result, k, None)
                print_yellow(f"ATTENTION the QuaJob of node <{self.label}> was NOT copied")
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def quantum_machine(self) -> QuantumMachine:
        return self._quantum_machine

    @quantum_machine.setter
    def quantum_machine(self, quantum_machine):
        if quantum_machine is not None:
            if not isinstance(quantum_machine, QuantumMachine):
                raise TypeError(f"In node <{self.label}> expected {QuantumMachine} but given {type(quantum_machine)}")
        self._quantum_machine = quantum_machine

    @property
    def simulate_or_execute(self) -> str:
        if hasattr(self, '_simulate_or_execute'):
            return self._simulate_or_execute
        else:
            if self._simulation_kwargs:
                self._simulate_or_execute = 'simulate'
            elif self._execution_kwargs:
                self._simulate_or_execute = 'execute'
            return self._simulate_or_execute

    @simulate_or_execute.setter
    def simulate_or_execute(self, s_or_e):
        if s_or_e is not None:
            if s_or_e != 'simulate' and s_or_e != 'execute':
                raise ValueError(f"In node <{self.label}> expected 'simulate' or 'execute' but got {s_or_e}")
            self._simulate_or_execute = s_or_e

    @property
    def execution_kwargs(self) -> dict:
        return self._execution_kwargs

    @execution_kwargs.setter
    def execution_kwargs(self, kwargs):
        if kwargs is not None:
            if type(kwargs) is not dict:
                raise TypeError(f"In node <{self.label}> expected {dict} but got {type(kwargs)}")
        self._execution_kwargs = kwargs

    @property
    def simulation_kwargs(self) -> dict:
        return self._simulation_kwargs

    @simulation_kwargs.setter
    def simulation_kwargs(self, kwargs):
        if kwargs is not None:
            if type(kwargs) is not dict:
                raise TypeError(f"In node <{self.label}> expected {dict} but got {type(kwargs)}")
        self._simulation_kwargs = kwargs

    @property
    def job(self):
        return self._job

    def _fetch_result(self) -> None:
        if self.output_vars is None:
            print_yellow(f"ATTENTION No output variables defined for node <{self.label}>")
            return
        for var in self.output_vars:
            try:
                # TODO: Make sure it works for all ways of saving data in qua
                self._result[var] = getattr(self._job.result_handles, var).fetch_all()['value']
            except AttributeError:
                print_red(f"WARNING Could not fetch variable '{var}' from the job results of node <{self.label}>")

    def qua_job(self) -> QuaJobNode:
        return QuaJobNode(self)

    async def run_async(self) -> None:
        if self.to_run:
            # Get the Qua program that is wrapped by the python function
            if iscoroutinefunction(self.program):
                qua_program = await self.program(**self.input_vars.__dict__)
            else:
                qua_program = self.program(**self.input_vars.__dict__)

            if not isinstance(qua_program, QuaProgram):
                raise TypeError(f"Program given to node <{self.label}> must return a "
                                f"Qua program as {QuaProgram} but returns {type(qua_program)}")

            self._qua_program = qua_program

            if self.simulate_or_execute is None:
                raise ValueError(f"Must specify simulation/execution parameters for QuaNode <{self.label}>")

            self._start_time = datetime.now()
            if self.simulate_or_execute == 'simulate':
                self._simulate()
            if self.simulate_or_execute == 'execute':
                self._execute()
            self._end_time = datetime.now()
            self._fetch_result()

    def _execute(self) -> None:
        print("\nEXECUTING QuaNode <{}>...".format(self.label))
        self._job = self._quantum_machine.execute(self._qua_program, **self._execution_kwargs)
        print_green(f"DONE running node <{self.label}>")

    def _simulate(self) -> None:
        print("\nSIMULATING QuaNode <{}>...".format(self.label))
        self._job = self._quantum_machine.simulate(self._qua_program, **self._simulation_kwargs)
        print_green(f"DONE running node <{self.label}>")


class PyNode(ProgramNode):

    def __init__(self, label: str = None,
                 program: Union[FunctionType, Coroutine] = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None,
                 metadata_funcs: Union[FunctionType, List[FunctionType]] = None):

        super().__init__(label, program, input_vars, output_vars, metadata_funcs)
        self._job_results = None
        self._type = 'Py'

    def _fetch_result(self):
        if self.output_vars is None:
            print_yellow(f"ATTENTION No output variables defined for node <{self.label}>")
            return
        for var in self.output_vars:
            try:
                self._result[var] = self._job_results[var]
            except KeyError:
                print_red(f"WARNING Could not fetch variable '{var}' from the results of node <{self.label}>")

    async def run_async(self) -> None:
        if self.to_run:
            self._start_time = datetime.now()

            print("\nRUNNING PyNode <{}>...".format(self.label))
            if iscoroutinefunction(self.program):
                self._job_results = await self.program(**self.input_vars.__dict__)
            else:
                self._job_results = self.program(**self.input_vars.__dict__)
            print_green(f"DONE running node <{self.label}>")

            self._end_time = datetime.now()

            if self._job_results:
                if type(self._job_results) is not dict:
                    raise TypeError(f"In node <{self.label}> expected {dict} but got <{type(self._job_results)}> "
                                    f"as the result")
                self._fetch_result()
