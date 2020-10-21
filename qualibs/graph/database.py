from __future__ import annotations


from .environment import env_resolve
from qualibs.results.impl.sqlalchemy import SqlAlchemyResultsConnector, NodeTypes
from qualibs.results.api import Graph, Node, Result, Metadatum
from qualibs.graph import *

from inspect import isfunction
from types import FunctionType
from typing import List
from colorama import Fore, Style
from copy import deepcopy
from io import BytesIO
import sys


def print_red(skk):
    print(Fore.RED + f"{skk}" + Style.RESET_ALL)


def print_green(skk):
    print(Fore.GREEN + f"{skk}" + Style.RESET_ALL)


def print_yellow(skk):
    print(Fore.YELLOW + f"{skk}" + Style.RESET_ALL)


class GraphDB:
    def __init__(self, results_path: str = ':memory:', global_metadata_funcs=[], envmodule=None):
        """
        Creating a link to a SQLite DB
        :param results_path: store location for DB
        :param results_path: store location for DB
        """
        self.results_path = results_path
        self._dbcon = SqlAlchemyResultsConnector(backend=self._results_path)
        self.global_metadata_funcs = global_metadata_funcs
        self._envmodule = envmodule

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
            if k == '_dbcon':
                setattr(result, k, v)
            else:
                setattr(result, k, deepcopy(v, memo))
        return result

    @property
    def results_path(self):
        return self._results_path

    @results_path.setter
    def results_path(self, results_path):
        if type(results_path) != str:
            raise TypeError(f"Excpected {str} but given {type(results_path)} as results path")
        self._results_path = results_path
        self._dbcon = SqlAlchemyResultsConnector(backend=self._results_path)

    @property
    def global_metadata_funcs(self):
        return self._global_metadata_funcs

    @global_metadata_funcs.setter
    def global_metadata_funcs(self, graph_metadata):
        if graph_metadata:
            if isfunction(graph_metadata):
                self._global_metadata_funcs = [graph_metadata]
            elif type(graph_metadata) is list:
                self._global_metadata_funcs = graph_metadata
            else:
                raise TypeError(f"Metadata parameter must be a {FunctionType} or a {List[FunctionType]}")
        else:
            self._global_metadata_funcs = list()

    def save_graph(self, graph: ProgramGraph, calling_script_path: str):
        try:
            calling_script = open(calling_script_path).read() if calling_script_path else None
        except OSError:
            calling_script = open(graph._calling_script).read()
        self._dbcon.save(Graph(graph_id=graph.id,
                               graph_script=calling_script,
                               graph_name=graph.label,
                               graph_dot_repr=graph.export_dot_graph()))  # TODO: add full graphID, nodeID to dot graph

    def save_node(self, graph: ProgramGraph, node: ProgramNode):
        # save nodes to database
        if NodeTypes[node.type] == NodeTypes.Qua:
            version = str(node.quantum_machine._manager.version())
        elif NodeTypes[node.type] == NodeTypes.Py:
            version = str(sys.version_info)
        elif NodeTypes[node.type] == NodeTypes.Graph:
            version = str(sys.version_info)
        else:
            version = str(sys.version_info)

        self._dbcon.save(Node(graph_id=graph.id,
                              node_id=node.id,
                              node_type=NodeTypes[node.type],
                              version=version,
                              node_name=node.label))
        # TODO: Add 'points_to = graph.edges[node.id]' and 'script=inspect.getsource(node._program)'

    def save_graph_results(self, graph: ProgramGraph):
        for node_id, node in graph.nodes.items():
            for name in node.result.keys():
                self._dbcon.save(Result(graph_id=graph.id, node_id=node_id,
                                        start_time=node._start_time,
                                        end_time=node._end_time,
                                        user_id='User',
                                        name=name,
                                        val=str(node.result[name])
                                        ))
                if node.type == 'Qua':
                    res = node._job.result_handles
                    npz_store = BytesIO()
                    res.save_to_store(writer=npz_store)
                    self._dbcon.save(Result(graph_id=graph.id, node_id=node_id,
                                            start_time=node._start_time,
                                            end_time=node._end_time,
                                            user_id='User',
                                            name='npz',
                                            val=npz_store.getvalue()
                                            ))

    def save_metadata(self, graph: ProgramGraph, node: ProgramNode):
        for fn in node.metadata_funcs:
            metadata = env_resolve(fn, self._envmodule)()
            for key, val in metadata.items():
                self._dbcon.save(Metadatum(graph_id=graph.id, node_id=node.id, name=key, val=str(val)))
