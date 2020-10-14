
import functools
import sys
from io import BytesIO

from .environment import env_resolve

from .program_node import LinkNode, ProgramNode, QuaJobNode
from qualibs.results.api import *
from qualibs.results.impl.sqlalchemy import SqlAlchemyResultsConnector, NodeTypes

from typing import Dict, Set, List, Tuple, Any
from copy import deepcopy
from time import time_ns
from datetime import datetime
from colorama import Fore, Style
from inspect import stack
from io import BytesIO

import asyncio
import sys


def print_red(skk): print(Fore.RED + f"{skk}" + Style.RESET_ALL)


def print_green(skk): print(Fore.GREEN + f"{skk}" + Style.RESET_ALL)


def print_yellow(skk): print(Fore.YELLOW + f"{skk}" + Style.RESET_ALL)


class GraphDB:
    def __init__(self, results_path: str = ':memory:', env_dependency_list=[], envmodule=None):
        """
        Creating a link to a SQLite DB
        :param results_path: store location for DB
        :param results_path: store location for DB
        """
        self.results_path = results_path
        self._dbcon = SqlAlchemyResultsConnector(backend=self._results_path)
        self._env_dependency_list = env_dependency_list
        self._envmodule = envmodule

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def __deepcopy__(self, memo=dict()):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            if k == '_dbcon':
                setattr(result, k, v)
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

    def save_graph(self, graph, calling_script_path):
        try:
            calling_script = open(calling_script_path).read() if calling_script_path else None
        except OSError:
            calling_script = open(graph._calling_script).read()
        self._dbcon.save(Graph(graph_id=graph.id,
                               graph_script=calling_script,
                               graph_name=graph.label,
                               graph_dot_repr=graph.export_dot_graph()))  # TODO: add full graphID, nodeID to dot graph
        # save nodes to database
        for node_id, node in graph.nodes.items():
            if NodeTypes[node.type] == NodeTypes.Qua:
                version = str(node.quantum_machine._manager.version())
            elif NodeTypes[node.type] == NodeTypes.Py:
                version = str(sys.version_info)

            self._dbcon.save(Node(graph_id=graph.id,
                                  node_id=node_id,
                                  node_type=NodeTypes[node.type],
                                  version=version,
                                  node_name=node.label))

    def save_graph_results(self, graph):
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


    def save_metadata(self, graph,node, node_id):
        metadata = {dep.__name__: env_resolve(dep, self._envmodule)() for dep in node.dependencies}
        for key, val in metadata.items():
            self._dbcon.save(Metadatum(graph_id=graph.id, node_id=node_id, name=key, val=val))


class ProgramGraph:

    def __init__(self, label: str = None, graph_db: GraphDB = None) -> None:
        """
        A program graph describes a program flow with input_vars/output dependencies
        :param label: a label for the graph
        :type label: str
        :param graph_db: a GraphDB to specify an optional DB to save graph related data
        :type graph_db: GraphDB
        """
        self._id: int = time_ns()
        self.label: str = label
        self._nodes: Dict[int, ProgramNode] = dict()
        self._nodes_by_label: Dict[str, Set[ProgramNode]] = dict()
        self._edges: Dict[int, Set[int]] = dict()
        self._backward_edges: Dict[int, Set[int]] = dict()
        self._start_time = None  # last time started running
        self._end_time = None  # last time when finished running
        self._link_nodes: Dict[
            int, Dict[str, Union[LinkNode, QuaJobNode]]] = dict()  # Dict[node_id,Dict[input_var_name,LinkNode]]
        self._link_nodes_ids: Dict[int, Dict[int, List[str]]] = dict()  # Dict[node_id,Dict[out_node_id,out_vars_list]]
        self._tasks = dict()
        self._calling_script = stack()[1][0].f_code.co_filename
        self.graph_db = graph_db

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return self.__str__()

    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result

    def copy(self):
        """
        Implements  shallow copy - copy by reference, provides new id
        :return:
        """
        self_copy = self.__copy__()
        self_copy._id = id(self_copy)
        return self_copy

    def __deepcopy__(self, memo=dict()):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result

    def deepcopy(self):
        """
        Implements deep copy - copy by value, provides new id
        :return:
        """
        self_copy = self.__deepcopy__()
        self_copy._id = time_ns()
        return self_copy

    @property
    def id(self) -> int:
        return self._id

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, _label):
        if _label is not None:
            if type(_label) is not str:
                raise TypeError(f"Expected {str} but given {type(_label)}")
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
            if self._graph_db:
                node.dependencies = list(set(node.dependencies + self._graph_db._env_dependency_list))

            self._nodes[node.id] = node
            self._nodes_by_label.setdefault(node.label, set()).add(node)
            if node.input_vars is not dict():
                for var, value in node.input_vars.__dict__.items():
                    if isinstance(value, LinkNode):
                        self.add_edges({(value.node, node)})
                        self._link_nodes.setdefault(node.id, dict())[var] = value
                        node_input_ids = self._link_nodes_ids.setdefault(node.id, {value.node.id: list()})
                        node_input_ids.setdefault(value.node.id, list()).append(value.output_var)
                    if isinstance(value, QuaJobNode):
                        self.add_edges({(value.node, node)})
                        self._link_nodes.setdefault(node.id, dict())[var] = value
                        node_input_ids = self._link_nodes_ids.setdefault(node.id, {value.node.id: list()})
                        node_input_ids.setdefault(value.node.id, list()).append('!Qua-Job')
        print_green(f"SUCCESS added nodes {[n.label for n in new_nodes]} to the graph")

    def remove_nodes(self, nodes_to_remove: Set[ProgramNode]):
        """
        Removes the given nodes from the graph
        :param nodes_to_remove: Set of nodes to remove
        :type nodes_to_remove: Set[ProgramNode]
        :return:
        """
        edges_to_remove: Set[Tuple[ProgramNode, ProgramNode]] = set()
        for node in nodes_to_remove:
            try:
                # remove node
                self._nodes.pop(node.id)
                self._nodes_by_label.get(node.label, {node}).remove(node)
                if not self._nodes_by_label[node.label]:
                    del self._nodes_by_label[node.label]

                # remove forward edges
                try:
                    ids_to_remove = self.edges[node.id]
                    for dest_node_id in ids_to_remove:
                        edges_to_remove.add((node, self.nodes[dest_node_id]))
                except KeyError:
                    print_yellow(f'Node <{node.label}> has no outgoing edges.')

                # remove backward edges
                try:
                    ids_to_remove = self.backward_edges[node.id]
                    for source_node_id in ids_to_remove:
                        edges_to_remove.add((self.nodes[source_node_id], node))
                except KeyError:
                    print_yellow(f'Node <{node.label}> has no incoming edges.')

                print_green(f"SUCCESS removed node <{node.label}>")
            except KeyError:
                print_yellow(f"ATTENTION Tried to remove node <{node.label}> but it was not found in the graph")

        self.remove_edges(edges_to_remove)

    @property
    def edges(self):
        return self._edges

    def add_edges(self, edges: Set[Tuple[ProgramNode, ProgramNode]]):
        """
        Add edges between given nodes.
        When used outside of add_nodes method,
        it describes either time order rather than input_vars/output dependency as usual.
        :param edges: set of tuples {(source_node, dest_node)...}
        :type edges: Set[Tuple[ProgramNode, ProgramNode]]
        :return:
        """
        for source, dest in edges:
            if source.id not in self.nodes:
                print_red(f"WARNING tried to add edge between <{source.label}> and <{dest.label}>, "
                          f"but <{source.label}> was not added to the graph "
                          f"(maybe a copy of the node was added instead)")
            if dest.id not in self.nodes:
                print_red(f"WARNING tried to add edge between <{source.label}> and <{dest.label}>, "
                          f"but <{dest.label}> was not added to the graph "
                          f"(maybe a copy of the node was added instead)")
            self._edges.setdefault(source.id, set()).add(dest.id)
            self._backward_edges.setdefault(dest.id, set()).add(source.id)

    def remove_edges(self, edges: Set[Tuple[ProgramNode, ProgramNode]]):
        """
        Remove edges between pairs of nodes
        :param edges: Pairs of nodes {(source_node, dest_node)...}
        :type edges: Set[Tuple[ProgramNode, ProgramNode]]
        :return:
        """
        # need to update backward edges
        for source, dest in edges:
            try:
                self._edges.get(source.id, set()).remove(dest.id)
                self._backward_edges.get(dest.id, set()).remove(source.id)
                if not self._edges[source.id]:
                    del self._edges[source.id]
                if not self._backward_edges[dest.id]:
                    del self._backward_edges[dest.id]

                print_green(f"SUCCESS removed edge from <{source.label}> to <{dest.label}>")
            except KeyError:
                print_yellow(f"ATTENTION Tried to remove edge from <{source.label}> to <{dest.label}> "
                             f"but it doesn't exist.")

    @property
    def backward_edges(self):
        return self._backward_edges

    @property
    def graph_db(self):
        return self._graph_db

    @graph_db.setter
    def graph_db(self, graph_db):
        if (graph_db is not None) and (not isinstance(graph_db, GraphDB)):
            raise TypeError(f"graph_db must be of type {GraphDB}")
        self._graph_db = graph_db

    async def run_async(self,
                        graph_db: GraphDB = None,
                        start_nodes: List[ProgramNode] = None,
                        _calling_script_path: str = None
                        ) -> GraphDB:
        """
        Run the nodes in the graph in a BFS traversal order, while waiting for dependent tasks to complete
        :param graph_db:
        :param _calling_script_path:
        :param start_nodes:
        :return:
        """
        self._id = time_ns()  # update graph id every run

        if (graph_db is not None) and (not isinstance(graph_db, GraphDB)):
            raise TypeError(f"graph_db must be of type {GraphDB}")
        if (start_nodes is not None) and (type(start_nodes) != list) and (type(start_nodes) != set):
            raise TypeError(f"start_nodes must be of type {list} or {set} containing ProgramNode "
                            f"but given {type(start_nodes)}")

        if _calling_script_path is None:
            # get the script of the file that called self.run_async()
            _calling_script_path = stack()[1][0].f_code.co_filename

        graph_db = graph_db if graph_db else self.graph_db
        if graph_db:
            # Save graph to DB
            graph_db.save_graph(self, _calling_script_path)

        # the starting point of the run
        if not start_nodes:
            start_nodes = list()
            for node_id in self.nodes:
                if node_id not in self.backward_edges:
                    start_nodes.append(node_id)
        else:
            start_nodes = [n.id for n in start_nodes]

        self._start_time = datetime.now()
        self._tasks = dict()
        for node_id in self._get_next(start_nodes):
            if node_id not in self._tasks:
                if self._dependencies_started(node_id):  # or node_id in start_nodes:  # TODO: make sure it works

                    # wait for dependencies to complete
                    # if node_id not in start_nodes:
                    await asyncio.gather(*{self._tasks[t] for t in (self.backward_edges.get(node_id, set()))})

                    # direct the output of the dependencies input the input of the node
                    input_vars: Dict[str, Union[LinkNode, QuaJobNode]] = self._link_nodes.get(node_id, set())
                    for var in input_vars:
                        if isinstance(input_vars[var], LinkNode):
                            link_node = input_vars[var]
                            try:
                                self.nodes[link_node.node.id]
                            except KeyError:
                                raise RuntimeError(f"Tried to use the output of node <{link_node.node.label}> "
                                                   f"but the node is not in the graph.")

                            setattr(self.nodes[node_id].input_vars, var, link_node.get_output())

                    # SAVE METADATA TO DB HERE graphdb.metadata.save(node_id)
                    if graph_db:
                        graph_db.save_metadata(self,self.nodes[node_id], node_id)
                    # metadat={dep.__name__: env_resolve(dep, self.graph_db._envmodule)() for dep in self.graph_db._env_dependency_list}
                    # for key in metadat.keys():
                    #     Metadatum(graph_id=
                    # create task to run the node and start running
                    self._tasks[node_id] = asyncio.create_task(self.nodes[node_id].run_async())

        # wait for all tasks(nodes) to complete
        await asyncio.gather(*self._tasks.values())
        self._end_time = datetime.now()

        if graph_db:
            # SAVE GRAPH RES TO DB HERE
            graph_db.save_graph_results(self)

        return graph_db

    def run(self,
            graph_db: GraphDB = None,
            start_nodes: Union[List[ProgramNode], Set[ProgramNode]] = None,
            ) -> GraphDB:
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs.
        :param start_nodes: list of nodes to start running the graph from
        :type: start_nodes: Union[List[ProgramNode], Set[ProgramNode]]
        :param graph_db: a GraphDB instance that provides a connection to a DB
        :return:
        """
        if (graph_db is not None) and (not isinstance(graph_db, GraphDB)):
            raise TypeError(f"graph_db must be of type {GraphDB}")
        if (start_nodes is not None) and (type(start_nodes) != list) and (type(start_nodes) != set):
            raise TypeError(f"start_nodes must be of type {list} or {set} containing ProgramNode "
                            f"but given {type(start_nodes)}")

        # get the script of the file that called self.run()
        calling_script_path = stack()[1][0].f_code.co_filename
        return asyncio.run(self.run_async(graph_db, start_nodes, calling_script_path))

    def _dependencies_started(self, node_id) -> bool:
        return self._backward_edges.get(node_id, set()) <= self._tasks.keys()

    def _dependencies_done(self, node_id) -> bool:
        for depend_id in self.backward_edges.get(node_id, set()):
            try:
                if self._tasks[depend_id].done():
                    continue
                else:
                    return False
            except KeyError:
                return False
        return True

    def _get_next(self, start_nodes: Union[List[int], Set[int]], try_again: List = list()):
        """
        Generator of graph nodes - implementing BFS
        :param try_again:
        :param start_nodes: the start positions of graph traversal
        :type start_nodes: List/Set of node ids
        :return:
        """
        to_do = start_nodes.copy()
        while to_do:
            s = to_do.pop(0)
            yield s
            try:
                for child in self.edges[s]:
                    to_do.append(child)
            except KeyError:
                pass

    def _topological_sort(self, start_nodes: List[ProgramNode] = list()) -> List[int]:
        """
        Returns a list of graph node ids in a topological order. Starting from given start nodes.
        Implements Kahn's algorithm.
        :param start_nodes: list of nodes to start the sort from
        :type start_nodes: List[ProgramNode]
        :return:
        """
        edges: Dict[int, Set[int]] = deepcopy(self.edges)
        backward_edges: Dict[int, Set[int]] = deepcopy(self.backward_edges)

        s: List[int]  # list of node ids with no incoming edges
        if not start_nodes:
            s = [n for n in self.nodes if n not in backward_edges]
            assert s != [], "Graph is cyclic! All nodes depend on other nodes, try changing dependencies."
        else:
            s = [n.id for n in start_nodes]

        sorted_list: List[int] = []  # list that will contain the topologically sorted node ids

        while s:
            n = s.pop(0)
            sorted_list.append(n)

            n_edges = edges.get(n, set()).copy()
            for m in n_edges:
                edges.get(n, set()).remove(m)
                if edges[n] == set():
                    del edges[n]

                backward_edges.get(m, set()).remove(n)
                if backward_edges[m] == set():
                    del backward_edges[m]
                    s.append(m)

        # TODO: currently works only for starting from non-dependent nodes
        assert edges.keys() == set(), \
            "Error: Graph is cyclic! Try changing dependencies."
        # If graph has edges containing the supposedly sorted nodes, then there's a cycle.

        return sorted_list

    def export_dot_graph(self, use_labels=True):
        """
        Converts the graph into DOT graph format
        :param use_labels:
        :return: str
        """
        dot_graph = 'digraph {} {{'.format(self.label)

        dot_graph += '{'
        for node_id in self.nodes:
            if use_labels:
                dot_graph += '{} [shape={}];' \
                    .format(self.nodes[node_id].label, 'ellipse' if self.nodes[node_id].type == 'Qua' else 'box')
            else:
                dot_graph += '{} [shape={}];' \
                    .format(node_id, 'ellipse' if self.nodes[node_id].type == 'Qua' else 'box')
        dot_graph += '};'

        for node_id in self.nodes:
            outgoing_edges = self.edges.get(node_id, None)
            if outgoing_edges is not None:
                for dest_id in outgoing_edges:
                    if use_labels:
                        dot_graph += f'"{self.nodes[node_id].label}" -> "{self.nodes[dest_id].label}"'
                    else:
                        dot_graph += f'"{node_id}" -> "{dest_id}"'

                    var_name = self._link_nodes_ids.get(dest_id, dict()).get(node_id, -1)
                    if var_name is None:
                        var_name = '!all'
                    if var_name == -1:
                        var_name = '!none'

                    dot_graph += f' [label="{var_name}"]'
                    dot_graph += ';'
            else:
                dot_graph += f'"{self.nodes[node_id].label}";'

        dot_graph += '}'

        return dot_graph


class GraphNode(ProgramNode):

    def __init__(self, label: str = None,
                 graph: ProgramGraph = None,
                 input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None):

        super().__init__(label, None, input_vars, output_vars)
        self.graph: ProgramGraph = graph
        self._type = 'Graph'
        self._job = None

    def _get_result(self):
        if self.output_vars is None:
            print_yellow(f"ATTENTION No output variables were defined for node <{self.label}>")
            return
        for var in self.output_vars:
            try:
                pass
                # self._result[var] = self.graph.result[var]
            except KeyError:
                print_red(f"WARNING Could not fetch '{var}' from node <{self.label}> results")

    async def run_async(self) -> None:
        if self.to_run:
            self._start_time = datetime.now()
            print("\nRUNNING GraphNode '{}'...".format(self.label))
            self._job = await asyncio.create_task(self.graph.run_async())
            print("DONE")
            self._end_time = datetime.now()
            self._get_result()
