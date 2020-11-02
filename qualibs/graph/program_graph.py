from __future__ import annotations

from qualibs.graph import *
from .database import GraphDB
from .program_node import LinkNode, QuaJobNode, print_yellow, print_green, print_red
from typing import Dict, Set, List, Tuple, Union
from copy import deepcopy
from time import time_ns
from datetime import datetime
from inspect import stack
import asyncio


class ProgramGraph:

    def __init__(self, label: str = None, graph_db: GraphDB = None, verbose=False) -> None:
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
        self._link_nodes_ids: Dict[int, Dict[int, List[str]]] = dict()  # Dict[node_id,Dict[out_node_id,out_vars_list]]
        self._tasks = dict()
        self._calling_script = stack()[1][0].f_code.co_filename
        self.graph_db: GraphDB = graph_db
        self.verbose = verbose

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

        # deep copy nodes
        _nodes_copy = dict()
        nodes_map = dict()
        for node in self.nodes.values():
            node_copy = node.deepcopy()
            _nodes_copy[node_copy.id] = node_copy
            # map the old node id's to the new ones
            nodes_map[node.id] = node_copy.id
        setattr(result, '_nodes', _nodes_copy)
        # update id's for LinkNode and _QuaJobNode
        for node in self.nodes.values():
            node_copy = _nodes_copy[nodes_map[node.id]]
            for var, value in node_copy.input_vars.__dict__.items():
                if isinstance(value, LinkNode) or isinstance(value, QuaJobNode):
                    setattr(node_copy.input_vars, var, deepcopy(value))
                    node_copy.input_vars[var].node = _nodes_copy[nodes_map[value.node.id]]

        for k, v in self.__dict__.items():
            if k == '_id':
                setattr(result, k, time_ns())
            elif k == '_nodes_by_label':
                _nodes_by_label_copy = dict()
                for k2, v2 in self._nodes_by_label.items():
                    _nodes_by_label_copy[k2] = {_nodes_copy[nodes_map[node.id]] for node in v2}
                setattr(result, k, _nodes_by_label_copy)
            elif k == '_edges':
                _edges_copy = dict()
                for k2, v2 in self.edges.items():
                    _edges_copy[nodes_map[k2]] = {nodes_map[node_id] for node_id in v2}
                setattr(result, k, _edges_copy)
            elif k == '_backward_edges':
                _backward_edges_copy = dict()
                for k2, v2 in self.backward_edges.items():
                    _backward_edges_copy[nodes_map[k2]] = {nodes_map[node_id] for node_id in v2}
                setattr(result, k, _backward_edges_copy)

            elif k == '_link_nodes_ids':
                _link_nodes_ids_copy = dict()
                for k2, v2 in self._link_nodes_ids.items():
                    v_copy = dict()
                    for node_id, vars2 in v2.items():
                        v_copy[nodes_map[node_id]] = vars2
                    _link_nodes_ids_copy[nodes_map[k2]] = v_copy
                setattr(result, k, _link_nodes_ids_copy)

            elif k != '_tasks' and k != '_nodes':
                setattr(result, k, deepcopy(v, memo))

        return result

    def deepcopy(self):
        """
        Implements deep copy - copy by value, provides new id
        :return:
        """
        return self.__deepcopy__()

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
    def start_time(self):
        return self._start_time

    @property
    def end_time(self):
        return self._end_time

    @property
    def nodes(self):
        return self._nodes

    @property
    def nodes_by_label(self):
        return self._nodes_by_label

    def add_nodes(self, *new_nodes: ProgramNode):
        """
        Adds the given nodes to the graph
        :param new_nodes: node objects
        :type new_nodes: ProgramNode
        :return:
        """
        for node in new_nodes:
            if self.graph_db:
                if self.graph_db.global_metadata_funcs:
                    node.metadata_funcs += self.graph_db.global_metadata_funcs

            self._nodes[node.id] = node
            self._nodes_by_label.setdefault(node.label, set()).add(node)
            if node.input_vars is not dict():
                for var, value in node.input_vars:
                    if isinstance(value, LinkNode):
                        self.add_edges((value.node, node))
                        node_input_ids = self._link_nodes_ids.setdefault(node.id, {value.node.id: list()})
                        node_input_ids.setdefault(value.node.id, list()).append(value.output_var)
                    if isinstance(value, QuaJobNode):
                        self.add_edges((value.node, node))
                        node_input_ids = self._link_nodes_ids.setdefault(node.id, {value.node.id: list()})
                        node_input_ids.setdefault(value.node.id, list()).append('!Qua-Job')
        if self.verbose:
            print_green(f"SUCCESS added nodes {[n.label for n in new_nodes]} to graph <{self.label}>")

    def remove_nodes(self, *nodes_to_remove: Union[int, str, ProgramNode]) -> None:
        """
        Removes the given nodes from the graph
        :param nodes_to_remove:nodes to remove by object, id or label
        :type nodes_to_remove: Union[ProgramNode,int,str]
        :return:
        """
        edges_to_remove: Set[Tuple[ProgramNode, ProgramNode]] = set()
        for node in nodes_to_remove:
            if type(node) is str:
                node_ids = [node.id for node in self._nodes_by_label[node]]
                print_yellow(f"ATTENTION removing {len(self._nodes_by_label[node])} node with the label <{node}>")
                for node_id in node_ids:
                    try:
                        edges_to_remove = self._remove_node(node_id, edges_to_remove)
                        if self.verbose:
                            print_green(f"SUCCESS removed node <{node}>")
                    except KeyError:
                        print_yellow(f"ATTENTION Tried to remove node <{node}> but it was not found in the graph")
                continue
            elif isinstance(node, ProgramNode):
                node_id = node.id
            elif type(node) is int:
                node_id = node
                node = self.nodes[node_id]
            else:
                print_red(f"WARNING given node must be one of a [{int},{str},{ProgramNode}] "
                          f"but given <{node}> is {type(node)}")
                continue
            try:
                edges_to_remove = self._remove_node(node_id, edges_to_remove)
                if self.verbose:
                    print_green(f"SUCCESS removed node <{node.label}>")
            except KeyError:
                print_yellow(f"ATTENTION Tried to remove node <{node.label}> but it was not found in the graph")

        self.remove_edges(*edges_to_remove)

    def _remove_node(self, node_id, edges_to_remove) -> set:
        # remove node
        node = self._nodes.pop(node_id)
        self._nodes_by_label.get(node.label, {node}).remove(node)
        if not self._nodes_by_label[node.label]:
            del self._nodes_by_label[node.label]

        # remove forward edges
        try:
            ids_to_remove = self.edges[node.id]
            for dest_node_id in ids_to_remove:
                edges_to_remove.add((node, self.nodes[dest_node_id]))
        except KeyError:
            if self.verbose:
                print_yellow(f'Node <{node.label}> has no outgoing edges.')

        # remove backward edges
        try:
            ids_to_remove = self.backward_edges[node.id]
            for source_node_id in ids_to_remove:
                edges_to_remove.add((self.nodes[source_node_id], node))
        except KeyError:
            if self.verbose:
                print_yellow(f'Node <{node.label}> has no incoming edges.')

        return edges_to_remove

    @property
    def edges(self) -> dict:
        return self._edges

    def add_edges(self, *edges: Tuple[ProgramNode, ProgramNode]):
        """
        Add edges between given nodes.
        When used outside of add_nodes method,
        it describes either time order rather than input_vars/output dependency as usual.
        :param edges: tuples (source_node, dest_node)...
        :type edges: Tuple[ProgramNode, ProgramNode]
        :return:
        """
        for source, dest in edges:
            if source.id not in self.nodes:
                print_red(f"WARNING tried to add edge between <{source.label}> and <{dest.label}>, "
                          f"but <{source.label}> was not added yet to the graph "
                          f"(maybe a copy of the node was added instead)")
                continue
            if dest.id not in self.nodes:
                print_red(f"WARNING tried to add edge between <{source.label}> and <{dest.label}>, "
                          f"but <{dest.label}> was not added yet to the graph "
                          f"(maybe a copy of the node was added instead)")
                continue
            self._edges.setdefault(source.id, set()).add(dest.id)
            self._backward_edges.setdefault(dest.id, set()).add(source.id)
            if self.verbose:
                print_green(f"SUCCESS added edge from <{source.label}> to <{dest.label}>")

        self._topological_sort()

    def remove_edges(self, *edges: Tuple[ProgramNode, ProgramNode]):
        """
        Remove edges between pairs of nodes
        :param edges: Pairs of nodes (source_node, dest_node)...
        :type edges: Tuple[ProgramNode, ProgramNode]
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
                if self.verbose:
                    print_green(f"SUCCESS removed edge from <{source.label}> to <{dest.label}>")
            except KeyError:
                print_yellow(f"ATTENTION Tried to remove edge from <{source.label}> to <{dest.label}> "
                             f"but it does not exist.")

    @property
    def backward_edges(self) -> dict:
        return self._backward_edges

    @property
    def graph_db(self) -> GraphDB:
        return self._graph_db

    @graph_db.setter
    def graph_db(self, graph_db: GraphDB):
        if (graph_db is not None) and (not isinstance(graph_db, GraphDB)):
            raise TypeError(f"graph_db must be of type {GraphDB}")
        self._graph_db = graph_db

    async def run_async(self,
                        graph_db: GraphDB = None,
                        start_nodes: List[ProgramNode] = None,
                        _calling_script_path: str = None,
                        verbose: bool = None
                        ) -> GraphDB:
        """
        Run the nodes in the graph in a BFS traversal order, while waiting for dependent tasks to complete
        :param verbose:
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

        if verbose:
            self.verbose = verbose

        graph_db = graph_db if graph_db else self.graph_db
        if graph_db:
            # Save graph to DB
            if self.verbose:
                print_green(f"Saving graph <{self.label}> to DB at '{graph_db.results_path}'")
            graph_db.save_graph(self, _calling_script_path)

        self._start_time = datetime.now()
        self._tasks = dict()

        # traverse the graph and run the nodes
        await self._graph_traversal(graph_db, start_nodes)

        # wait for all tasks(nodes) to complete
        await asyncio.gather(*self._tasks.values())
        self._end_time = datetime.now()

        if graph_db:
            # SAVE GRAPH RES TO DB HERE
            if self.verbose:
                print_green(f"Saving graph <{self.label}> results to DB at '{graph_db.results_path}'")
            graph_db.save_graph_results(self)

        return graph_db

    def run(self,
            graph_db: GraphDB = None,
            start_nodes: Union[List[ProgramNode], Set[ProgramNode]] = None,
            verbose: bool = None
            ) -> GraphDB:
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs.
        :param verbose: set True to print a more detailed description during the run
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
        return asyncio.run(self.run_async(graph_db, start_nodes, calling_script_path, verbose))

    async def _graph_traversal(self, graph_db, start_nodes):

        # the starting nodes of the run
        if not start_nodes:
            # start from the nodes that don't have incoming edges
            start_nodes = list()
            for node_id in self.nodes:
                if node_id not in self.backward_edges:
                    start_nodes.append(node_id)
            if not start_nodes:
                raise ValueError("Graph must be acyclic! Try changing dependencies.")
        else:
            start_nodes = [n.id for n in start_nodes]

        for node_id in self._get_next(start_nodes):
            if node_id not in self._tasks:
                if self._dependencies_started(node_id) or node_id in start_nodes:

                    if node_id not in start_nodes:
                        # wait for dependencies to complete
                        await asyncio.gather(*{self._tasks[t] for t in (self.backward_edges.get(node_id, set()))})

                    # direct the output of the dependencies input the input of the node

                    self._feed_input(node_id)
                    node = self.nodes[node_id]

                    if graph_db:
                        graph_db.save_node(node, self)
                        if self.verbose: print_green(f"Saving metadata before running node <{node.label}>")
                        graph_db.save_node_metadata(node, self)

                    # create task to run the node and start running
                    self._tasks[node_id] = asyncio.create_task(node.run_async())

    def _feed_input(self, node_id):
        for var, value in self.nodes[node_id].input_vars:
            if isinstance(value, LinkNode):
                link_node = value
                try:
                    self.nodes[link_node.node.id]
                except KeyError:
                    raise RuntimeError(f"Tried to use the output of node <{link_node.node.label}> "
                                       f"but the node is not in the graph.")

                setattr(self.nodes[node_id].input_vars, var, link_node.get_output())

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

    def _get_next(self, start_nodes: Union[List[int], Set[int]]):
        """
        Generator of graph nodes - implementing BFS
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

    def _topological_sort(self, start_nodes=None) -> List[int]:
        """
        Returns a list of graph node ids in a topological order. Starting from given start nodes.
        Implements Kahn's algorithm.
        :param start_nodes: list of nodes to start the sort from
        :type start_nodes: List[ProgramNode]
        :return:
        """
        if start_nodes is None:
            start_nodes = list()

        edges: Dict[int, Set[int]] = deepcopy(self.edges)
        backward_edges: Dict[int, Set[int]] = deepcopy(self.backward_edges)

        s: List[int]  # list of node ids with no incoming edges
        if not start_nodes:
            s = [n for n in self.nodes if n not in backward_edges]
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
        if edges.keys() != set():
            raise ValueError("Graph must be acyclic! Try changing dependencies.")
        # If graph has edges containing the supposedly sorted nodes, then there's a cycle.

        return sorted_list

    def join(self, graph: ProgramGraph = None) -> None:
        """
         Join a different graph to the current graph
        :param graph:
        :type graph: ProgramGraph
        :return:
        """
        if graph is not None:
            try:
                self._edges.update(graph.edges)
                self._backward_edges.update(graph.backward_edges)
                self.add_nodes(*graph.nodes.values())
            except:
                raise
        print_green(f"SUCCESS joined graph <{graph.label}> to graph <{self.label}")

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

# class GraphNode(ProgramNode):
#
#     def __init__(self, label: str = None,
#                  graph: ProgramGraph = None,
#                  input_vars: Dict[str, List[Tuple[Any, ProgramNode]]] = None,
#                  output_vars: Set[Tuple[str, str, ProgramNode]] = None,
#                  node_metadata_func: FunctionType = None):
#
#         super().__init__(label, None, input_vars, output_vars, node_metadata_func)
#         self.graph: ProgramGraph = graph
#         self._type = 'Graph'
#         self._job = None
#         # TODO: figure out how to combine graphs properly or i/o through graph node
#         # self.input_vars: {var:[(value, node),...],...}, the value wil go as input to of var to node
#         # self.output_vars: {(self_var, node_var,node),...}
#
#     def _get_result(self):
#         if self.output_vars is None:
#             print_yellow(f"ATTENTION No output variables were defined for node <{self.label}>")
#             return
#         for self_var, node_var, node in self.output_vars:
#             try:
#                 pass
#                 self._result[self_var] = self.graph.nodes[node.id].result[node_var]
#             except KeyError:
#                 print_red(f"WARNING Could not fetch '{node_var}' from node <{node.label}> results")
#
#     async def run_async(self) -> None:
#         if self.to_run:
#             # populate inputs
#             for var, vals in self.input_vars:
#                 for val, node in vals:
#                     if isinstance(val, LinkNode):
#                         node.input_vars.var = val.get_output()
#                     else:
#                         node.input_vars.var = val
#
#             self._start_time = datetime.now()
#             print("\nRUNNING GraphNode '{}'...".format(self.label))
#
#             self._job = await asyncio.create_task(self.graph.run_async())
#             print("DONE")
#             self._end_time = datetime.now()
#             self._get_result()
