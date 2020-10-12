import inspect

from .program_node import LinkNode, ProgramNode, QuaJobNode
from qualibs.results.api import *
from qualibs.results.impl.sqlalchemy import Results, SqlAlchemyResultsConnector
from typing import Dict, Set, List, Tuple, Any
from copy import deepcopy
from time import time_ns
import asyncio


class GraphJob:
    def __init__(self, graph):
        self.timestamp = time_ns()  # when job started
        self.graph = graph


class GraphDB:
    def __init__(self, results_path: str = ':memory:'):
        self._results_path = results_path
        self._dbcon = SqlAlchemyResultsConnector(backend=self._results_path)
        self.calling_script_path = inspect.stack()[1][0].f_code.co_filename
        self.calling_script = open(self.calling_script_path).read() if self.calling_script_path else None


class ProgramGraph:

    def __init__(self, label: str = None, results_path: str = ':memory:') -> None:
        """
        A program graph describes a program flow with input_vars/output dependencies
        :param label: a label for the graph
        :type label: str
        """
        self._id: int = time_ns()
        self.label: str = label
        self._nodes: Dict[int, ProgramNode] = dict()
        self._edges: Dict[int, Set[int]] = dict()
        self._backward_edges: Dict[int, Set[int]] = dict()
        self._timestamp = None  # when last finished running
        self._link_nodes: Dict[
            int, Dict[str, Union[LinkNode, QuaJobNode]]] = dict()  # Dict[node_id,Dict[input_var_name,LinkNode]]
        self._link_nodes_ids: Dict[int, Dict[int, List[str]]] = dict()  # Dict[node_id,Dict[out_node_id,out_vars_list]]
        self._tasks = dict()
        self._results_path = results_path
        
        # connect graph to DB
        self._dbcon = SqlAlchemyResultsConnector(backend=self._results_path)
        self.init_graph_db(inspect.stack()[1][0].f_code.co_filename)

    def init_graph_db(self, calling_script_path):
        # get script that runs graph
        calling_script = open(calling_script_path).read() if calling_script_path else None
        # save graph to database
        self._dbcon.save(Graph(graph_id=self.id, graph_script=calling_script, graph_name=self.label))
        # save nodes to database
        for node_id, node in self.nodes.items():
            self._dbcon.save(Node(graph_id=self.id, node_id=node_id, node_name=node.label))

    @property
    def id(self) -> int:
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
            if node.input_vars is not dict():
                for var, value in node.input_vars.items():
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
                self._nodes.pop(node.id)
                # remove forward edges
                try:
                    ids_to_remove = self.edges[node.id]
                    for dest_node_id in ids_to_remove:
                        edges_to_remove.add((node, self.nodes[dest_node_id]))
                except KeyError:
                    print('Node <{}> has no outgoing edges.'.format(node.label))
                # remove backward edges
                try:
                    ids_to_remove = self.backward_edges[node.id]
                    for source_node_id in ids_to_remove:
                        edges_to_remove.add((self.nodes[source_node_id], node))
                except KeyError:
                    print('Node <{}> has no incoming edges.'.format(node.label))
                print("Successfully removed node <{}>".format(node.label))
            except KeyError:
                print("KeyError: Tried to remove node <{}>, but was not found".format(node.label))
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
            self._edges.setdefault(source.id, set()).add(dest.id)
            self._backward_edges.setdefault(dest.id, set()).add(source.id)

    def remove_edges(self, edges: Set[Tuple[ProgramNode, ProgramNode]]):
        """
        Remove edges from graph
        :param edges: set of tuples {(source_node, dest_node)...}
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

    async def run_async(self, start_nodes: List[ProgramNode] = list()) -> GraphJob:
        """
        Run the nodes in the graph by BFS order, while waiting for dependent tasks to complete
        :param start_nodes:
        :return:
        """
        if not start_nodes:
            for node_id in self.nodes:
                if node_id not in self.backward_edges:
                    start_nodes.append(self.nodes[node_id])

        current_job = GraphJob(self)

        for node_id in self.get_next(start_nodes):
            if node_id not in self._tasks:
                if self.dependencies_started(node_id):  # or node_id in start_nodes: # TODO: make sure it works
                    # wait for dependencies to complete
                    await asyncio.gather(*{self._tasks[t] for t in self.backward_edges.get(node_id, set())})
                    # direct the output of the dependencies input the input of the node
                    input_vars: Dict[str, Union[LinkNode, QuaJobNode]] = self._link_nodes.get(node_id, set())
                    for var in input_vars:
                        if isinstance(input_vars[var], LinkNode):
                            link_node = input_vars[var]
                            assert self.nodes.get(link_node.node.id, None), \
                                f"Tried to use the output of node <{link_node.node.label}> " \
                                f"as input to <{self.nodes[node_id].label}>," \
                                f"\nbut <{link_node.node.label}> isn't in the graph."
                            self.nodes[node_id].input_vars[var] = link_node.get_output()

                    # SAVE METADATA TO DB HERE

                    # create task to run the node and start running
                    self._tasks[node_id] = asyncio.create_task(self.nodes[node_id].run_async())

        # wait for all tasks(nodes) to complete running
        await asyncio.gather(*self._tasks.values())
        # self._tasks = dict()  # TODO: Figure out whether there's need to reset the tasks dict
        self._timestamp = time_ns()
        for node_id, node in self.nodes.items():
            for res_name in node.result.keys():
                self._dbcon.save(Result(graph_id=self.id, node_id=node_id, result_id=1,
                                        start_time=node._start_time,
                                        end_time=node._end_time,
                                        user_id='User',
                                        res_name=res_name,
                                        res_val=str(node.result[res_name])
                                        ))

            # SAVE GRAPH RES TO DB HERE
            # TODO: Maybe do something to current job before returning
        return current_job

    def run(self, start_nodes: Union[List[ProgramNode], Set[ProgramNode]] = list()) -> GraphJob:
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs.
        :param start_nodes: list of nodes to start running the graph from
        :type: start_nodes: : Union[List[ProgramNode], Set[ProgramNode]]
        :return:
        """
        return asyncio.run(self.run_async(start_nodes))

    def dependencies_started(self, node_id) -> bool:
        return self._backward_edges.get(node_id, set()) <= self._tasks.keys()

    def dependencies_done(self, node_id) -> bool:
        for depend_id in self.backward_edges.get(node_id, set()):
            try:
                if self._tasks[depend_id].done():
                    continue
                else:
                    return False
            except KeyError:
                return False
        return True

    def get_next(self, start_nodes: Union[List[ProgramNode], Set[ProgramNode]]):
        """
        Generator of graph nodes - implementing BFS
        :param start_nodes: the start positions of graph traversal
        :type start_nodes: List/Set of ProgramNode
        :return:
        """
        to_do = [n.id for n in start_nodes]
        while to_do:
            s = self.nodes[to_do.pop(0)]
            yield s.id
            try:
                for child in self.edges[s.id]:
                    to_do.append(child)
            except KeyError:
                pass

    def topological_sort(self, start_nodes: List[ProgramNode] = list()) -> List[int]:
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
        else:
            s = [n.id for n in start_nodes]
        assert s != [], "Error: Graph is cyclic ! All nodes depend on other nodes, try changing dependencies."
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
                        dot_graph += '"{}" -> "{}"'.format(self.nodes[node_id].label, self.nodes[dest_id].label)
                    else:
                        dot_graph += '"{}" -> "{}"'.format(node_id, dest_id)

                    var_name = self._link_nodes_ids.get(dest_id, dict()).get(node_id, -1)
                    if var_name is None:
                        var_name = '!all'
                    if var_name == -1:
                        var_name = '!none'

                    dot_graph += ' [label="{}"]'.format(var_name)

                    dot_graph += ';'
            else:
                dot_graph += '"{}";'.format(self.nodes[node_id].label)

        dot_graph += '}'

        return dot_graph


class GraphNode(ProgramNode):
    def __init__(self, label: str = None, graph: ProgramGraph = None, input_vars: Dict[str, Any] = None,
                 output_vars: Set[str] = None):
        super().__init__(label, None, input_vars, output_vars)
        self.graph: ProgramGraph = graph
        self._type = 'Graph'
        self._job = None

    def get_result(self):
        if self.output_vars is None:
            print("ATTENTION! No output variables defined for node <{}>".format(self.label))
            return
        for var in self.output_vars:
            try:
                self._result[var] = self.graph.result[var]
            except KeyError:
                print("Couldn't fetch '{}' from the program graph results".format(var))

    async def run(self):
        if self.to_run:
            self._start_time = datetime.now()
            print("\nRUNNING GraphNode '{}'...".format(self.label))
            self._job = await asyncio.create_task(self.graph.run_async())
            print("DONE")
            self._end_time = datetime.now()
            self.get_result()
