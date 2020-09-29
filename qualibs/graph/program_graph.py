from .program_node import LinkNode, ProgramNode
from typing import Dict, Set, List, Tuple
from copy import deepcopy


class GraphJob:
    def __init__(self, graph):
        self.execution = graph


class ProgramGraph:

    def __init__(self, label: str = None) -> None:
        """
        A program graph describes a program flow with input_vars/output dependencies
        :param label: a label for the graph
        :type label: str
        """
        self._id: int = id(self)
        self.label: str = label
        self._nodes: Dict[int, ProgramNode] = dict()
        self._node_counter: int = 0
        self._edges: Dict[int, Set[int]] = dict()
        self._backward_edges: Dict[int, Set[int]] = dict()
        self._timestamp = None
        self._output: dict = dict()
        self._link_nodes: Dict[int, Dict[str, LinkNode]] = dict()  # Dict[node_id,Dict[input_var_name,LinkNode]]
        self._link_nodes_ids: Dict[int, Dict[int, List[str]]] = dict()  # Dict[node_id,Dict[out_node_id,out_vars_list]]
        self._execution_order: List[int] = list()
        self.update_order: bool = True  # Whether to update the execution order when running

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
            self._node_counter += 1
            if node.input_vars is not dict():
                for var, value in node.input_vars.items():
                    if isinstance(value, LinkNode):
                        self.add_edges({(value.node, node)})
                        self._link_nodes.setdefault(node.id, dict())[var] = value
                        self._link_nodes_ids.setdefault(node.id, {value.node.id: list()})[value.node.id].append(
                            value.output_var)
        self.update_order = True

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

        self.update_order = True

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

        self.update_order = True

    @property
    def backward_edges(self):
        return self._backward_edges

    @property
    def timestamp(self):
        return self._timestamp

    @property
    def output(self):
        return self._output

    def run(self, start_nodes: List[ProgramNode] = list()):
        """
        Run the graph nodes in the correct order while propagating the inputs/outputs.
        NOT YET: If given start_nodes, run the directed subgraph starting from those nodes.
        :param start_nodes: list of nodes to start running the graph from
        :type: start_nodes: List[ProgramNode]
        :return:
        """
        if not start_nodes:
            for node_id in self.nodes:
                if node_id not in self.backward_edges:
                    start_nodes.append(self.nodes[node_id])

        if self.update_order:
            self._execution_order = self.topological_sort(start_nodes)
            self.update_order = False

        for node_id in self._execution_order:
            # Put one output variable of one node into one input_vars variable of of a different node
            input_vars: Dict[str, LinkNode] = self._link_nodes.get(node_id, set())
            for var in input_vars:
                link_node = input_vars[var]
                assert self.nodes.get(link_node.node.id, None), \
                    "Tried to use the output of node <{}> as input_vars to <{}>,\n" \
                    "but <{}> isn't in the graph." \
                    .format(link_node.node.label, self.nodes[node_id].label, link_node.node.label)
                if link_out := link_node.output_var:
                    self.nodes[node_id].input_vars[var] = link_node.node.result[link_out]
                else:  # if output_var in the link node is not specified, forward the full result
                    self.nodes[node_id].input_vars[var] = link_node.node.result

            self.nodes[node_id].run()

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
        sorted_set: Set[int] = set()  # set of node ids
        if not start_nodes:
            s = [n for n in self.nodes if n not in backward_edges]
        else:
            s = [n.id for n in start_nodes]
        assert s != [], "Error: Graph is cyclic ! All nodes depend on other nodes, try changing dependencies."
        sorted_list: List[int] = []  # list that will contain the topologically sorted node ids

        while s:
            n = s.pop(0)
            sorted_set.add(n)
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

    def plot(self, start_nodes=None):
        """
        Plot the directed graph.
        If given start_nodes, plot the directed subgraph starting from those nodes.
        :param start_nodes: list of nodes to start plotting from
        :return:
        """

    def merge(self, graph):
        """
        Merge graph into the current graph
        :param graph:
        :return:
        """
        pass
