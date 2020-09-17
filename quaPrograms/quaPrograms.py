import networkx as nx
import matplotlib.pyplot as plt


class QuaProgramNode:
    def __init__(self, node_id, label=None, qua_prog=None, input_params=dict(), output_params=set()):
        """
        Initialized the qua program node
        :param node_id: a unique id to identify the node in the graph
        :param label: the label of the qua program
        :type label: str
        :param qua_prog: a python function which returns a qua program
        :type qua_prog: function
        :param input_params: variable names values to assign to variables in the qua program
        :type input_params: dict
        :param output_params: the names of the output variables of the qua program
        :type output_params: set
        """
        self.id = node_id
        self.label = label
        self.qua_prog = qua_prog
        self.input_params = input_params
        self.output_params = output_params
        self.result = None

    def load_inputs(self, input_params=None):
        """
        Loads the specified variable values to the qua program
        :param input_params: variable names values to assign to variables in the qua program
        :type input_params: dict
        :return: a executable qua program
        """
        if input_params is None:
            return self.qua_prog(**self.input_params)
        else:
            self.input_params = input_params
        return self.qua_prog(**self.input_params)

    def get_outputs(self, output_params=None):
        """
        Returns the specified output values of the qua program result
        :param output_params: the desired output parameters
        :type output_params: set
        :return: dict with the desired output parameters
        """
        output_values = dict()
        if output_params is None:
            output_params = self.output_params
        for param in output_params:
            try:
                output_values[param] = getattr(self.result, param).fetch_all()['value']
            except AttributeError:
                print("The result of '{}' doesn't contain the variable '{}'".format(self.label, param))
        return output_values

    def duplicate(self, qua_program_node):
        """
        Duplicate all the attributes of a qua node except the id and result
        :param qua_program_node: a node to duplicate all the values from
        :type qua_program_node: QuaProgramNode
        """
        self.label = qua_program_node.label
        self.qua_prog = qua_program_node.qua_prog
        self.input_params = qua_program_node.input_params
        self.output_params = qua_program_node.output_params

class QuaGraphExecutor:

    def __init__(self, executor, graph):
        """
        Initialize qua program graph executor
        :param executor: a python function that returns a QmJob given a qua program
        :type executor: function
        :param graph: is a DAG that contains the qua program nodes and data flow structure
        :type graph: networkx.DiGraph
        """

        self.executor = executor
        self.graph = graph
        self.labels = dict()

    def add_nodes(self, qua_programs):
        """
        Adds all the programs in the list to the graph
        :param qua_programs: a list of QuaProgramNode nodes
        :type qua_programs: list
        """
        for node in qua_programs:
            self.labels[node.id] = node.label
            self.graph.add_node(node.id, prog=node)

    def execute(self, start_node_name=None):
        """
        Execute the qua programs in the graph in a topological order, and save the results in the nodes
        """
        program_queue = nx.topological_sort(self.graph)

        for prog_id in program_queue:
            curr_qua_node = self.get_qua_node(prog_id)

            for pred_id in self.graph.predecessors(prog_id):
                '''
                Update the current qua program inputs using the predecessors' qua program outputs
                '''
                pred_qua_node = self.get_qua_node(pred_id)
                updated_input_params = pred_qua_node.get_outputs(curr_qua_node.input_params)

                curr_qua_node.input_params.update(updated_input_params)

            job = self.executor(curr_qua_node.load_inputs())
            curr_qua_node.result = job.result_handles

    def plot(self, start_node_id=None):
        """
        Visualize the graph structure
        """
        plt.tight_layout()
        nx.draw_networkx(self.graph, arrows=True, labels=self.labels)

    def get_qua_node(self, node_id):
        """
        Returns a qua program contained in the graph node with the given name
        :param node_id: the name of the qua program node
        :return: QuaProgramNode
        """
        return self.graph.nodes[node_id]['prog']

    def topological_order(self, node_id):
        """

        :param node_id: graph node name
        :return: graph containing qua nodes
        """
        return
