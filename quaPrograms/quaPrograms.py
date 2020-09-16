import networkx as nx
import matplotlib.pyplot as plt


class QuaProgramNode:
    def __init__(self, name, qua_prog, input_params, output_params=None):
        """
        Initialized the qua program node
        :param name: the name of the qua program
        :type name: str
        :param qua_prog: a python function which returns a qua program
        :type qua_prog: function
        :param input_params: variable names values to assign to variables in the qua program
        :type input_params: dict
        :param output_params: the names of the output variables of the qua program
        :type output_params: set
        """
        self.name = name
        self.qua_prog = qua_prog
        self.input_params = input_params
        self.output_params = output_params
        self.result = None

    def update_input_values(self, updated_inputs_values):
        self.input_params.update(updated_inputs_values)

    def load_input_values(self, input_params=None):
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

    def get_output_values(self, output_params=None):
        """
        Returns the specified output values from the qua program execution result
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
                print("The result of '{}' does not contain the variable '{}'".format(self.name, param))
        return output_values


class QuaGraphExecutor:

    def __init__(self, executor, qua_graph):
        """
        Initialize qua program graph executor
        :param executor: a python function that returns a QmJob given a qua program
        :type executor: function
        :param qua_graph: is a DAG that contains the qua program nodes and data flow structure
        :type qua_graph: networkx.DiGraph
        """

        self.executor = executor
        self.graph = qua_graph

    def execute(self):
        """
        Execute the qua programs in the graph in a topological order, and save the results in the nodes
        """
        program_queue = nx.topological_sort(self.graph)

        for prog_name in program_queue:
            curr_prog_node = self.get_prog_node(prog_name)

            for pred_name in self.graph.predecessors(prog_name):
                '''
                Update the current program inputs using the predecessors outputs
                '''
                curr_prog_node = self.get_prog_node(pred_name)
                updated_input_params = curr_prog_node.get_output_values(curr_prog_node.input_params)

                curr_prog_node.update_input_values(updated_input_params)

            job = self.executor(curr_prog_node.load_input_values())
            curr_prog_node.result = job.result_handles

    def plot(self):
        """
        Visualize the graph structure
        """
        plt.tight_layout()
        nx.draw_networkx(self.graph, arrows=True)

    def get_prog_node(self, name):
        """
        Returns a qua program given the name
        :param name: the name of the qua program node
        :return: QuaProgramNode
        """
        return self.graph.nodes[name]['prog']
