from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig, LoopbackInterface
import queue
import networkx as nx
import matplotlib.pyplot as plt


class QuaProgram:

    def __init__(self, name, qua_prog, input_params, output_params):
        self.name = name
        self.qua_prog = qua_prog  # the code of the qua program wrapped by a python decorator for variables
        self.input_params = input_params  # the values to assign the vars of the qua program as
        # keyword arguments
        self.output_params = output_params  # dict with the output variables of the qua progs
        self.result = None  # result handles of the program after it ran

    def load_input(self):
        return self.qua_prog(**self.input_params)

    def load_input(self, new_input_params):
        return self.qua_prog(**new_input_params)

    def get_output(self):
        # get a dictionary out of the results with the wanted variables as keys and the results values as values
        for param in self.output_params:
            self.output_params[param] = getattr(self.result, param).fetch_all()['value']
        return self.output_params


class QuaGraphExecutor:

    def __init__(self, executor, qua_graph):
        self.executor = executor  # is the QM that's executes or simulates

        self.qua_graph = qua_graph  # networkx.Graph object of DAG

    def execute(self):
        program_queue = nx.topological_sort(self.qua_graph)

        for prog_name in program_queue:
            curr_prog = self.qua_graph.nodes[prog_name]['prog']
            job = self.executor(curr_prog.load_input())
            curr_prog.result = job.result_handles

    def plot(self):
        plt.tight_layout()
        nx.draw_networkx(self.qua_graph, arrows=True)
