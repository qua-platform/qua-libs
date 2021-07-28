import numpy as np
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import GSTModelPack
import pygsti


class QuaGST:
    def __init__(
        self,
        file: str,
        *gate_macros,
        config=None,
        quantum_machines_manager: QuantumMachinesManager = None,
        **execute_kwargs,
    ):
        self.file = file
        self.gates = gate_macros
        self.config = config
        self.qmm = quantum_machines_manager
        self.execute_kwargs = execute_kwargs
        self.results = None

    def _read_file(self):
        circ_list = []
        qubit_list = []
        with open(file=self.file, mode="r") as f:
            circuits = f.readlines()
            for circ in circuits:
                gates = []
                qubit_indices = []
                c = circ.rstrip()
                for char in range(len(c)):
                    if char == 0:
                        gates.append(c[0 : c.find(":")])
                        qubit_indices.append(int(c[c.find(":") + 1]))

                    else:
                        if char == "G":
                            gates.append(c[char : c.find(":", char)])
                            qubit_indices.append(c[c.find(":", char) + 1])

                circ_list.append(gates)
                qubit_list.append(qubit_indices)

        f.close()

    def get_qua_program(self, counts):
        pass

    def save_circuit_list(self, file):
        pass

    def run(self, counts=100):
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(self.get_qua_program(counts), **self.execute_kwargs)

    def get_results(self):
        pass

    def save_results(self):
        pass
