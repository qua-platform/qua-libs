import numpy as np
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import GSTModelPack
import pygsti


class QuaGST:
    def __init__(self, model: GSTModelPack, max_lengths, *gate_macros, pre_circuit=None, post_circuit=None, config=None,
                 quantum_machines_manager: QuantumMachinesManager = None, **execute_kwargs):
        self.pygsti_model = model
        self.max_lengths = max_lengths
        self.gates = gate_macros
        assert len(self.pygsti_model.gates) == len(self.gates)
        self.pre_circuit = pre_circuit
        self.post_circuit = post_circuit
        self.config = config
        self.qmm = quantum_machines_manager
        self.execute_kwargs = execute_kwargs
        self.results = None

    def _get_model_circuits(self):
        germs = self.pygsti_model.germs()
        prep_fiducials = self.pygsti_model.prep_fiducials()
        meas_fiducials = self.pygsti_model.meas_fiducials()
        ops = {}
        germs_indices = []
        prep_indices = []
        meas_indices = []

        op_index = 0
        for g in germs:
            for label in g:
                pass

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
