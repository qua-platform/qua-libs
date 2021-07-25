import numpy as np
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from pygsti.construction import make_lsgst_experiment_list
from pygsti.modelpacks import GSTModelPack
import pygsti
import time


class QuaGST:
    def __init__(self, file: str, max_lengths, *gate_macros, circuit_list, config=None,
                 quantum_machines_manager: QuantumMachinesManager = None, sequence_dict=None, N_shots=1000,
                 **execute_kwargs):
        if sequence_dict is None:
            sequence_dict = {}
        self.file = file
        self.max_lengths = max_lengths
        self.gates = gate_macros
        # assert len(self.pygsti_model.gates) == len(self.gates)
        self.config = config
        self.qmm = quantum_machines_manager
        self.N_shots = N_shots
        self.execute_kwargs = execute_kwargs
        self.results = None
        self.sequence_dict = sequence_dict
        self.circuit_list = circuit_list

    # def _get_model_circuits(self):
    #     germs = self.pygsti_model.germs()
    #     prep_fiducials = self.pygsti_model.prep_fiducials()
    #     meas_fiducials = self.pygsti_model.meas_fiducials()
    #     ops = {}
    #     germs_indices = []
    #     prep_indices = []
    #     meas_indices = []
    #
    #     op_index = 0
    #     for g in germs:
    #         for label in g:
    #             pass

    def _encode_angles_in_IO(self, qm, job):
        for circuit in self.circuit_list:
            while not (job.is_paused()):
                time.sleep(0.01)
            qm.set_io2_value(len(circuit))
            job.resume()
            for gate in range(0, len(circuit), 2):
                while not (job.is_paused()):
                    time.sleep(0.01)
                qm.set_io1_value(circuit[gate])
                qm.set_io2_value(circuit[gate+1])
                job.resume()

    def play_gate(self, g):
        with switch_(g):
            for i in range(len(self.sequence_dict.keys())):
                with case_(i):
                    self.sequence_dict[i]["macro"]  # Play macro attached to index i

    def get_qua_program(self):
        nb_of_circuits = len(self.circuit_list)
        with program() as gst_prog:
            g = declare(int)
            n = declare(int)
            n2 = declare(int)
            out_stream = declare_stream()

            with for_(n2, 0, n2 < nb_of_circuits, n2+1):
                pause()
                circuit = declare(int, size=IO2)  # Plug circuit size
                with for_(g, 0, g < circuit.length(), g+2):
                    pause()
                    assign(circuit[g], IO1)
                    assign(circuit[g+1], IO2)

                with for_(n, 0, n < self.N_shots, n+1):
                    with for_(g, 0, g < circuit.length(), g+1):
                        self.play_gate(circuit[g])  # Attach to sequence dict a macro for measurement + saving

            with stream_processing():
                out_stream.average().buffer(nb_of_circuits).save("out")

        return gst_prog

    def save_circuit_list(self, file):
        pass

    def run(self):
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(self.get_qua_program(), **self.execute_kwargs)

        self._encode_angles_in_IO(qm, job)


    def get_results(self):
        pass

    def save_results(self):
        pass
