from qm.QuantumMachinesManager import QuantumMachinesManager
import time
import numpy as np
from qm.qua import *
from encode_circuits import *


class QuaGST:
    def __init__(self, file: str, model, basic_gates_macros: dict, pre_circuit=None, post_circuit=None,
                 N_shots=1000,
                 config=None,
                 quantum_machines_manager: QuantumMachinesManager = None,
                 **execute_kwargs):

        self.file = file
        self.model = model
        self._get_circuit_list()
        self.basic_gates_macros = basic_gates_macros
        self._get_sequence_macros()
        self.pre_circuit = pre_circuit
        self.post_circuit = post_circuit
        self.config = config
        self.qmm = quantum_machines_manager
        self.N_shots = N_shots
        self.execute_kwargs = execute_kwargs
        self.results = []

    def _get_circuit_list(self):
        with open(file=self.file, mode='r') as f:
            circuits = f.readlines()
            self.circuit_list = encode_circuits(circuits, self.model)

    def _get_sequence_macros(self):
        self.gate_sequence, self.sequence_macros = gate_sequence_and_macros(self.model, self.basic_gates_macros)

    def _qua_circuit(self, encoded_circuit):
        _n_ = declare(int)
        # prep fiducials
        with switch_(encoded_circuit[0]):
            for i, m in enumerate(self.sequence_macros):
                with case_(i):
                    m()
        # germ
        with switch_(encoded_circuit[2]):
            for i, m in enumerate(self.sequence_macros):
                with case_(i):
                    with for_(_n_, 0, _n_ < encoded_circuit[3], _n_ + 1):
                        m()
        # meas fiducials
        with switch_(encoded_circuit[1]):
            for i, m in enumerate(self.sequence_macros):
                with case_(i):
                    m()

    def _encode_circuit_using_IO(self, qm, job):
        for circuit in self.circuit_list:
            for gate in range(0, len(circuit), 2):
                while not (job.is_paused()):
                    time.sleep(0.01)
                qm.set_io1_value(circuit[gate])
                qm.set_io2_value(circuit[gate + 1])
                job.resume()

    def gst_qua_IO(self, circuits, out_stream):
        _g_ = declare(int)
        _n_shots_ = declare(int)
        _n_circuits_ = declare(int)

        with for_(_n_circuits_, 0, _n_circuits_ < len(circuits), _n_circuits_ + 1):
            circuit = declare(int, size=4)  # Plug circuit size
            with for_(_g_, 0, _g_ < circuit.length(), _g_ + 2):
                pause()
                assign(circuit[_g_], IO1)
                assign(circuit[_g_ + 1], IO2)

            with for_(_n_shots_, 0, _n_shots_ < self.N_shots, _n_shots_ + 1):
                if self.pre_circuit:
                    self.pre_circuit()
                self._qua_circuit(circuit)
                if self.post_circuit:
                    self.post_circuit(out_stream)

    def gst_qua(self, circuits, out_stream):
        _n_shots_ = declare(int)
        circuit = [declare(int) for _ in range(4)]  # Plug circuit size
        with for_each_(circuit, circuits):
            with for_(_n_shots_, 0, _n_shots_ < self.N_shots, _n_shots_ + 1):
                if self.pre_circuit:
                    self.pre_circuit()
                self._qua_circuit(circuit)
                if self.post_circuit:
                    self.post_circuit(out_stream)

    def get_qua_program(self, gst_body, circuits):
        with program() as gst_prog:
            out_stream = declare_stream()

            gst_body(circuits, out_stream)

            with stream_processing():
                out_stream.buffer(len(circuits), self.N_shots).save("counts")

        return gst_prog

    def save_circuit_list(self, file):
        pass

    def run_IO(self, n_circuits=None):
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(self.get_qua_program(self.gst_qua_IO, self.circuit_list[:n_circuits]), **self.execute_kwargs)

        self._encode_circuit_using_IO(qm, job)

        job.result_handles.wait_for_all_values()

        self.results = job.result_handles.counts.fetch_all()

    def run(self, n_circuits: int):
        qm = self.qmm.open_qm(self.config)
        for i in range(len(self.circuit_list) // n_circuits + 1):
            circuits = self.circuit_list[i * n_circuits:(i + 1) * n_circuits]
            if circuits:
                job = qm.execute(self.get_qua_program(self.gst_qua, np.array(circuits).T.tolist()),
                                 **self.execute_kwargs)
                job.result_handles.wait_for_all_values()

                self.results.append(job.result_handles.counts.fetch_all())

    def get_results(self):
        pass

    def save_results(self):
        pass
