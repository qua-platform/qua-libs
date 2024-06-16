from dataclasses import asdict
from typing import Union, List, Optional
import numpy as np
from qm.qua import *
from macros import qua_declaration, reset_qubit, cross_entropy, binary, exponential_decay, fit_exponential_decay
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerJob
import pandas as pd
from xeb_config import XEBConfig
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from components import QuAM
from quam.components import Channel
from qm import SimulationConfig
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
import seaborn as sns
from copy import deepcopy

SW = UnitaryGate(np.array([[1, -np.sqrt(1j)], [np.sqrt(-1j), 1]]) / np.sqrt(2), label="sw")  # from Supremacy paper


class XEB:
    def __init__(self, xeb_config: XEBConfig, quam: QuAM):
        """
        Initialize the XEB experiment
        Args:
            xeb_config: XEBConfig object containing the parameters of the experiment
            quam: QuAM object containing the Quantum Machine configuration
        """
        self.xeb_config = xeb_config
        self.quam = quam
        self.qubits = [quam.qubits[qubit_id] for qubit_id in xeb_config.qubits_ids]
        self.qubit_elements = [qubit.xy for qubit in self.qubits]
        self.readout_elements = [qubit.resonator for qubit in self.qubits]
        self.data_handler = DataHandler(name="XEB", root_data_folder=xeb_config.save_dir)

    def _assign_amplitude_matrix(self, gate_idx, amp_matrix):
        """
        Assign the amplitude matrix of a gate based on the gate index

        Args:
            gate_idx (QUA int): Index of the gate
            amp_matrix (List): Amplitude matrix of the gate
        """
        with switch_(gate_idx):
            for i in range(len(self.xeb_config.gate_set)):
                with case_(i):
                    for j in range(4):
                        assign(amp_matrix[j], self.xeb_config.gate_set[i].amp_matrix[j])

    def _play_random_sq_gate(self, qubit: Channel, gate_idx, amp_matrix: Optional[List] = None):
        """
        Play a random single qubit gate on a given qubit element.

        This function plays a random single qubit gate on a given qubit element, by modulating the amplitude matrix of a
        baseline calibrated X/2 (SX) pulse.

        Args:
            qubit (Channel): Qubit element on which to play the gate.
            gate_idx (QUA int): Index of the gate to play.
            amp_matrix (List): Amplitude matrix of the gate.
        """
        if self.xeb_config.gate_set.run_through_amp_matrix_modulation and amp_matrix is not None:
            qubit.play(self.xeb_config.baseline_gate_name, amplitude_scale=amp(*amp_matrix))
        else:
            with switch_(gate_idx, unsafe=True):
                for i in range(len(self.xeb_config.gate_set)):
                    with case_(i):
                        self.xeb_config.gate_set[i].gate_macro(qubit)

    def _xeb_prog(self, simulate: bool = False):
        # Define the QUA program
        n_qubits = self.xeb_config.n_qubits
        dim = self.xeb_config.dim
        random_gates = len(self.xeb_config.gate_set)
        ge_thresholds = [readout_element.operations["readout"].threshold for readout_element in self.readout_elements]

        with program() as xeb_prog:
            # Declare QUA variables
            I, I_st, Q, Q_st = qua_declaration(n_qubits=n_qubits, readout_elements=self.readout_elements)
            depth, depth_, n, s, tot_state_ = [declare(int) for _ in range(5)]
            gate = [
                declare(int, size=self.xeb_config.depths[-1]) for _ in range(n_qubits)
            ]  # Gate indices list for both qubits
            if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                amp_matrix = [
                    [declare(fixed, size=self.xeb_config.depths[-1]) for _ in range(4)] for _ in range(n_qubits)
                ]
            counts = declare(int, value=[0] * dim)  # Counts for all possible bitstrings (00, 01, 10, 11)
            state = [declare(bool) for _ in range(n_qubits)]  # Qubit states
            # Declare streams
            counts_st = [
                declare_stream() for _ in range(dim)
            ]  # Stream for counts of all possible bitstrings (00, 01, 10, 11)
            state_st = [declare_stream() for _ in range(n_qubits)]  # Stream for individual qubit states
            gate_st = [
                declare_stream() for _ in range(n_qubits)
            ]  # Stream for gate indices (enabling circuit reconstruction in post-processing)

            # Setting seed for reproducibility
            r = Random()
            r.set_seed(12321)

            # If simulating, update the frequency to 0 to visualize sequence
            if simulate:
                amp_st = [[declare_stream() for _ in range(4)] for _ in range(n_qubits)]
                for qubit in self.qubit_elements:
                    update_frequency(qubit.name, 0)

            # Generate the random sequences
            with for_(s, 0, s < self.xeb_config.seqs, s + 1):
                with for_each_(depth, self.xeb_config.depths):
                    # NOTE: randomizing is done for each growing-depths and sequence (some other strategies could be used)
                    for q in range(n_qubits):
                        first_gate = random_gates - 1 if self.xeb_config.impose_0_cycle else random_gates
                        assign(gate[q][0], r.rand_int(first_gate))
                        save(gate[q][0], gate_st[q])
                    with for_(depth_, 1, depth_ < depth, depth_ + 1):
                        for q in range(n_qubits):
                            assign(gate[q][depth_], r.rand_int(random_gates))
                            with while_(
                                gate[q][depth_] == gate[q][depth_ - 1]
                            ):  # Make sure same gate is not applied twice in a row
                                assign(gate[q][depth_], r.rand_int(random_gates))

                            save(gate[q][depth_], gate_st[q])

                            # Map indices into amplitude matrix arguments
                            # (each index corresponds to a random gate)
                            if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                                self._assign_amplitude_matrix(
                                    gate[q][depth_], [amp_matrix[q][i][depth_] for i in range(4)]
                                )
                                if simulate:
                                    for amp_matrix_element in range(4):
                                        save(amp_matrix[q][amp_matrix_element][depth_], amp_st[q][amp_matrix_element])

                    # Run the XEB sequence
                    with for_(n, 0, n < self.xeb_config.n_shots, n + 1):
                        # Reset the qubits to their ground states (here simple wait but could be an active reset macro)
                        if simulate:
                            wait(25, *[qubit.name for qubit in self.qubit_elements])

                        # NOTE: imposing first gate at 0-cycle:
                        # Could be modified by the user to impose a specific gate at 0-cycle
                        if self.xeb_config.impose_0_cycle:
                            for qubit in self.qubit_elements:
                                qubit.play(
                                    self.xeb_config.baseline_gate_name,
                                    amplitude_scale=amp(*list(0.70710678 * np.array([1.0, -1.0, 1.0, 1.0]))),
                                )

                        # Play all cycles generated for sequence s of depth d
                        with for_(depth_, 0, depth_ < depth, depth_ + 1):
                            for q, qubit in enumerate(self.qubit_elements):  # Play single qubit gates on both qubits
                                self._play_random_sq_gate(
                                    qubit,
                                    gate[q][depth_],
                                    (
                                        [amp_matrix[q][i][depth_] for i in range(4)]
                                        if self.xeb_config.gate_set.run_through_amp_matrix_modulation
                                        else None
                                    ),
                                )

                            # Insert your two-qubit gate macro here
                            if self.xeb_config.two_qb_gate is not None:
                                for qubit in self.qubit_elements:
                                    qubit.align(qubit.parent.z.name, qubit.parent.resonator.name)
                                self.xeb_config.two_qb_gate.gate_macro(self.qubits[0], self.qubits[1])
                                for qubit in self.qubit_elements:
                                    qubit.align(qubit.parent.z, qubit.parent.resonator)

                        # Measure the state (insert your readout macro here)
                        for q_idx, readout_element in enumerate(self.readout_elements):
                            readout_element.measure("readout", qua_vars=(I[q_idx], Q[q_idx]))
                            # State Estimation: returned as an integer, to be later converted to bitstrings
                            assign(state[q_idx], I[q_idx] > ge_thresholds[q_idx])
                            save(state[q_idx], state_st[q_idx])
                            save(I[q_idx], I_st[q_idx])
                            save(Q[q_idx], Q_st[q_idx])
                            assign(tot_state_, tot_state_ + 2**q_idx * Cast.to_int(state[q_idx]))

                            reset_qubit(
                                self.xeb_config.reset_method,
                                self.qubits[q_idx],
                                threshold=ge_thresholds[q_idx],
                                **self.xeb_config.reset_kwargs,
                            )

                        with switch_(tot_state_):
                            for i in range(dim):  # Bitstring conversion
                                with case_(i):
                                    assign(counts[i], counts[i] + 1)  # counts for 00, 01, 10 and 11
                        assign(tot_state_, 0)  # Resetting the state
                    for i in range(dim):  # Resetting Bitstring collection
                        save(counts[i], counts_st[i])
                        assign(counts[i], 0)

            # Save the results
            with stream_processing():
                for q in range(n_qubits):
                    gate_st[q].save_all(f"g{q}")
                    I_st[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"I{q}")
                    Q_st[q].buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"Q{q}")
                    state_st[q].boolean_to_int().buffer(self.xeb_config.n_shots).map(FUNCTIONS.average()).buffer(
                        len(self.xeb_config.depths)
                    ).save_all(f"state{q}")
                for i in range(dim):
                    string = "s" + binary(i, n_qubits)
                    counts_st[i].buffer(len(self.xeb_config.depths)).save_all(string)

                if simulate:
                    for q in range(n_qubits):
                        for d_ in range(4):
                            amp_st[q][d_].save_all(f"a{q + 1}_{binary(d_, 2)}")

        return xeb_prog

    def run(self, simulate: bool = False):
        """
        Run QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not

        Returns: XEBJob object containing the information about the experiment (including results)

        """
        # Compile the QUA program

        config = self.quam.generate_config()
        xeb_prog = self._xeb_prog(simulate)
        qmm = self.quam.connect()
        if simulate:
            job = qmm.simulate(config, xeb_prog, simulate=SimulationConfig(1000))
        elif self.xeb_config.generate_new_data:
            qm = qmm.open_qm(config)
            job = qm.execute(xeb_prog)
        else:
            raise NotImplementedError("Data fetching from previous runs is not yet implemented")

        return XEBJob(job, self.xeb_config, self.data_handler)

    def simulate(self, backend: BackendV2):
        """
            Simulate the XEB experiment: To simulate it, you must provide an AerBackend object with a noise model
            corresponding to your experiments parameters.
            For instance,
        ```python
        from qiskit import Aer
        from qiskit.providers.aer import AerSimulator
        from qiskit.providers.aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
        error1q = 0.07
        error2q = 0.03
        depol_error1q = depolarizing_error(error1q, 1)
        depol_error2q = depolarizing_error(error2q, 2)
        sq_gate_set = ["h", "t", "rx", "ry", "sw"] # Specify which gates are subject to noise
        noise_model = NoiseModel(basis_gates = sq_gate_set)
        noise_model.add_all_qubit_quantum_error(depol_error2q, ["cz"])
        noise_model.add_all_qubit_quantum_error(depol_error1q, sq_gate_set)
        backend = AerSimulator(noise_model=noise_model, method="density_matrix", basis_gates=noise_model.basis_gates)
        ```
            Args:
                backend: AerBackend object to simulate the experiment. Note that it should carry a noise model to see
                a fidelity decay.

            Returns: XEBJob object containing the information about the experiment (including results)

        """
        from qiskit_aer.backends.aerbackend import AerBackend

        assert isinstance(backend, AerBackend), "The backend should be an AerBackend object"
        num_qubits = len(self.xeb_config.qubits_ids)
        random_gates = len(self.xeb_config.gate_set)
        sq_gates, counts_list, states_list, circuits_list = [], [], [], []
        # Generate sequences
        for s in range(self.xeb_config.seqs):  # For each sequence
            sq_gates.append([])
            circuits_list.append([])
            for i, d in enumerate(self.xeb_config.depths):  # For each maximum depth
                sq_gates[s].append(np.zeros((2, d), dtype=int))
                for q in range(num_qubits):  # For each qubit
                    # Generate random single qubit gates
                    # Start the sequence with a random gate
                    if self.xeb_config.impose_0_cycle:
                        sq_gates[s][i][q][0] = np.random.randint(
                            random_gates - 1 if self.xeb_config.impose_0_cycle else random_gates
                        )
                for d_ in range(1, d):  # For each growing depth (all cycles until maximum depth d)
                    for q in range(num_qubits):  # For each qubit
                        sq_gates[s][i][q][d_] = np.random.randint(random_gates)
                        # Make sure that the same gate is not applied twice in a row
                        while sq_gates[s][i][q][d_] == sq_gates[s][i][q][d_ - 1]:
                            sq_gates[s][i][q][d_] = np.random.randint(random_gates)
                # Define the circuit
                qc = QuantumCircuit(num_qubits)
                # First cycle: apply SW
                if self.xeb_config.impose_0_cycle:
                    for q in range(num_qubits):
                        qc.append(SW, [q])
                if num_qubits == 2 and self.xeb_config.two_qb_gate is not None:
                    qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
                for d_ in range(d):  # Apply layers
                    for q in range(num_qubits):  # For each qubit, append single qubit gates
                        qc.append(self.xeb_config.gate_set[sq_gates[s][i][q][d_]].gate, [q])
                    # Apply CZ gate
                    if num_qubits == 2 and self.xeb_config.two_qb_gate is not None:
                        qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
                qc.save_density_matrix()  # Actual state, subject to noise sim
                circuits_list[s].append(qc)
                # Simulate the circuit
                # Execute circuit (transpiled) and store counts
        circ_list = [
            circuits_list[s][i].measure_all(inplace=False)
            for s in range(self.xeb_config.seqs)
            for i in range(len(self.xeb_config.depths))
        ]
        transpiled_circs = circ_list
        job = backend.run(transpiled_circs, shots=self.xeb_config.n_shots)

        return XEBJob(job, self.xeb_config, self.data_handler, simulate=True)


class XEBJob:
    def __init__(
        self,
        running_job: Union[SimulatedJob, RunningQmJob, AerJob],
        xeb_config: XEBConfig,
        data_handler: DataHandler,
        simulate=False,
    ):
        self.job = running_job
        self._simulate = simulate
        self._result_handles = self.job.result() if isinstance(running_job, AerJob) else self.job.result_handles
        if not isinstance(running_job, AerJob):
            self._result_handles.wait_for_all_values()
        self.xeb_config = xeb_config
        self.data_handler = data_handler
        self._gate_sequences = []
        self._sq_indices = []
        self._circuits = self._get_circuits()

    def _get_circuits(self):
        """
        Returns the circuits generated by the XEB experiment
        Returns:
            List of lists of QuantumCircuit objects representing the circuits generated by the XEB experiment.
            The list is formatted as follows:
            - The first dimension corresponds to the sequence index
            - The second dimension corresponds to the depth index

        """
        circuits = []
        idx = 0
        if self._simulate:
            assert isinstance(self.job, AerJob), "The job should be an AerJob object"
            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for _, _ in enumerate(self.xeb_config.depths):
                    circuits[s].append(self.job.circuits()[idx].remove_final_measurements(inplace=False))
                    circuits[s][-1].data.pop(-1)  # Remove save_density_matrix instruction
                    circuits[s][-1].measure_all(inplace=True)
                    idx += 1

        else:
            g = [self._result_handles.get(f"g{q}").fetch_all()["value"] for q in range(self.xeb_config.n_qubits)]
            for s in range(self.xeb_config.seqs):
                self._sq_indices.append([])
                for i, d in enumerate(self.xeb_config.depths):
                    self._sq_indices[s].append(np.zeros((self.xeb_config.n_qubits, d), dtype=int))
                    for d_ in range(d):
                        for q in range(self.xeb_config.n_qubits):
                            self._sq_indices[s][i][q, d_] = g[q][idx]
                        idx += 1

            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for d_, depth in enumerate(self.xeb_config.depths):
                    qc = QuantumCircuit(self.xeb_config.n_qubits)
                    if self.xeb_config.impose_0_cycle:
                        qc.append(SW, range(self.xeb_config.n_qubits))
                    for d in range(depth):
                        for q in range(self.xeb_config.n_qubits):
                            qc.append(self.xeb_config.gate_set[self._sq_indices[s][d_][q, d]].gate, [q])
                        if self.xeb_config.two_qb_gate is not None:
                            qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
                    qc.measure_all()
                    circuits[s].append(qc)
        return circuits

    def result(self):
        """
        Returns the results of the XEB experiment
        Returns: XEBResult object containing the results of the experiment

        """
        if self._simulate:
            result = self.job.result()
            counts = result.get_counts()
            dms = np.array([result.data(i)["density_matrix"].data for i in range(len(counts))])
            for count in counts:
                for key in [binary(i, self.xeb_config.n_qubits) for i in range(self.xeb_config.dim)]:
                    if key not in count.keys():
                        count[key] = 0
            states = [{f"state{q}": 0.0 for q in range(self.xeb_config.n_qubits)} for _ in range(len(counts))]
            for c, count in enumerate(counts):
                for key, value in count.items():
                    for i, bit in enumerate(reversed(key)):
                        if bit == "1":
                            states[c][f"state{i}"] += value
            states = {
                key: np.reshape(
                    [state[key] / self.xeb_config.n_shots for state in states],
                    (self.xeb_config.seqs, len(self.xeb_config.depths)),
                )
                for key in states[0].keys()
            }

            counts = {
                key: np.reshape([count[key] for count in counts], (self.xeb_config.seqs, len(self.xeb_config.depths)))
                for key in counts[0].keys()
            }

            saved_data = {"counts": counts, "states": states, "density_matrices": dms}
        else:
            quadratures = {
                f"{i}{q}": self._result_handles.get(f"{i}{q}").fetch_all()["value"]
                for i in ["I", "Q"]
                for q in range(self.xeb_config.n_qubits)
            }
            states = {
                f"state{q}": self._result_handles.get(f"state{q}").fetch_all()["value"]
                for q in range(self.xeb_config.n_qubits)
            }
            counts = {
                binary(i, self.xeb_config.n_qubits): self._result_handles.get(
                    f"s{binary(i, self.xeb_config.n_qubits)}"
                ).fetch_all()["value"]
                for i in range(self.xeb_config.dim)
            }

            saved_data = {"quadratures": quadratures, "states": states, "counts": counts}

        if self.xeb_config.should_save_data:
            xeb_config = asdict(self.xeb_config)
            xeb_config["two_qb_gate"] = self.xeb_config.two_qb_gate.gate.name
            if self.xeb_config.should_save_data:
                self.data_handler.save_data(
                    data={
                        "xeb_config": xeb_config,
                        "sq_indices": self._sq_indices,
                        **deepcopy(saved_data),
                    }
                )

        return XEBResult(self.xeb_config, self.circuits, saved_data, self.data_handler)

    @property
    def circuits(self):
        """
        Returns the circuits generated by the XEB experiment
        Circuits are formatted as follows:
        - The first dimension corresponds to the sequence index
        - The second dimension corresponds to the depth index
        Returns:
            List of lists of QuantumCircuit objects representing the circuits generated by the XEB experiment.

        """
        return self._circuits


class XEBResult:

    def __init__(self, xeb_config: XEBConfig, circuits, saved_data, data_handler: DataHandler = None):
        self.xeb_config = xeb_config
        self.circuits: List[List[QuantumCircuit]] = circuits
        self.saved_data = saved_data
        self.data_handler = data_handler
        (
            self.measured_probs,
            self.expected_probs,
            self.records,
            self.log_fidelities,
            self.linear_fidelities,
            self.singularities,
            self.outliers,
        ) = self.retrieve_data()

    def retrieve_data(self):
        """
        Retrieve the data from the XEB experiment

        Returns:
            measured_probs: Measured probabilities of the states
            expected_probs: Expected probabilities of the states
            records: Records of the experiment
            log_fidelities: Logarithmic fidelities
            linear_fidelities: Linear fidelities
            singularities: Singularities
            outliers: Outliers
        """
        dim = 2 ** len(self.xeb_config.qubits_ids)
        n_qubits = len(self.xeb_config.qubits_ids)
        seqs = self.xeb_config.seqs
        depths = self.xeb_config.depths
        counts = self.saved_data["counts"]
        states = self.saved_data["states"]
        if not self.xeb_config.disjoint_processing:
            records = []
            incoherent_distribution = np.ones(dim) / dim
            expected_probs = np.zeros((seqs, len(depths), dim))
            measured_probs = np.zeros((seqs, len(depths), dim))
            log_fidelities = np.zeros((seqs, len(depths)))
            singularity = []
            outlier = []
        else:
            records = [[], []]
            incoherent_distribution = np.ones(2) / 2
            expected_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            measured_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            log_fidelities = np.zeros((n_qubits, seqs, len(depths)))
            singularity = [[], []]
            outlier = [[], []]

        for s in range(seqs):
            for d_, d in enumerate(depths):
                if not self.xeb_config.disjoint_processing:
                    qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
                    expected_probs[s, d_] = np.round(Statevector(qc).probabilities(), 5)  # [1, 0]
                    measured_probs[s, d_] = (
                        np.array([counts[binary(i, n_qubits)][s][d_] for i in range(dim)]) / self.xeb_config.n_shots
                    )

                    xe_incoherent = cross_entropy(incoherent_distribution, expected_probs[s, d_])
                    xe_measured = cross_entropy(measured_probs[s, d_], expected_probs[s, d_])
                    xe_expected = cross_entropy(expected_probs[s, d_], expected_probs[s, d_])

                    f_xeb = (xe_incoherent - xe_measured) / (xe_incoherent - xe_expected)
                    if np.isinf(f_xeb) or np.isnan(f_xeb):
                        singularity.append((s, d_))
                        log_fidelities[s, d_] = np.nan  # Set all singularities to NaN
                    elif f_xeb < 0 or f_xeb > 1:
                        outlier.append(((s, d_), f_xeb))
                        log_fidelities[s, d_] = np.nan  # Set all outliers to NaN
                    else:
                        log_fidelities[s, d_] = f_xeb

                        records += [
                            {
                                "sequence": s,
                                "depth": depths[d_],
                                "pure_probs": expected_probs[s, d_],
                                "sampled_probs": measured_probs[s, d_],
                            }
                        ]

                else:
                    for q in range(n_qubits):
                        qc = self.circuits[s][d_]
                        expected_probs[q, s, d_] = np.round(Statevector(qc).probabilities([q]), 5)
                        measured_probs[q, s, d_] = np.array(
                            [1 - states[f"state{q}"][s][d_], states[f"state{q}"][s][d_]]
                        )

                        xe_incoherent = cross_entropy(incoherent_distribution, expected_probs[q, s, d_])
                        xe_measured = cross_entropy(measured_probs[q, s, d_], expected_probs[q, s, d_])
                        xe_expected = cross_entropy(expected_probs[q, s, d_], expected_probs[q, s, d_])

                        f_xeb = (xe_incoherent - xe_measured) / (xe_incoherent - xe_expected)
                        if np.isinf(f_xeb) or np.isnan(f_xeb):
                            singularity[q].append((s, d_))
                            log_fidelities[q, s, d_] = np.nan
                        elif f_xeb < 0 or f_xeb > 1:
                            outlier[q].append(((s, d_), f_xeb))
                            log_fidelities[q, s, d_] = np.nan
                        else:
                            log_fidelities[q, s, d_] = f_xeb

                            records[q] += [
                                {
                                    "sequence": s,
                                    "depth": depths[d_],
                                    "pure_probs": expected_probs[q, s, d_],
                                    "sampled_probs": measured_probs[q, s, d_],
                                }
                            ]

        def per_cycle_depth(df):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            for record in records:
                e_u = np.sum(record["pure_probs"] ** 2)
                u_u = np.sum(record["pure_probs"]) / dim
                m_u = np.sum(record["pure_probs"] * record["sampled_probs"])
                record.update(e_u=e_u, u_u=u_u, m_u=m_u)
            df = pd.DataFrame(records)
            df["y"] = df["m_u"] - df["u_u"]
            df["x"] = df["e_u"] - df["u_u"]

            df["numerator"] = df["x"] * df["y"]
            df["denominator"] = df["x"] ** 2
            linear_fidelities = df.groupby("depth").apply(per_cycle_depth).reset_index()
        else:
            linear_fidelities = []
            for i in range(n_qubits):
                for record in records[i]:
                    e_u = np.sum(record["pure_probs"] ** 2)
                    u_u = np.sum(record["pure_probs"]) / 2
                    m_u = np.sum(record["pure_probs"] * record["sampled_probs"])
                    record.update(e_u=e_u, u_u=u_u, m_u=m_u)
            df = [pd.DataFrame(record) for record in records]
            for i in range(n_qubits):
                df[i]["y"] = df[i]["m_u"] - df[i]["u_u"]
                df[i]["x"] = df[i]["e_u"] - df[i]["u_u"]

                df[i]["numerator"] = df[i]["x"] * df[i]["y"]
                df[i]["denominator"] = df[i]["x"] ** 2
                linear_fidelities.append(df[i].groupby("depth").apply(per_cycle_depth).reset_index())

        return measured_probs, expected_probs, df, log_fidelities, linear_fidelities, singularity, outlier

    def plot_fidelities(self, fit_linear: bool = True, fit_log_entropy: bool = True):
        """
        Plot the state fidelities for the XEB experiment
        Args:
            fit_linear: Indicate if the linear XEB data should be fitted
            fit_log_entropy: Indicate if the log-entropy XEB data should be fitted
        Returns:

        """
        plt.figure()
        plt.rcParams["text.usetex"] = False
        xx = np.linspace(0, self.linear_fidelities["depth"].max())

        try:  # Fit the data for the linear XEB
            if fit_linear:
                a_lin, layer_fid_lin, a_std_lin, layer_fid_std_lin = fit_exponential_decay(
                    self.linear_fidelities["depth"], self.linear_fidelities["fidelity"]
                )
                plt.plot(
                    xx,
                    exponential_decay(xx, a_lin, layer_fid_lin),
                    label="Fit (Linear XEB), layer_fidelity={:.1f}%".format(layer_fid_lin * 100),
                    color="red",
                )
        except Exception as e:
            raise e
        Fxeb = np.nanmean(self.log_fidelities, axis=0)

        try:  # Fit the data for the log-entropy XEB
            if fit_log_entropy:
                a_log, layer_fid_log, a_std_log, layer_fid_std_log = fit_exponential_decay(self.xeb_config.depths, Fxeb)
                plt.plot(
                    xx,
                    exponential_decay(xx, a_log, layer_fid_log),
                    label="Fit (Log XEB), layer_fidelity={:.1f}%".format(layer_fid_log * 100),
                    color="green",
                )
        except Exception as e:
            print("Fit for Log XEB data failed")
            raise e

        mask_lin = (self.linear_fidelities["fidelity"] > 0) & (self.linear_fidelities["fidelity"] < 1)
        masked_linear_depths = self.linear_fidelities["depth"][mask_lin]
        masked_linear_fids = self.linear_fidelities["fidelity"][mask_lin]
        if fit_linear:
            plt.scatter(masked_linear_depths, masked_linear_fids, color="red", label="Linear XEB")

        mask_log = (Fxeb > 0) & (Fxeb < 1)
        if fit_log_entropy:
            plt.scatter(
                self.xeb_config.depths[mask_log],
                Fxeb[mask_log],
                marker="o",
                color="green",
                label="Log-XEB",
            )

        plt.ylabel("Circuit fidelity", fontsize=20)
        plt.xlabel("Cycle Depth $d$", fontsize=20)
        plt.title("XEB Fidelity")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_records(self):
        """
        Plot the records for the XEB experiment
        Returns:

        """
        depths = self.xeb_config.depths
        n_qubits = self.xeb_config.n_qubits
        colors = sns.cubehelix_palette(n_colors=len(depths))
        colors = {k: colors[i] for i, k in enumerate(depths)}
        _lines = []
        plt.figure()

        def per_cycle_depth(df, _lines=None):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            cycle_depth = df.name
            xx = np.linspace(0, df["x"].max())
            (l,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
            plt.scatter(df["x"], df["y"], color=colors[cycle_depth])
            _lines += [l]  # for legend
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            fids = self.records.groupby("depth").apply(per_cycle_depth, _lines).reset_index()
            plt.xlabel(r"$e_U - u_U$", fontsize=18)
            plt.ylabel(r"$m_U - u_U$", fontsize=18)
            _lines = np.asarray(_lines)
            plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
            plt.title("q-%s: Fxeb_linear = %s" % (self.xeb_config.qubits_ids, [fids["fidelity"][x] for x in [0, 1]]))
            plt.tight_layout()
        else:
            fids = []
            for i in range(n_qubits):
                _lines = []
                plt.figure()
                fids.append(self.records[i].groupby("depth").apply(per_cycle_depth).reset_index())
                plt.xlabel(r"$e_U - u_U$", fontsize=18)
                plt.ylabel(r"$m_U - u_U$", fontsize=18)
                _lines = np.asarray(_lines)
                plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
                plt.title(
                    "q-%s: Fxeb_linear = %s" % (self.xeb_config.qubits_ids[i], [fids[i]["fidelity"][x] for x in [0, 1]])
                )
                plt.show()

    def plot_state_heatmap(self):
        """
        Plot the state heatmap for the XEB experiment
        Returns:

        """

        def create_subplot(data, subplot_number, title):
            plt.pcolor(self.xeb_config.depths, range(self.xeb_config.seqs), np.abs(data))
            ax = plt.gca()
            ax.set_title(title)
            if subplot_number > 244:
                ax.set_xlabel("Circuit depth")
            ax.set_ylabel("Sequences")
            ax.set_xticks(self.xeb_config.depths)
            ax.set_yticks(np.arange(1, self.xeb_config.seqs + 1))
            plt.colorbar()

        titles, data = [], []
        if not self.xeb_config.disjoint_processing:
            for i in range(self.xeb_config.dim):
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Measured")
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Expected")
                data.append(self.measured_probs[:, :, i])
                data.append(self.expected_probs[:, :, i])
        else:
            for i, q in enumerate(self.xeb_config.qubits_ids):
                for j in range(2):
                    titles.append(f"q{q}<{j}> Measured")
                    titles.append(f"q{q}<{j}> Expected")
                    data.append(self.measured_probs[i, :, :, j])
                    data.append(self.expected_probs[i, :, :, j])

        plot_number = [241, 242, 243, 244, 245, 246, 247, 248]

        for title, d, n in zip(titles, data, plot_number):
            plt.suptitle("XEB State Heatmaps")

            create_subplot(d, n, title)
            plt.subplots_adjust(wspace=0.1, hspace=0.7)

            plt.tight_layout()
            plt.show()

    def save(self):
        """
        Save the results of the XEB experiment
        """
        self.data_handler.save_data(
            data={
                "config": asdict(self.xeb_config),
                "fidelities": self.log_fidelities,
                "linear_fidelities": self.linear_fidelities,
                "records": self.records,
                "measured_probs": self.measured_probs,
                "expected_probs": self.expected_probs,
                "singularity": self.singularities,
                "outliers": self.outliers,
            }
        )
