from dataclasses import asdict
from typing import Union, List, Optional, Dict, Tuple
import numpy as np
from qm.qua import *
from .macros import (
    qua_declaration,
    reset_qubit,
    cross_entropy,
    binary,
    exponential_decay,
    fit_exponential_decay,
    get_parallel_gate_combinations as gate_combinations,
    align_transmon,
    align_transmon_pair,
)
import matplotlib.pyplot as plt
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerJob
import pandas as pd
from .xeb_config import XEBConfig
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from quam_libs.components import QuAM, Transmon, TransmonPair
from qm import SimulationConfig
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
import seaborn as sns
from copy import deepcopy
from warnings import warn

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
        self.qubits: List[Transmon] = xeb_config.qubits
        self.qubit_dict: Dict = {i: qubit for i, qubit in enumerate(self.qubits)}
        self.qubit_dict2: Dict = {qubit: i for i, qubit in enumerate(self.qubits)}
        self.qubit_pairs: List[TransmonPair] = xeb_config.qubit_pairs
        if len(self.qubit_pairs) == 0:
            warn("No qubit pairs provided. The experiment will run with single qubit gates only.")

        # Create CouplingMap from QuAM qubit pairs
        coupling_map = CouplingMap()
        for qubit_pair in self.qubit_pairs:
            if qubit_pair.qubit_control not in self.qubits or qubit_pair.qubit_target not in self.qubits:
                raise ValueError("Qubit pairs must be formed by qubits present in the qubits list")
            coupling_map.add_edge(self.qubit_dict2[qubit_pair.qubit_control], self.qubit_dict2[qubit_pair.qubit_target])
        self.coupling_map = coupling_map
        self.available_combinations: List[Tuple[Tuple[int, int]]] = gate_combinations(self.coupling_map)
        try:
            self.qubit_drive_channels = [qubit.xy for qubit in self.qubits]
            self.readout_channels = [qubit.resonator for qubit in self.qubits]
        except AttributeError:
            raise AttributeError(
                "Qubit objects must have 'xy' and 'resonator' attributes, "
                "Contact CS Team if your QuAM structure is different."
            )
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

    def _play_random_sq_gate(self, qubit: Transmon, gate_idx, amp_matrix: Optional[List] = None):
        """
        Play a random single qubit gate on a given qubit element.

        This macro plays a random single qubit gate on a given qubit element, by modulating
        the amplitude matrix of a baseline calibrated X/2 (SX) pulse if the gate set
        is set up to run through amplitude matrix modulation.
        Otherwise, it plays the gate through a switch case over the gate index.

        Args:
            qubit (Transmon): Qubit element on which to play the gate.
            gate_idx (QUA int): Index of the gate to play.
            amp_matrix (List): Amplitude matrix of the gate.
        """
        if self.xeb_config.gate_set.run_through_amp_matrix_modulation and amp_matrix is not None:
            # Play all gates through real-time amplitude matrix modulation
            qubit.xy.play(self.xeb_config.baseline_gate_name, amplitude_scale=amp(*amp_matrix))
        else:
            # Play all gates through switch case over the gate index
            with switch_(gate_idx, unsafe=True):
                for i in range(len(self.xeb_config.gate_set)):
                    with case_(i):
                        self.xeb_config.gate_set[i].gate_macro(qubit)

    def _xeb_prog(self, simulate: bool = False):
        """
        Generate the QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
        Returns: QUA program for the XEB experiment
        """
        n_qubits = self.xeb_config.n_qubits
        dim = self.xeb_config.dim
        random_gates = len(self.xeb_config.gate_set)
        ge_thresholds = [
            readout_element.operations[self.xeb_config.readout_pulse_name].threshold
            for readout_element in self.readout_channels
        ]

        with program() as xeb_prog:
            # Declare QUA variables
            I, I_st, Q, Q_st = qua_declaration(n_qubits=n_qubits, readout_elements=self.readout_channels)
            depth, depth_, n, s, tot_state_ = [declare(int) for _ in range(5)]
            gate = [
                declare(int, size=self.xeb_config.depths[-1]) for _ in range(n_qubits)
            ]  # Gate indices list for both qubits
            two_qubit_gate_pattern = declare(int, value=0)
            if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                amp_matrix = [
                    [declare(fixed, size=self.xeb_config.depths[-1]) for _ in range(4)] for _ in range(n_qubits)
                ]
            counts = declare(int, value=[0] * dim)  # Counts for all possible bit-strings (00, 01, 10, 11)
            state = [declare(bool) for _ in range(n_qubits)]  # Qubit states
            # Declare streams
            counts_st = [
                declare_stream() for _ in range(dim)
            ]  # Stream for counts of all possible bit-strings (00, 01, 10, 11)
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
                for qubit in self.qubit_drive_channels:
                    update_frequency(qubit.name, 0)

            # Generate the random sequences
            with for_(s, 0, s < self.xeb_config.seqs, s + 1):
                # Generate random gate sequence for each qubit
                for q in range(n_qubits):
                    assign(gate[q][0], r.rand_int(random_gates))
                    save(gate[q][0], gate_st[q])
                with for_(depth_, 1, depth_ < self.xeb_config.depths[-1], depth_ + 1):
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
                                gate[q][depth_],
                                [amp_matrix[q][i][depth_] for i in range(4)],
                            )
                            if simulate:
                                for amp_matrix_element in range(4):
                                    save(
                                        amp_matrix[q][amp_matrix_element][depth_],
                                        amp_st[q][amp_matrix_element],
                                    )

                # Run the XEB sequence
                with for_each_(depth, self.xeb_config.depths):  # Truncate depth to each value in depths
                    with for_(n, 0, n < self.xeb_config.n_shots, n + 1):
                        if simulate:
                            wait(25, *[qubit.name for qubit in self.qubit_drive_channels])

                        # Play all cycles generated for sequence s of depth d
                        with for_(depth_, 0, depth_ < depth, depth_ + 1):
                            for q, qubit in enumerate(self.qubits):  # Play single qubit gates on all qubits
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
                            if self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                                if len(self.qubit_pairs) > 1:  # Multi-qubit XEB case
                                    with switch_(two_qubit_gate_pattern):
                                        for i, combination in enumerate(self.available_combinations):
                                            with case_(i):
                                                for pair in combination:
                                                    for ctrl_idx, tgt_idx in pair:
                                                        qubit_ctrl = self.qubit_dict[ctrl_idx]
                                                        qubit_tgt = self.qubit_dict[tgt_idx]
                                                        align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                                        # Two qubit gate macro
                                                        self.xeb_config.two_qb_gate.gate_macro(qubit_ctrl @ qubit_tgt)
                                                        align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                else:  # Two-qubit XEB case (no need for switch case)
                                    qubit_pair = self.qubit_pairs[0]
                                    align_transmon_pair(qubit_pair)
                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                    align_transmon_pair(qubit_pair)

                                with if_(two_qubit_gate_pattern == len(self.available_combinations) - 1):
                                    assign(two_qubit_gate_pattern, 0)
                                with else_():
                                    assign(two_qubit_gate_pattern, two_qubit_gate_pattern + 1)

                        # Measure the state
                        for q_idx, qubit in enumerate(self.qubits):
                            align_transmon(qubit)
                            qubit.resonator.measure(
                                self.xeb_config.readout_pulse_name,
                                qua_vars=(I[q_idx], Q[q_idx]),
                            )
                            # State Estimation: returned as an integer, to be later converted to bit-strings
                            assign(state[q_idx], I[q_idx] > ge_thresholds[q_idx])
                            save(state[q_idx], state_st[q_idx])
                            save(I[q_idx], I_st[q_idx])
                            save(Q[q_idx], Q_st[q_idx])
                            assign(
                                tot_state_,
                                tot_state_ + 2**q_idx * Cast.to_int(state[q_idx]),
                            )

                            reset_qubit(
                                self.xeb_config.reset_method,
                                qubit,
                                threshold=ge_thresholds[q_idx],
                                **self.xeb_config.reset_kwargs,
                            )
                        assign(two_qubit_gate_pattern, 0)

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
                    gate_st[q].buffer(self.xeb_config.depths[-1]).save_all(f"g{q}")
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

        return XEBJob(job, self.xeb_config, self.data_handler, self.available_combinations, False)

    def simulate(self, backend: BackendV2):
        """
            Simulate the XEB experiment: To simulate it, you must provide an AerBackend object
             with a noise model corresponding to your experiments parameters.
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
        num_qubits = len(self.qubits)
        random_gates = len(self.xeb_config.gate_set)
        sq_gates, counts_list, states_list, circuits_list = [], [], [], []
        two_qubit_gate_pattern = 0
        # Generate sequences
        for s in range(self.xeb_config.seqs):  # For each sequence
            circuits_list.append([])
            sq_gates.append(np.zeros((num_qubits, self.xeb_config.depths[-1]), dtype=int))
            for q in range(num_qubits):  # For each qubit
                # Generate random single qubit gates
                # Start the sequence with a random gate
                sq_gates[s][q][0] = np.random.randint(random_gates)
            for d_ in range(1, self.xeb_config.depths[-1]):  # Generate sequences of max_depth, to be truncated later
                for q in range(num_qubits):  # For each qubit
                    sq_gates[s][q][d_] = np.random.randint(random_gates)
                    # Make sure that the same gate is not applied twice in a row
                    while sq_gates[s][q][d_] == sq_gates[s][q][d_ - 1]:
                        sq_gates[s][q][d_] = np.random.randint(random_gates)
            for i, d in enumerate(self.xeb_config.depths):  # For each maximum depth
                # Define the circuit
                qc = QuantumCircuit(num_qubits)
                for d_ in range(d):  # Apply layers
                    for q in range(num_qubits):  # For each qubit, append single qubit gates
                        qc.append(self.xeb_config.gate_set[sq_gates[s][q][d_]].gate, [q])
                    qc.barrier()
                    # Apply two-qubit gates
                    if num_qubits > 2 and self.xeb_config.two_qb_gate is not None and len(self.qubit_pairs) > 0:
                        for i, combination in enumerate(self.available_combinations):
                            if i == two_qubit_gate_pattern:
                                for pair in combination:
                                    qc.append(self.xeb_config.two_qb_gate.gate, pair)
                                qc.barrier()
                                break
                        if two_qubit_gate_pattern == len(self.available_combinations) - 1:
                            two_qubit_gate_pattern = 0
                        else:
                            two_qubit_gate_pattern += 1
                        # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])

                two_qubit_gate_pattern = 0
                qc.save_density_matrix()  # Actual state, subject to noise simulation
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

        return XEBJob(job, self.xeb_config, self.data_handler, self.available_combinations, True)


class XEBJob:
    def __init__(
        self,
        running_job: Union[SimulatedJob, RunningQmJob, AerJob],
        xeb_config: XEBConfig,
        data_handler: DataHandler,
        available_combinations: List[Tuple[Tuple[int, int]]],
        simulate=False,
    ):
        self.job = running_job
        self.available_combinations = available_combinations
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
        if self._simulate:
            assert isinstance(self.job, AerJob), "The job should be an AerJob object"
            idx = 0
            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for _ in range(len(self.xeb_config.depths)):
                    circuits[s].append(self.job.circuits()[idx].remove_final_measurements(inplace=False))
                    circuits[s][-1].data.pop(-1)  # Remove save_density_matrix instruction
                    circuits[s][-1].measure_all(inplace=True)
                    idx += 1

        else:
            max_depth = self.xeb_config.depths[-1]
            n_qubits = self.xeb_config.n_qubits
            g = [self._result_handles.get(f"g{q}").fetch_all()["value"] for q in range(n_qubits)]
            two_qubit_gate_pattern = 0

            for s in range(self.xeb_config.seqs):
                self._sq_indices.append(np.zeros((n_qubits, max_depth), dtype=int))
                for d in range(max_depth):
                    for q in range(n_qubits):
                        self._sq_indices[s][q, d] = g[q][s, d]

            for s in range(self.xeb_config.seqs):
                circuits.append([])
                for d_, depth in enumerate(self.xeb_config.depths):
                    qc = QuantumCircuit(n_qubits)
                    for d in range(depth):
                        for q in range(n_qubits):
                            sq_gate = self.xeb_config.gate_set[self._sq_indices[s][q, d]].gate
                            qc.append(sq_gate, [q])
                        qc.barrier()
                        if self.xeb_config.two_qb_gate is not None:
                            for i, combination in enumerate(self.available_combinations):
                                if i == two_qubit_gate_pattern:
                                    for pair in combination:
                                        qc.append(self.xeb_config.two_qb_gate.gate, pair)
                                    qc.barrier()
                                    break
                            if two_qubit_gate_pattern == len(self.available_combinations) - 1:
                                two_qubit_gate_pattern = 0
                            else:
                                two_qubit_gate_pattern += 1

                            # qc.append(self.xeb_config.two_qb_gate.gate, [0, 1])
                    qc.measure_all()
                    circuits[s].append(qc)
                    two_qubit_gate_pattern = 0
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
                key: np.reshape(
                    [count[key] for count in counts],
                    (self.xeb_config.seqs, len(self.xeb_config.depths)),
                )
                for key in counts[0].keys()
            }

            saved_data = {"counts": counts, "states": states, "density_matrices": dms}
        else:
            quadratures = {
                f"{i}_{q}": self._result_handles.get(f"{i}_{q}").fetch_all()["value"]
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

            saved_data = {
                "quadratures": quadratures,
                "states": states,
                "counts": counts,
            }

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
    def __init__(
        self,
        xeb_config: XEBConfig,
        circuits,
        saved_data,
        data_handler: DataHandler = None,
    ):
        self.xeb_config = xeb_config
        self.circuits: List[List[QuantumCircuit]] = circuits
        self.saved_data = saved_data
        self.data_handler = data_handler
        (
            self._measured_probs,
            self._expected_probs,
            self._records,
            self._log_fidelities,
            self._linear_fidelities,
            self._singularities,
            self._outliers,
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
        dim = 2 ** len(self.xeb_config.qubits)
        n_qubits = len(self.xeb_config.qubits)
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
            records = [[] for _ in range(n_qubits)]
            incoherent_distribution = np.ones(2) / 2
            expected_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            measured_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            log_fidelities = np.zeros((n_qubits, seqs, len(depths)))
            singularity = [[] for _ in range(n_qubits)]
            outlier = [[] for _ in range(n_qubits)]

        for s in range(seqs):
            for d_, d in enumerate(depths):
                if not self.xeb_config.disjoint_processing:
                    qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
                    expected_probs[s, d_] = np.round(Statevector(qc).probabilities(), 5)
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
                        qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
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
            df = []
            for q in range(n_qubits):
                for record in records[q]:
                    e_u = np.sum(record["pure_probs"] ** 2)
                    u_u = np.sum(record["pure_probs"]) / 2
                    m_u = np.sum(record["pure_probs"] * record["sampled_probs"])
                    record.update(e_u=e_u, u_u=u_u, m_u=m_u)
                df_q = pd.DataFrame(records[q])
                df_q["y"] = df_q["m_u"] - df_q["u_u"]
                df_q["x"] = df_q["e_u"] - df_q["u_u"]

                df_q["numerator"] = df_q["x"] * df_q["y"]
                df_q["denominator"] = df_q["x"] ** 2
                linear_fidelities.append(df_q.groupby("depth").apply(per_cycle_depth).reset_index())
                df.append(df_q)

        return (
            measured_probs,
            expected_probs,
            df,
            log_fidelities,
            linear_fidelities,
            singularity,
            outlier,
        )

    def plot_fidelities(self, fit_linear: bool = True, fit_log_entropy: bool = True):
        """
        Plot the cross-entropy fidelities for the XEB experiment
        Args:
            fit_linear: Indicate if the linear XEB data should be fitted
            fit_log_entropy: Indicate if the log-entropy XEB data should be fitted
        Returns:
        """

        plt.rcParams["text.usetex"] = False

        if self.xeb_config.disjoint_processing:
            for q in range(len(self.xeb_config.qubits)):
                linear_fidelities = self._linear_fidelities[q]
                xx = np.linspace(0, linear_fidelities["depth"].max())
                try:  # Fit the data for the linear XEB
                    if fit_linear:
                        (
                            a_lin,
                            layer_fid_lin,
                            a_std_lin,
                            layer_fid_std_lin,
                        ) = fit_exponential_decay(linear_fidelities["depth"], linear_fidelities["fidelity"])
                        plt.plot(
                            xx,
                            exponential_decay(xx, a_lin, layer_fid_lin),
                            label=f"Fit (Linear XEB Qubit {q}), layer_fidelity={layer_fid_lin * 100:.1f}%",
                        )
                except Exception as e:
                    raise e

                Fxeb = np.nanmean(self.log_fidelities[q], axis=0)
                try:  # Fit the data for the log-entropy XEB
                    if fit_log_entropy:
                        (
                            a_log,
                            layer_fid_log,
                            a_std_log,
                            layer_fid_std_log,
                        ) = fit_exponential_decay(self.xeb_config.depths, Fxeb)
                        plt.plot(
                            xx,
                            exponential_decay(xx, a_log, layer_fid_log),
                            label=f"Fit (Log XEB Qubit {q}), layer_fidelity={layer_fid_log * 100:.1f}%",
                        )
                except Exception as e:
                    print("Fit for Log XEB data failed")
                    raise e

                mask_lin = (linear_fidelities["fidelity"] > 0) & (linear_fidelities["fidelity"] < 1)
                masked_linear_depths = linear_fidelities["depth"][mask_lin]
                masked_linear_fids = linear_fidelities["fidelity"][mask_lin]
                if fit_linear:
                    label = f"Linear XEB Data Qubit {q}"
                    plt.scatter(masked_linear_depths, masked_linear_fids, label=label)

                mask_log = (Fxeb > 0) & (Fxeb < 1)
                if fit_log_entropy:
                    plt.scatter(
                        self.xeb_config.depths[mask_log],
                        Fxeb[mask_log],
                        label=f"Log XEB Data Qubit {q}",
                    )

        else:
            xx = np.linspace(0, self.linear_fidelities["depth"].max())
            try:  # Fit the data for the linear XEB
                if fit_linear:
                    (
                        a_lin,
                        layer_fid_lin,
                        a_std_lin,
                        layer_fid_std_lin,
                    ) = fit_exponential_decay(self.linear_fidelities["depth"], self.linear_fidelities["fidelity"])
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
                    (
                        a_log,
                        layer_fid_log,
                        a_std_log,
                        layer_fid_std_log,
                    ) = fit_exponential_decay(self.xeb_config.depths, Fxeb)
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
                plt.scatter(masked_linear_depths, masked_linear_fids, label="Linear XEB Data", color="blue")

            mask_log = (Fxeb > 0) & (Fxeb < 1)
            if fit_log_entropy:
                plt.scatter(self.xeb_config.depths[mask_log], Fxeb[mask_log], label="Log XEB Data", color="orange")

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

        def per_cycle_depth(df, _lines=None):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            cycle_depth = df.name
            xx = np.linspace(0, df["x"].max())
            (l,) = plt.plot(xx, fid_lsq * xx, color=colors[cycle_depth])
            plt.scatter(df["x"], df["y"], color=colors[cycle_depth])
            _lines += [l]  # for legend
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            plt.figure()
            fids = self.records.groupby("depth").apply(per_cycle_depth, _lines).reset_index()
            plt.xlabel(r"$e_U - u_U$", fontsize=18)
            plt.ylabel(r"$m_U - u_U$", fontsize=18)
            _lines = np.asarray(_lines)
            plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
            title = "Fxeb_linear = %s" % [fids["fidelity"][x] for x in [0, 1]]
            plt.title(title)
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
                    "q-%s: Fxeb_linear = %s"
                    % (
                        self.xeb_config.qubits_ids[i],
                        [fids[i]["fidelity"][x] for x in [0, 1]],
                    )
                )
                plt.show()

    def plot_state_heatmap(self):
        """
        Plot a comparison between expected and actual probability distributions for all sequences.

        This method creates subplots within a single figure to display the state heatmaps. If the number of plots exceeds
        the subplot capacity, it generates a new figure containing the remaining subplots.

        The method handles both disjoint and non-disjoint processing cases.

        Returns:
            None
        """
        titles, data = [], []
        if not self.xeb_config.disjoint_processing:
            for i in range(self.xeb_config.dim):
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Measured")
                titles.append(f"<{binary(i, self.xeb_config.n_qubits)}> Expected")
                data.append(self.measured_probs[:, :, i])
                data.append(self.expected_probs[:, :, i])
        else:
            for i, q in enumerate(self.xeb_config.qubits):
                for j in range(2):
                    titles.append(f"q{q}<{j}> Measured")
                    titles.append(f"q{q}<{j}> Expected")
                    data.append(self.measured_probs[i, :, :, j])
                    data.append(self.expected_probs[i, :, :, j])

        num_plots = len(titles)
        plots_per_fig = 4  # Adjust this value based on the desired grid size (e.g., 2x2 grid)
        num_figs = (num_plots + plots_per_fig - 1) // plots_per_fig

        for fig_idx in range(num_figs):
            fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust the grid size as needed
            axs = axs.flatten()
            start_idx = fig_idx * plots_per_fig
            end_idx = min(start_idx + plots_per_fig, num_plots)

            for plot_idx in range(start_idx, end_idx):
                ax = axs[plot_idx - start_idx]
                ax.pcolor(self.xeb_config.depths, range(self.xeb_config.seqs), np.abs(data[plot_idx]))
                ax.set_title(titles[plot_idx])
                ax.set_xlabel("Circuit depth")
                ax.set_ylabel("Sequences")
                ax.set_xticks(self.xeb_config.depths)
                ax.set_yticks(np.arange(1, self.xeb_config.seqs + 1))
                fig.colorbar(
                    ax.pcolor(self.xeb_config.depths, range(self.xeb_config.seqs), np.abs(data[plot_idx])), ax=ax
                )

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

    @property
    def measured_probs(self):
        """
        Measured probabilities of the states
        Returns: Measured probabilities of the states
        """
        return self._measured_probs

    @property
    def expected_probs(self):
        """
        Expected probabilities of the states
        Returns: Expected probabilities of the states
        """
        return self._expected_probs

    @property
    def records(self):
        """
        Records of the experiment
        Returns: Records of the experiment
        """
        return self._records

    @property
    def log_fidelities(self):
        """
        Logarithmic fidelities
        Returns: Logarithmic fidelities
        """
        return self._log_fidelities

    @property
    def linear_fidelities(self):
        """
        Linear fidelities
        Returns: Linear fidelities
        """
        return self._linear_fidelities

    @property
    def singularities(self):
        """
        Singularities
        Returns: Singularities
        """
        return self._singularities

    @property
    def outliers(self):
        """
        Outliers
        Returns: Outliers
        """
        return self._outliers

    @property
    def purities(self):
        """
        Estimated purities of final states, computed from the variance of the measured probabilities
        Returns: Purities
        """
        var_pt = (2**self.xeb_config.n_qubits - 1) / (
            2 ** (2 * self.xeb_config.n_qubits)(2**self.xeb_config.n_qubits + 1)
        )
        purities = np.var(self.measured_probs, axis=-1) / var_pt
        return purities
