import json
import warnings
from typing import Union, List, Optional, Dict, Tuple, Literal
import numpy as np
from qm.qua import *
from .macros import (
    qua_declaration,
    reset_qubit,
    binary,
    exponential_decay,
    fit_exponential_decay,
    get_parallel_gate_combinations as gate_combinations,
    align_transmon_pair,
    generate_circuits,
    compute_log_fidelity,
    evaluate_log_fidelity,
    update_record,
    update_data_frame,
)
import matplotlib.pyplot as plt
from qiskit_aer import AerJob
import pandas as pd
from .xeb_config import XEBConfig
from qiskit.circuit import QuantumCircuit
from qiskit.providers import BackendV2
from qiskit.transpiler import CouplingMap
from qiskit.quantum_info import Statevector
from qualang_tools.results import DataHandler

from quam_libs.components import QuAM, Transmon
from qm import SimulationConfig, QuantumMachinesManager, generate_qua_script
from qm.jobs.running_qm_job import RunningQmJob
from qm.jobs.simulated_job import SimulatedJob
import seaborn as sns
from warnings import warn

from qualang_tools.units import unit

u = unit(coerce_to_integer=True)


class XEB:
    def __init__(self, xeb_config: XEBConfig, machine: QuAM):
        """
        Initialize the XEB experiment
        Args:
            xeb_config: XEBConfig object containing the parameters of the experiment
            machine: Machine object containing the Quantum Machine configuration
        """
        self.xeb_config = xeb_config
        self.machine = machine
        self.qubit_dict: Dict = {i: qubit for i, qubit in enumerate(self.qubits)}

        if len(self.qubit_pairs) == 0:
            warn("No qubit pairs provided. The experiment will run with single qubit gates only.")

        try:
            self.qubit_drive_channels = [qubit.xy for qubit in self.qubits]
            self.readout_channels = [qubit.resonator for qubit in self.qubits]
        except AttributeError:
            raise AttributeError(
                "Qubit objects must have 'xy' and 'resonator' attributes, "
                "Contact CS Team if your QuAM structure is different."
            )

        # Create CouplingMap from QuAM qubit pairs
        qubit_dict = {qubit: i for i, qubit in enumerate(self.qubits)}
        coupling_map = CouplingMap()
        for qubit in range(len(self.qubits)):
            coupling_map.add_physical_qubit(qubit)
        for qubit_pair in self.qubit_pairs:
            if qubit_pair.qubit_control not in self.qubits or qubit_pair.qubit_target not in self.qubits:
                raise ValueError("Qubit pairs must be formed by qubits present in the qubits list")
            coupling_map.add_edge(qubit_dict[qubit_pair.qubit_control], qubit_dict[qubit_pair.qubit_target])
        self._coupling_map = coupling_map
        self._available_combinations = gate_combinations(self.coupling_map)
        self.xeb_config.available_combinations = self.available_combinations
        self.xeb_config.coupling_map = self.coupling_map
        self.data_handler = DataHandler(name="XEB", root_data_folder=xeb_config.save_dir)

    @property
    def qubit_pairs(self):
        """
        Returns the qubit pairs for the XEB experiment
        """
        return self.xeb_config.qubit_pairs

    @property
    def qubits(self):
        """
        Returns the qubits for the XEB experiment
        """
        return self.xeb_config.qubits

    @property
    def readout_qubits(self):
        """
        Returns the readout qubits for the XEB experiment
        """
        return self.xeb_config.readout_qubits

    @property
    def available_combinations(self):
        """
        Returns the available combinations of qubit pairs for the XEB experiment
        """
        return self._available_combinations

    @property
    def coupling_map(self) -> CouplingMap:
        """
        Returns the coupling map for the XEB experiment
        """
        return self._coupling_map

    def _assign_amplitude_matrix(self, gate_idx, amp_matrix, amp_stream=None):
        """
        Assign the amplitude matrix of a gate based on the gate index

        Args:
            gate_idx (QUA int): Index of the gate
            amp_matrix (List): Amplitude matrix of the gate
            amp_stream (QUA stream): Stream to save the amplitude matrix
        """
        with switch_(gate_idx):
            for i in range(len(self.xeb_config.gate_set)):
                with case_(i):
                    for j in range(4):
                        assign(amp_matrix[j], self.xeb_config.gate_set[i].amp_matrix[j])
        if amp_stream is not None:  # Save the amplitude matrix to a stream
            for j in range(4):
                save(
                    amp_matrix[j],
                    amp_stream,
                )

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
            # qubit.xy.play('x180') # NOTE: for debugging purposes
            # qubit.xy.play('x90') # NOTE: for debugging purposes
            # pass # NOTE: for debugging purposes
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
            amp_st = [declare_stream() for _ in range(n_qubits)]  # Stream for amplitude matrices
            # Bring the active qubits to the idle points:
            self.machine.apply_all_flux_to_min()
            self.machine.apply_all_couplers_to_min()
            # for q, qubit in enumerate(self.qubits):
            #     qubit.z.to_independent_idle()

            # Setting seed for reproducibility
            r = Random(seed=self.xeb_config.seed)

            # If simulating, update the frequency to 0 to visualize sequence
            if simulate:
                for qubit in self.qubit_drive_channels:
                    update_frequency(qubit.name, 0)

            # Generate the random sequences
            with for_(s, 0, s < self.xeb_config.seqs, s + 1):
                # Generate random gate sequence for each qubit
                for q in range(n_qubits):
                    # Iteration 0: Start the sequence with a random gate
                    assign(gate[q][0], r.rand_int(random_gates))
                    save(gate[q][0], gate_st[q])

                    # Map indices into amplitude matrix arguments
                    # (each index corresponds to a random gate)

                    if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                        self._assign_amplitude_matrix(
                            gate[q][0],
                            [amp_matrix[q][i][0] for i in range(4)],
                            amp_st[q],
                        )

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
                                amp_st[q],
                            )

                # Run the XEB sequence
                with for_each_(depth, self.xeb_config.depths):  # Truncate depth to each value in depths
                    with for_(n, 0, n < self.xeb_config.n_shots, n + 1):
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
                                                    ctrl_idx, tgt_idx = pair
                                                    # qubit_ctrl = self.qubit_dict[ctrl_idx]
                                                    # qubit_tgt = self.qubit_dict[tgt_idx]
                                                    if tgt_idx < ctrl_idx:
                                                        ctrl_idx, tgt_idx = tgt_idx, ctrl_idx
                                                    qubit_pair = self.machine.qubit_pairs["coupler_q{}_q{}".format(ctrl_idx+1, tgt_idx+1)]
                                                    # align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                                    align_transmon_pair(qubit_pair)
                                                    # Two qubit gate macro
                                                    # self.xeb_config.two_qb_gate.gate_macro(qubit_ctrl @ qubit_tgt)
                                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                                    # align_transmon_pair(qubit_ctrl @ qubit_tgt)
                                                    align_transmon_pair(qubit_pair)

                                    with if_(two_qubit_gate_pattern == len(self.available_combinations) - 1):
                                        assign(two_qubit_gate_pattern, 0)
                                    with else_():
                                        assign(two_qubit_gate_pattern, two_qubit_gate_pattern + 1)
                                else:  # Two-qubit XEB case (no need for switch case)
                                    qubit_pair = self.qubit_pairs[0]
                                    # align_transmon_pair(qubit_pair)
                                    self.xeb_config.two_qb_gate.gate_macro(qubit_pair)
                                    # align_transmon_pair(qubit_pair)

                        # Measure the state
                        wait(150 * u.ns)
                        align()
                        for q_idx, qubit in enumerate(self.qubits):
                            # Play the readout on the other resonator to measure in the same condition as when optimizing readout
                            for other_qubit in self.readout_qubits:
                                if other_qubit.resonator != qubit.resonator:
                                    other_qubit.resonator.play("readout")

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
                        if len(self.qubit_pairs) > 1:  # Multi-qubit XEB case (Reset pattern)
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
                    amp_st[q].buffer(self.xeb_config.depths[-1], 2, 2).save_all(f"amp_matrix_q{q}")
                for i in range(dim):
                    string = "s" + binary(i, n_qubits)
                    counts_st[i].buffer(len(self.xeb_config.depths)).save_all(string)

        return xeb_prog

    def run(
        self,
        simulate: bool = False,
        simulation_config: Optional[SimulationConfig] = None,
        qmm_cloud_simulator: Optional[QuantumMachinesManager] = None,
        **simulate_kwargs,
    ):
        """
        Run QUA program for the XEB experiment
        Args:
            simulate: Indicate if output should be simulated or not
            simulation_config: SimulationConfig object containing the parameters of the simulation
            qmm_cloud_simulator: QuantumMachinesManager object to simulate the experiment
            simulate_kwargs: Optional additional keyword arguments passed to `qm.simulate`

        Returns: XEBJob object containing the information about the experiment (including results)

        """
        # Compile the QUA program

        config = self.machine.generate_config()
        if simulation_config is None:
            simulation_config = SimulationConfig(duration=10_000)
        xeb_prog = self._xeb_prog(simulate=simulate)  # set simulate=True to get the amplitude matrix
        if simulate and qmm_cloud_simulator is not None:
            qmm = qmm_cloud_simulator
        else:
            qmm = self.machine.connect()
        qm = qmm.open_qm(config)
        if simulate:
            with open("debug.py", "w+") as f:
                f.write(generate_qua_script(xeb_prog, config))
            job = qm.simulate(xeb_prog, simulate=simulation_config, **simulate_kwargs)
        elif self.xeb_config.generate_new_data:
            job = qm.execute(xeb_prog)
        else:
            warnings.warn("Running deactivated. Set generate_new_data to True to run the experiment."
                          "Use XEBResult.from_data() method to load data from a previous run.")
            return 

        return XEBJob(job, self.xeb_config, self.data_handler, self.available_combinations, False, simulate)

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
        hardware_simulate=False,
    ):
        self.job = running_job
        self.available_combinations = available_combinations
        self._simulate = simulate
        self._hardware_simulate = hardware_simulate
        self._result_handles = self.job.result() if isinstance(running_job, AerJob) else self.job.result_handles
        if not isinstance(running_job, AerJob):
            self._result_handles.wait_for_all_values()
        self.xeb_config = xeb_config
        self.data_handler = data_handler
        self._gate_indices = np.zeros(
            (self.xeb_config.seqs, self.xeb_config.n_qubits, self.xeb_config.depths[-1]), dtype=int
        )
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

            for s in range(self.xeb_config.seqs):
                for d in range(max_depth):
                    for q in range(n_qubits):
                        self._gate_indices[s, q, d] = g[q][s, d]

            circuits = generate_circuits(self.xeb_config, self.gate_indices, self.available_combinations)
        return circuits

    def result(self, disjoint_processing: Optional[bool] = None):
        """
        Returns the results of the XEB experiment

        Args:
            disjoint_processing: Indicate if disjoint processing should be applied to the results
        Returns: XEBResult object containing the results of the experiment

        """

        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            self.xeb_config.disjoint_processing = disjoint_processing
        if self._simulate:
            result = self.job.result()
            counts = result.get_counts()
            dms = np.array([result.data(i)["density_matrix"].data for i in range(len(counts))])
            for count in counts:  # Fill in missing bit-strings with 0 counts
                for key in [binary(i, self.xeb_config.n_qubits) for i in range(self.xeb_config.dim)]:
                    if key not in count.keys():
                        count[key] = 0
            states = [{f"state_{qubit.name}": 0.0 for qubit in self.xeb_config.qubits} for _ in range(len(counts))]
            for c, count in enumerate(counts):
                for key, value in count.items():
                    for i, bit in enumerate(reversed(key)):
                        if bit == "1":
                            states[c][f"state_{self.xeb_config.qubits[i].name}"] += value
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

            gate_indices, states, counts, quadratures, amp_st = {}, {}, {}, {}, {}
            result = self._result_handles
            for q, qubit in enumerate(self.xeb_config.qubits):
                gate_indices[f"g_{qubit.name}"] = result.get(f"g{q}").fetch_all()["value"]
                quadratures[f"I_{qubit.name}"] = result.get(f"I{q}").fetch_all()["value"]
                quadratures[f"Q_{qubit.name}"] = result.get(f"Q{q}").fetch_all()["value"]
                states[f"state_{qubit.name}"] = result.get(f"state{q}").fetch_all()["value"]
                if self.xeb_config.gate_set.run_through_amp_matrix_modulation:
                    amp_st[f"amp_matrix_{qubit.name}"] = result.get(f"amp_matrix_q{q}").fetch_all()["value"]

            n_qubits = self.xeb_config.n_qubits
            for i in range(self.xeb_config.dim):
                counts[binary(i, n_qubits)] = result.get(f"s{binary(i, n_qubits)}").fetch_all()["value"]

            saved_data = {
                **quadratures,
                **states,
                **counts,
                **amp_st,
                "gate_indices": self.gate_indices,
                "gate_sequences": self.gate_sequences,
            }

        return XEBResult(
            self.xeb_config,
            self.circuits,
            counts,
            states,
            saved_data,
            self.data_handler if self.xeb_config.should_save_data else None,
        )

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

    @property
    def simulate(self):
        return self._simulate

    @property
    def hardware_simulate(self):
        return self._hardware_simulate

    @property
    def gate_indices(self):
        """
        Returns the gate indices of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        return self._gate_indices

    @property
    def gate_sequences(self):
        """
        Returns the gate sequences of the XEB experiment in the form of a 3D numpy array (sequence, qubit, depth)
        """
        gate_sequences = np.zeros(
            (self.xeb_config.seqs, self.xeb_config.n_qubits, self.xeb_config.depths[-1]), dtype=str
        )
        for s in range(self.xeb_config.seqs):
            for q in range(self.xeb_config.n_qubits):
                for d in range(self.xeb_config.depths[-1]):
                    gate_sequences[s, q, d] = self.xeb_config.gate_set[self.gate_indices[s, q, d]].name

        return gate_sequences

    def plot_simulated_samples(self):
        if self.hardware_simulate:
            samples = self.job.get_simulated_samples()
            plt.subplots(nrows=len(samples.keys()), sharex=True)
            for i, con in enumerate(samples.keys()):
                plt.subplot(len(samples.keys()), 1, i + 1)
                samples[con].plot()
                plt.title(con)
            plt.tight_layout()
            plt.show()
        else:
            warnings.warn("Simulated samples are not available because the job was run and not hardware-simulated.")


class XEBResult:
    def __init__(
        self, xeb_config: XEBConfig, circuits, counts: Dict, states: Dict, saved_data, data_handler: DataHandler = None
    ):
        self.xeb_config = xeb_config
        self.circuits: List[List[QuantumCircuit]] = circuits
        self.counts = counts
        self.states = states
        self.data = saved_data
        self.data_handler = data_handler
        (
            self._joint_measured_probs,
            self._disjoint_measured_probs,
            self._joint_expected_probs,
            self._disjoint_expected_probs,
            self._records,
            self._log_fidelities,
            self._linear_fidelities,
            self._singularities,
            self._outliers,
        ) = self.retrieve_data()

        self.data.update(
            {
                "joint_measured_probs": self._joint_measured_probs,
                "disjoint_measured_probs": self._disjoint_measured_probs,
                "joint_expected_probs": self._joint_expected_probs,
                "disjoint_expected_probs": self._disjoint_expected_probs,
                "log_fidelities": self._log_fidelities,
                "linear_fidelities": (
                    np.array(self.linear_fidelities["fidelity"])
                    if not self.xeb_config.disjoint_processing
                    else np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])
                ),
                "singularities": self._singularities,
                "outliers": self._outliers,
            }
        )

        if self.xeb_config.should_save_data and self.data_handler is not None:
            save_data = {}
            for key in save_data.keys(): # Remove the amplitude matrices from the saved data
                if 'amp_matrix' in key:
                    continue
                save_data[key] = self.data[key]
            self.data_handler.save_data(saved_data,
                                        self.xeb_config.data_folder_name,
                                        metadata=self.xeb_config.as_dict())

    @classmethod
    def from_data(
        cls, directory: str, disjoint_processing: Optional[bool] = None, data_handler: Optional[DataHandler] = None
    ):
        """
        Retrieve the XEBResult object from a saved data file (JSON format)

        Args:
            directory: Directory of the saved data files (should contain data.json and node.json)
            disjoint_processing: Indicate if disjoint processing should be applied to the results
            data_handler: DataHandler object to handle the data
        """
        data: Dict = json.load(open(directory + "/data.json", "r"))
        arrays: Dict = np.load(directory + "/arrays.npz")
        metadata: Dict = json.load(open(directory + "/node.json", "r"))
        xeb_config = XEBConfig.from_dict(metadata["metadata"])
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
            xeb_config.disjoint_processing = disjoint_processing
        gate_indices = arrays["gate_indices"]
        circuits = generate_circuits(xeb_config, gate_indices, xeb_config.available_combinations)

        new_data = {"states": {}, "counts": {}, "quadratures": {}, "amp_st": {}}

        for key, value in data.items():
            if "state" in key:
                new_data["states"][key] = arrays[key]
            elif key.isnumeric():
                new_data["counts"][key] = arrays[key]
            elif "amp_matrix" in key:
                new_data["amp_st"][key] = arrays[key]
            elif key.startswith("I") or key.startswith("Q"):
                new_data["quadratures"][key] = arrays[key]
            else:
                if key in arrays:
                    new_data[key] = arrays[key]
                else:
                    new_data[key] = value

        return cls(xeb_config, circuits, new_data["counts"], new_data["states"], new_data, data_handler)

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
        counts = self.counts
        states = self.states

        existing_data = "joint_expected_probs" in self.data.keys()

        if not existing_data:
            joint_expected_probs = np.zeros((seqs, len(depths), dim))
            joint_measured_probs = np.zeros((seqs, len(depths), dim))

            disjoint_expected_probs = np.zeros((n_qubits, seqs, len(depths), 2))
            disjoint_measured_probs = np.zeros((n_qubits, seqs, len(depths), 2))
        else:
            joint_expected_probs = self.data["joint_expected_probs"]
            joint_measured_probs = self.data["joint_measured_probs"]

            disjoint_expected_probs = self.data["disjoint_expected_probs"]
            disjoint_measured_probs = self.data["disjoint_measured_probs"]

        if not self.xeb_config.disjoint_processing:
            records, singularity, outlier = [], [], []
            incoherent_distribution = np.ones(dim) / dim
            log_fidelities = np.zeros((seqs, len(depths)))

        else:
            records = [[] for _ in range(n_qubits)]
            singularity = [[] for _ in range(n_qubits)]
            outlier = [[] for _ in range(n_qubits)]
            incoherent_distribution = np.ones(2) / 2
            log_fidelities = np.zeros((n_qubits, seqs, len(depths)))

        for s in range(seqs):
            for d_, depth in enumerate(depths):
                qc = self.circuits[s][d_].remove_final_measurements(inplace=False)
                if not existing_data:
                    statevector = Statevector(qc)
                    joint_expected_probs[s, d_] = statevector.probabilities(decimals=5)
                    joint_measured_probs[s, d_] = (
                        np.array([counts[binary(i, n_qubits)][s][d_] for i in range(dim)]) / self.xeb_config.n_shots
                    )

                    for q in range(n_qubits):
                        disjoint_expected_probs[q, s, d_] = statevector.probabilities([q], 5)
                        qubit_state = states[f"state_{self.qubit_names[q]}"][s, d_]
                        disjoint_measured_probs[q, s, d_] = np.array([1 - qubit_state, qubit_state])

                if not self.xeb_config.disjoint_processing:
                    # Calculate the cross-entropy fidelities (logarithmic)
                    f_xeb = compute_log_fidelity(
                        incoherent_distribution, joint_expected_probs[s, d_], joint_measured_probs[s, d_]
                    )
                    log_fidelities[s, d_] = evaluate_log_fidelity(f_xeb, singularity, outlier, s, int(depth))

                    # Store records for linear XEB post-processing
                    records = update_record(
                        records, s, depth, joint_expected_probs[s, d_], joint_measured_probs[s, d_], dim
                    )

                else:
                    for q, qubit_name in enumerate(self.qubit_names):
                        # Calculate the cross-entropy fidelities (logarithmic)
                        f_xeb = compute_log_fidelity(
                            incoherent_distribution,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                        )
                        log_fidelities[q, s, d_] = evaluate_log_fidelity(f_xeb, singularity[q], outlier[q], 
                                                                         s, int(depth))
                        # Store records for linear XEB post-processing
                        records[q] = update_record(
                            records[q],
                            s,
                            depth,
                            disjoint_expected_probs[q, s, d_],
                            disjoint_measured_probs[q, s, d_],
                            2,
                        )

        def per_cycle_depth(df):
            fid_lsq = df["numerator"].sum() / df["denominator"].sum()
            return pd.Series({"fidelity": fid_lsq})

        if not self.xeb_config.disjoint_processing:
            df = update_data_frame(pd.DataFrame(records))
            linear_fidelities = df.groupby("depth").apply(per_cycle_depth).reset_index()
        else:
            df, linear_fidelities = [], []
            for q in range(n_qubits):
                df_q = update_data_frame(pd.DataFrame(records[q]))
                linear_fidelities.append(df_q.groupby("depth").apply(per_cycle_depth).reset_index())
                df.append(df_q)

        if np.isnan(log_fidelities).all():
            warnings.warn("All fidelities computed from log-entropies are singularities.")

        return (
            joint_measured_probs,
            disjoint_measured_probs,
            joint_expected_probs,
            disjoint_expected_probs,
            df,
            log_fidelities,
            linear_fidelities,
            singularity,
            outlier,
        )

    def get_layer_fidelity(self, fidelity_metric: Literal["log", "linear"] = "linear", disjoint_processing: bool = None):
        """
        Returns the layer fidelities for the XEB experiment
        Args:
            fidelity_metric: Indicate which fidelity metric should be computed: "log" or "linear"
            disjoint_processing: Indicate if disjoint processing should be applied to the results
        """
        if disjoint_processing is not None:
            assert isinstance(disjoint_processing, bool), "disjoint_processing should be a boolean"
        else:
            disjoint_processing = self.xeb_config.disjoint_processing
        
        if disjoint_processing:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=1)
            else:
                Fxeb = np.array([fidelity["fidelity"] for fidelity in self.linear_fidelities])
            
            a = [None] * len(self.qubit_names)
            layer_fid = [None] * len(self.qubit_names)
            for q, qubit in enumerate(self.qubit_names):
                a[q], layer_fid[q], *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb[q])
        else:
            if fidelity_metric == "log":
                Fxeb = np.nanmean(self.log_fidelities, axis=0)
            else:
                Fxeb = np.array(self.linear_fidelities["fidelity"])
            a, layer_fid, *_ = fit_exponential_decay(self.xeb_config.depths, Fxeb)
        return layer_fid


    def plot_fidelities(self, fit_linear: bool = True, fit_log_entropy: bool = True, separate_plots: bool = False):
        """
        Plot the cross-entropy fidelities for the XEB experiment
        Args:
            fit_linear: Indicate if the linear XEB data should be fitted
            fit_log_entropy: Indicate if the log-entropy XEB data should be fitted
            separate_plots: Indicate if the fidelities should be plotted on separate plots (one per qubit, relevant
                only when disjoint_processing is True)
        Returns:
        """
        figs = [plt.figure()]
        plt.rcParams["text.usetex"] = False

        def plot_fidelity_data(xx, depths, linear_fidelities, Fxeb, qubit_label=""):
            try:
                if fit_linear:
                    a_lin, layer_fid_lin, *_ = fit_exponential_decay(
                        linear_fidelities["depth"], linear_fidelities["fidelity"]
                    )
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_lin, layer_fid_lin),
                        label=f"Fit (Linear XEB{qubit_label}), layer_fidelity={layer_fid_lin * 100:.1f}%",
                    )
            except Exception:
                warnings.warn("Fit for Linear XEB data failed")

            try:
                if fit_log_entropy:
                    a_log, layer_fid_log, *_ = fit_exponential_decay(depths, Fxeb)
                    plt.plot(
                        xx,
                        exponential_decay(xx, a_log, layer_fid_log),
                        label=f"Fit (Log XEB{qubit_label}), layer_fidelity={layer_fid_log * 100:.1f}%",
                        linewidth=5.7, color="red",
                    )
            except Exception:
                warnings.warn("Fit for Log XEB data failed")

            if fit_linear:
                mask_lin = (linear_fidelities["fidelity"] > 0) & (linear_fidelities["fidelity"] < 1)
                plt.scatter(
                    linear_fidelities["depth"][mask_lin],
                    linear_fidelities["fidelity"][mask_lin],
                    label=f"Linear XEB Data {qubit_label}",
                )

            if fit_log_entropy and not np.isnan(Fxeb).all():
                mask_log = (Fxeb > 0) & (Fxeb < 1)
                plt.scatter(depths[mask_log], Fxeb[mask_log], label=f"Log XEB Data {qubit_label}", 
                            s=13.5, c="blue")
            else:
                warnings.warn(f"Log XEB data for {qubit_label} is a singularity.")

        if self.xeb_config.disjoint_processing:
            for q, qubit in enumerate(self.qubit_names):
                if separate_plots and q > 0:
                    figs.append(plt.figure())
                linear_fidelities = self.linear_fidelities[q]
                xx = np.linspace(0, linear_fidelities["depth"].max())
                Fxeb = np.nanmean(self.log_fidelities[q], axis=0)
                plot_fidelity_data(xx, self.xeb_config.depths, linear_fidelities, Fxeb, f" {qubit}")
                plt.ylabel("Circuit fidelity", fontsize=20)
                plt.xlabel("Cycle Depth $d$", fontsize=20)
                plt.title("XEB Fidelity")
                plt.legend(loc="best")
                if separate_plots:
                    plt.tight_layout()
                    plt.show()
        else:
            xx = np.linspace(0, self.linear_fidelities["depth"].max())
            Fxeb = np.nanmean(self.log_fidelities, axis=0)
            plot_fidelity_data(xx, self.xeb_config.depths, self.linear_fidelities, Fxeb)
            plt.ylabel("Circuit fidelity", fontsize=20)
            plt.xlabel("Cycle Depth $d$", fontsize=20)
            plt.title("XEB Fidelity")
            plt.legend(loc="best")
            plt.tight_layout()
            plt.show()

        return figs

    def plot_records(self):
        """
        Plot the records for the XEB experiment
        Returns:

        """
        depths = self.xeb_config.depths
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
            for i, q in enumerate(self.qubit_names):
                _lines = []
                plt.figure()
                fids.append(self.records[i].groupby("depth").apply(per_cycle_depth, _lines).reset_index())
                plt.xlabel(r"$e_U - u_U$", fontsize=18)
                plt.ylabel(r"$m_U - u_U$", fontsize=18)
                _lines = np.asarray(_lines)
                plt.legend(_lines[[0, -1]], depths[[0, -1]], loc="best", title="Cycle depth")
                plt.title(
                    "q-%s: Fxeb_linear = %s"
                    % (
                        q,
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
            for i, q in enumerate(self.qubit_names):
                for j in range(2):
                    titles.append(f"{q}<{j}> Measured")
                    titles.append(f"{q}<{j}> Expected")
                    data.append(self.disjoint_measured_probs[i, :, :, j])
                    data.append(self.disjoint_expected_probs[i, :, :, j])

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
                ax.pcolor(
                    self.xeb_config.depths,
                    range(self.xeb_config.seqs),
                    np.abs(data[plot_idx]),
                    vmin=0,
                    vmax=1,
                    cmap="viridis",
                )
                ax.set_title(titles[plot_idx])
                ax.set_xlabel("Circuit depth")
                ax.set_ylabel("Sequences")
                ax.set_xticks(self.xeb_config.depths)
                ax.set_yticks(np.arange(1, self.xeb_config.seqs + 1))
                fig.colorbar(
                    ax.pcolor(
                        self.xeb_config.depths,
                        range(self.xeb_config.seqs),
                        np.abs(data[plot_idx]),
                        vmin=0,
                        vmax=1,
                    ),
                    ax=ax,
                )

            plt.tight_layout()
            plt.show()

    @property
    def measured_probs(self):
        """
        Measured probabilities of the states
        Returns: Measured probabilities of the states
        """
        return self._joint_measured_probs

    @property
    def disjoint_measured_probs(self):
        """
        Measured probabilities of the states for disjoint processing
        Returns: Measured probabilities of the states for disjoint processing
        """
        return self._disjoint_measured_probs

    @property
    def expected_probs(self):
        """
        Expected probabilities of the states
        Returns: Expected probabilities of the states
        """
        return self._joint_expected_probs

    @property
    def disjoint_expected_probs(self):
        """
        Expected probabilities of the states for disjoint processing
        Returns: Expected probabilities of the states for disjoint processing
        """
        return self._disjoint_expected_probs

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
        if self.xeb_config.disjoint_processing:
            fidelities = [self._linear_fidelities[q] for q in range(self.xeb_config.n_qubits)]
        else:
            fidelities = self._linear_fidelities
        return fidelities

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

    @property
    def qubit_names(self):
        """
        Qubit names
        """
        return [qubit.name if isinstance(qubit, Transmon) else qubit for qubit in self.xeb_config.qubits]
