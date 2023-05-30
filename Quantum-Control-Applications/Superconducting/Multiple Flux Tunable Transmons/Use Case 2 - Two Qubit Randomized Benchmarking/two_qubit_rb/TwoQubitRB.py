from typing import Callable, Dict, List, Literal, Tuple
import cirq
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm.qua._dsl import _Expression

from qualang_tools.bakery.bakery import Baking
from .RBBaker import RBBaker
from .RBResult import RBResult
from .gates import GateGenerator, gate_db
from .simple_tableau import SimpleTableau
from .util import run_in_thread, pbar


class TwoQubitRb:
    _buffer_length = 4096

    def __init__(
        self,
        config: dict,
        single_qubit_gate_generator: Callable[[Baking, int, float, float, float], None],
        two_qubit_gate_generators: Dict[Literal["sqr_iSWAP", "CNOT", "CZ"], Callable[[Baking, int, int], None]],
        prep_func: Callable[[], None],
        measure_func: Callable[[], Tuple[_Expression, _Expression]],
        verify_generation: bool = False,
    ):
        """ 
        A class for running two qubit randomized benchmarking experiments.

        This class is used to generate the experiment configuration and run the experiment.
        The experiment is run by calling the run method.

        Gate generation is performed using the `Baking`[https://github.com/qua-platform/py-qua-tools/blob/main/qualang_tools/bakery/README.md] class.
        This class adds to QUA the ability to generate arbitrary waveforms ("baked waveforms") using syntax similar to QUA.

        Args:
            config: A QUA configuration containing the configuration for the experiment.

            single_qubit_gate_generator: A callable used to generate a single qubit gate using a signature similar to `phasedXZ`[https://quantumai.google/reference/python/cirq/PhasedXZGate].
                This is done using the baking object (see above).
                Note that this allows us to execute every type of single qubit gate.
                Callable arguments: 
                    baking: The baking object.
                    qubit: The qubit number.
                    x: The x rotation exponent.
                    z: The z rotation exponent.
                    a: the axis phase exponent.

            two_qubit_gate_generators: A dictionary mapping one or more two qubit gate names to callables used to generate those gates.
                This is done using the baking object (see above).
                Callable arguments:
                    baking: The baking object.
                    qubit1: The first qubit number.
                    qubit2: The second qubit number.
                This callable should generate a two qubit gate. 


            prep_func: A callable used to reset the qubits to the |00> state. This function does not use the baking object, and is a proper QUA code macro.
                Callable arguments: None

            measure_func: A callable used to measure the qubits. This function does not use the baking object, and is a proper QUA code macro.
                Callable arguments: None
                Returns:
                    A tuple containing the measured values of the two qubits as Qua expressions.
                    The expression must evaluate to a boolean value. False means |0>, True means |1>. The MSB is the first qubit.

            verify_generation: A boolean indicating whether to verify the generated sequences. Not be used in production, as it is very slow.
        """
        for i, qe in config["elements"].items():
            if "operations" not in qe:
                qe["operations"] = {}
        self._rb_baker = RBBaker(
            config, single_qubit_gate_generator, two_qubit_gate_generators)
        self._config = self._rb_baker.bake()
        self._symplectic_generator = GateGenerator(
            set(two_qubit_gate_generators.keys()))
        self._prep_func = prep_func
        self._measure_func = measure_func
        self._verify_generation = verify_generation

    def _verify_rb_sequence(self, gate_ids, final_tableau: SimpleTableau):
        if final_tableau != SimpleTableau(np.eye(4), [0, 0, 0, 0]):
            raise RuntimeError("Verification of RB sequence failed")
        gates = []
        for gate_id in gate_ids:
            gates.extend(self._symplectic_generator.generate(gate_id))

        unitary = cirq.Circuit(gates).unitary()
        fixed_phase_unitary = np.conj(np.trace(unitary) / 4) * unitary
        if np.linalg.norm(fixed_phase_unitary - np.eye(4)) > 1e-12:
            raise RuntimeError("Verification of RB sequence failed")

    def _gen_rb_sequence(self, depth):
        gate_ids = []
        tableau = SimpleTableau(np.eye(4), [0, 0, 0, 0])
        for i in range(depth):
            symplectic = gate_db.rand_symplectic()
            pauli = gate_db.rand_pauli()
            gate_ids.append(symplectic)
            gate_ids.append(pauli)

            tableau = tableau.then(gate_db.get_tableau(
                symplectic)).then(gate_db.get_tableau(pauli))

        inv_tableau = tableau.inverse()
        inv_id = gate_db.find_symplectic_gate_id_by_tableau_g(inv_tableau)
        after_inv_tableau = tableau.then(gate_db.get_tableau(inv_id))

        pauli = gate_db.find_pauli_gate_id_by_tableau_alpha(after_inv_tableau)

        gate_ids.append(inv_id)
        gate_ids.append(pauli)

        if self._verify_generation:
            final_tableau = after_inv_tableau.then(gate_db.get_tableau(pauli))
            self._verify_rb_sequence(gate_ids, final_tableau)

        return gate_ids

    def _gen_qua_program(
        self,
        sequence_depths: list[int],
        num_repeats: int,
        num_averages: int,
    ):

        with program() as prog:
            sequence_depth = declare(int)
            repeat = declare(int)
            n_avg = declare(int)
            state = declare(int)
            length = declare(int)
            progress = declare(int)
            progress_os = declare_stream()
            state_os = declare_stream()
            gates_len_is = declare_input_stream(
                int, name="gates_len_is", size=1)
            qubit1_gates_is = declare_input_stream(
                int, name="qubit1_gates_is", size=self._buffer_length)
            qubit2_gates_is = declare_input_stream(
                int, name="qubit2_gates_is", size=self._buffer_length)
            two_qubits_gates_is = declare_input_stream(
                int, name="two_qubits_gates_is", size=self._buffer_length)

            assign(progress, 0)
            with for_each_(sequence_depth, sequence_depths):
                with for_(repeat, 0, repeat < num_repeats, repeat + 1):
                    assign(progress, progress + 1)
                    save(progress, progress_os)
                    advance_input_stream(gates_len_is)
                    advance_input_stream(qubit1_gates_is)
                    advance_input_stream(qubit2_gates_is)
                    advance_input_stream(two_qubits_gates_is)
                    assign(length, gates_len_is[0])
                    with for_(n_avg, 0, n_avg < num_averages, n_avg + 1):
                        self._prep_func()
                        self._rb_baker.run(
                            qubit1_gates_is, qubit2_gates_is, two_qubits_gates_is, length)
                        out1, out2 = self._measure_func()
                        assign(state, (Cast.to_int(out2) << 1) +
                               Cast.to_int(out1))
                        save(state, state_os)

            with stream_processing():
                state_os.buffer(len(sequence_depths),
                                num_repeats, num_averages).save("state")
                progress_os.save("progress")
        return prog

    def _decode_sequence_for_channel(self, channel: str, seq: list):
        seq = [self._rb_baker.decode(i, channel) for i in seq]
        if len(seq) > self._buffer_length:
            RuntimeError("Buffer is too small")
        return seq + [0] * (self._buffer_length - len(seq))

    @run_in_thread
    def _insert_all_input_stream(self, job, sequence_depths, num_repeats):
        for sequence_depth in sequence_depths:
            for repeat in range(num_repeats):
                sequence = self._gen_rb_sequence(sequence_depth)
                job.insert_input_stream("gates_len_is", len(sequence))
                job.insert_input_stream(
                    "qubit1_gates_is", self._decode_sequence_for_channel("qubit1", sequence))
                job.insert_input_stream(
                    "qubit2_gates_is", self._decode_sequence_for_channel("qubit2", sequence))
                job.insert_input_stream(
                    "two_qubits_gates_is", self._decode_sequence_for_channel(
                        "two_qubit_gates", sequence)
                )

    def run(
        self,
        qmm: QuantumMachinesManager,
        circuit_depths: List[int],
        num_circuits_per_depth: int, 
        num_shots_per_circuit: int, 
        interleaving_gate: Optional[list] = None,
    ):
        """
        Runs the randomized benchmarking experiment. The experiment is sweep over Clifford circuits with varying depths.
        For every depth, we generate a number of random circuits and run them. The number of different circuits is determined by
        the num_circuits_per_depth parameter. The number of shots per individual circuit is determined by the num_averages parameter.
        
        Args:
            qmm (QuantumMachinesManager): The Quantum Machines Manager object which is used to run the experiment.
            circuit_depths (List[int]): A list of the number of Cliffords per circuit (not including inverse).
            num_circuits_per_depth (int): The number of different circuit randomizations per depth.
            num_shots_per_circuit (int): The number of shots per particular circuit.
            interleaving_gate (Optional[list]): Not supported yet. Please contact QM if you need this feature.

        Example:
            >>> from qm.QuantumMachinesManager import QuantumMachinesManager
            >>> from qm.qua import *  
            >>> from qua_config import config  # generation not in scope of this example
            >>> from TwoQubitRB import TwoQubitRB
            >>> qmm = QuantumMachinesManager(config)
            
        """

        if interleaving_gate is not None:
            raise NotImplementedError("Interleaving gates are not supported yet")
        prog = self._gen_qua_program(
            circuit_depths, num_circuits_per_depth, num_shots_per_circuit
        )

        qm = qmm.open_qm(self._config)
        job = qm.execute(prog)

        self._insert_all_input_stream(job, circuit_depths, num_circuits_per_depth)

        full_progress = len(circuit_depths) * num_circuits_per_depth
        pbar(job.result_handles, full_progress, "progress")
        job.result_handles.wait_for_all_values()

        return RBResult(
            circuit_depths=circuit_depths,
            num_repeats=num_circuits_per_depth,
            num_averages=num_shots_per_circuit,
            state=job.result_handles.get("state").fetch_all(),
        )
