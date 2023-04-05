import cirq
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig  # Kevin
import matplotlib.pyplot as plt  # Kevin

from RBBaker import RBBaker
from RBResult import RBResult
from gates import GateGenerator, gate_db
from simple_tableau import SimpleTableau
from util import run_in_thread, pbar


class TwoQubitRb:
    _buffer_length = 4096

    def __init__(
        self,
        config: dict,
        phased_xz_generator: callable,
        two_qubit_gate_generators: dict[str, callable],
        prep_func: callable,
        measure_func: callable,
        verify_generation: bool = False,
    ):
        self._rb_baker = RBBaker(config, phased_xz_generator, two_qubit_gate_generators)
        self._config = self._rb_baker.bake()
        self._symplectic_generator = GateGenerator(set(two_qubit_gate_generators.keys()))
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

            tableau = tableau.then(gate_db.get_tableau(symplectic)).then(gate_db.get_tableau(pauli))

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
        prep_args: list = None,
        prep_kwargs: dict = None,
        measure_args: list = None,
        measure_kwargs: dict = None,
    ):
        if prep_args is None:
            prep_args = []
        if prep_kwargs is None:
            prep_kwargs = {}
        if measure_args is None:
            measure_args = []
        if measure_kwargs is None:
            measure_kwargs = {}

        with program() as prog:
            sequence_depth = declare(int)
            repeat = declare(int)
            n_avg = declare(int)
            state = declare(int)
            length = declare(int)
            progress = declare(int)
            progress_os = declare_stream()
            state_os = declare_stream()
            gates_len_is = declare_input_stream(int, name="gates_len_is", size=1)
            qubit1_gates_is = declare_input_stream(int, name="qubit1_gates_is", size=self._buffer_length)
            qubit2_gates_is = declare_input_stream(int, name="qubit2_gates_is", size=self._buffer_length)
            two_qubits_gates_is = declare_input_stream(int, name="two_qubits_gates_is", size=self._buffer_length)

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
                        self._prep_func(*prep_args, **prep_kwargs)
                        self._rb_baker.run(qubit1_gates_is, qubit2_gates_is, two_qubits_gates_is, length)
                        out1, out2 = self._measure_func(*measure_args, **measure_kwargs)
                        assign(state, (Cast.to_int(out2) << 1) + Cast.to_int(out1))
                        save(state, state_os)

            with stream_processing():
                state_os.buffer(len(sequence_depths), num_repeats, num_averages).save("state")
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
                job.insert_input_stream("qubit1_gates_is", self._decode_sequence_for_channel("qubit1", sequence))
                job.insert_input_stream("qubit2_gates_is", self._decode_sequence_for_channel("qubit2", sequence))
                job.insert_input_stream(
                    "two_qubits_gates_is", self._decode_sequence_for_channel("two_qubit_gates", sequence)
                )

    def run(
        self,
        qmm: QuantumMachinesManager,
        sequence_depths: list[int],
        num_repeats: int,  # this is for different sequences
        num_averages: int,  # how much to average every sequence
        interleaving_gate: Optional[list] = None,
        prep_args: list = None,
        prep_kwargs: dict = None,
        measure_args: list = None,
        measure_kwargs: dict = None,
    ):

        if interleaving_gate is not None:
            raise RuntimeError("Interleaving gates are not supported yet")
        prog = self._gen_qua_program(
            sequence_depths, num_repeats, num_averages, prep_args, prep_kwargs, measure_args, measure_kwargs
        )

        with program() as prog_now:
            play("baked_Op_10", "qe0")

        qm = qmm.open_qm(self._config, close_other_machines=False)  # Kevin
        # job_sim = qmm.simulate(self._config, prog_now, SimulationConfig(2500))
        # job_sim.get_simulated_samples().con1.plot()
        job = qm.execute(prog)

        self._insert_all_input_stream(job, sequence_depths, num_repeats)

        full_progress = len(sequence_depths) * num_repeats
        pbar(job.result_handles, full_progress, "progress")
        job.result_handles.wait_for_all_values()
        print(job.execution_report())  # Kevin
        qm.close()

        return RBResult(
            sequence_depths=sequence_depths,
            num_repeats=num_repeats,
            num_averages=num_averages,
            state=job.result_handles.get("state").fetch_all(),
        )
