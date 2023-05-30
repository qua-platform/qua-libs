import copy
from typing import Callable, Dict, Tuple

import cirq
from cirq import GateOperation
from qm.qua import switch_, case_, declare, align, for_
from qualang_tools.bakery.bakery import Baking, baking

from .gates import GateCommand, GateGenerator, gate_db


class RBBaker:
    def __init__(self, config, single_qubit_gate_generator: Callable, two_qubit_gate_generators: Dict[str, Callable]):
        self._config = copy.deepcopy(config)
        self._single_qubit_gate_generator = single_qubit_gate_generator
        self._two_qubit_gate_generators = two_qubit_gate_generators
        self._symplectic_generator = GateGenerator(set(two_qubit_gate_generators.keys()))
        self._gate_length = {}
        self._bakers = {}
        self._op_id_by_cmd_ids = {}
        tmp_config = copy.deepcopy(config)
        self._two_qubits_qes = set()
        for gen in two_qubit_gate_generators.values():
            with baking(tmp_config) as b:
                gen(b, 0, 1)
                self._two_qubits_qes.update(b.get_qe_set())

    @staticmethod
    def _get_qubits(op):
        return [q.x for q in op.qubits]

    @staticmethod
    def _get_phased_xz_args(op):
        return op.gate.x_exponent, op.gate.z_exponent, op.gate.axis_phase_exponent

    def _validate_two_qubit_gate_available(self, name):
        if name not in self._two_qubit_gate_generators:
            raise RuntimeError(f"Two qubit gate '{name}' implementation not provided.")

    def _gen_gate(self, baker: Baking, gate_op: GateOperation):
        if type(gate_op.gate) == cirq.PhasedXZGate:
            self._single_qubit_gate_generator(baker, self._get_qubits(gate_op)[0], *self._get_phased_xz_args(gate_op))
        elif type(gate_op.gate) == cirq.ISwapPowGate and gate_op.gate.exponent == 0.5:
            self._validate_two_qubit_gate_available("sqr_iSWAP")
            self._two_qubit_gate_generators["sqr_iSWAP"](baker, *self._get_qubits(gate_op))
        elif type(gate_op.gate) == cirq.CNotPowGate and gate_op.gate.exponent == 1:
            self._validate_two_qubit_gate_available("CNOT")
            self._two_qubit_gate_generators["CNOT"](baker, *self._get_qubits(gate_op))
        elif type(gate_op.gate) == cirq.CZPowGate and gate_op.gate.exponent == 1:
            self._validate_two_qubit_gate_available("CZ")
            self._two_qubit_gate_generators["CZ"](baker, *self._get_qubits(gate_op))
        else:
            raise RuntimeError("unsupported gate")

    def _get_gate_op_time(self, gate_op: GateOperation):
        if gate_op in self._gate_length:
            return self._gate_length[gate_op]
        config = copy.deepcopy(self._config)
        with baking(config) as b:
            self._gen_gate(b, gate_op)
            length = b.get_current_length()
        self._gate_length[gate_op] = length
        return length

    def _gen_cmd_per_qubits(self, config: dict, cmd_id, qubits):
        gate_ops = self._symplectic_generator.generate(cmd_id)
        with baking(config) as b:
            for gate_op in gate_ops:
                gate_qubits = self._get_qubits(gate_op)
                # Kevin -> because cirq.CNOT(q2,q1) had to add the case to eval [1, 0]
                if len(qubits) == 2:
                    if gate_qubits == [0, 1] or gate_qubits == [1, 0]:
                        self._gen_gate(b, gate_op)
                    elif (len(qubits) == 1 and len(gate_qubits) == 2) or (len(qubits) == 2 and gate_qubits == [0]):
                        qes = b.get_qe_set()
                        if len(qes) == 0 and len(qubits) == 2:
                            qes = self._two_qubits_qes
                        b.wait(self._get_gate_op_time(gate_op), *qes)
                # Kevin
                elif gate_qubits == qubits:
                    self._gen_gate(b, gate_op)
                elif (len(qubits) == 1 and len(gate_qubits) == 2) or (len(qubits) == 2 and gate_qubits == [0]):
                    qes = b.get_qe_set()
                    if len(qes) == 0 and len(qubits) == 2:
                        qes = self._two_qubits_qes
                    b.wait(self._get_gate_op_time(gate_op), *qes)
        return b

    def _partial_bake_qubit_ops(self, config: dict, qubit):
        output = {}
        op_id = 0
        for cmd_id, command in enumerate(gate_db.commands):
            if command.type not in output:
                output[command.type] = {}
            ops = command.get_qubit_ops(qubit)
            if ops in output[command.type]:
                continue
            output[command.type][ops] = (op_id, self._gen_cmd_per_qubits(config, cmd_id, [qubit]))
            op_id += 1

        return output

    def _partial_bake_two_qubit_ops(self, config: dict, qubits):
        output = {}
        op_id = 0
        for cmd_id, command in enumerate(gate_db.commands):
            if command.type in output:
                continue
            output[command.type] = (op_id, self._gen_cmd_per_qubits(config, cmd_id, qubits))
            op_id += 1
        return output

    def _get_baker(self, channel: str, command: GateCommand) -> Tuple[int, Baking]:
        if channel == "qubit1":
            return self._bakers[channel][command.type][command.q1]
        elif channel == "qubit2":
            return self._bakers[channel][command.type][command.q2]
        elif channel == "two_qubit_gates":
            return self._bakers[channel][command.type]

    def _validate_bakers(self):
        all_qubit1_qes = set()
        all_qubit2_qes = set()
        all_two_qubit_gates_qes = set()
        for cmd_id, command in enumerate(gate_db.commands):
            qubit1_baker = self._get_baker("qubit1", command)[1]
            qubit2_baker = self._get_baker("qubit2", command)[1]
            two_qubit_gates_baker = self._get_baker("two_qubit_gates", command)[1]
            all_qubit1_qes.update(qubit1_baker.get_qe_set())
            all_qubit2_qes.update(qubit2_baker.get_qe_set())
            all_two_qubit_gates_qes.update(two_qubit_gates_baker.get_qe_set())
            qubit1_len = qubit1_baker.get_current_length()
            qubit2_len = qubit2_baker.get_current_length()
            two_qubit_gates_len = two_qubit_gates_baker.get_current_length()
            if len({qubit1_len, qubit2_len, two_qubit_gates_len}) > 1:
                print(qubit1_len, qubit2_len, two_qubit_gates_len, all_two_qubit_gates_qes, command)
                print(len({qubit1_len, qubit2_len, two_qubit_gates_len}))
                print({qubit1_len, qubit2_len, two_qubit_gates_len})

                raise RuntimeError("All gates should be of the same length")
        if (
            len(all_qubit1_qes.intersection(all_qubit2_qes)) > 0
            or len(all_qubit1_qes.intersection(all_two_qubit_gates_qes)) > 0
            or len(all_qubit2_qes.intersection(all_two_qubit_gates_qes)) > 0
        ):
            raise RuntimeError("Overlapped QEs were used for Qubit1/Qubit2/Two qubit gates")

    def bake(self) -> dict:
        config = copy.deepcopy(self._config)
        self._bakers = {
            "qubit1": self._partial_bake_qubit_ops(config, 0),
            "qubit2": self._partial_bake_qubit_ops(config, 1),
            "two_qubit_gates": self._partial_bake_two_qubit_ops(config, [0, 1]),
        }
        self._validate_bakers()
        self._op_id_by_cmd_ids = {
            "qubit1": [self._get_baker("qubit1", c)[0] for c in gate_db.commands],
            "qubit2": [self._get_baker("qubit2", c)[0] for c in gate_db.commands],
            "two_qubit_gates": [self._get_baker("two_qubit_gates", c)[0] for c in gate_db.commands],
        }

        return config

    def decode(self, cmd_id, element):
        return self._op_id_by_cmd_ids[element][cmd_id]

    def run(self, q1_cmds, q2_cmds, two_qubit_cmds, length, unsafe=True):
        q1_cmd_i = declare(int)
        q2_cmd_i = declare(int)
        two_qubit_cmd_i = declare(int)

        align()
        with for_(q1_cmd_i, 0, q1_cmd_i < length, q1_cmd_i + 1):
            with switch_(q1_cmds[q1_cmd_i], unsafe=unsafe):
                for type_ops in self._bakers["qubit1"].values():
                    for case_id, b in type_ops.values():
                        with case_(case_id):
                            b.run()

        with for_(q2_cmd_i, 0, q2_cmd_i < length, q2_cmd_i + 1):
            with switch_(q2_cmds[q2_cmd_i], unsafe=unsafe):
                for type_ops in self._bakers["qubit2"].values():
                    for case_id, b in type_ops.values():
                        with case_(case_id):
                            b.run()

        with for_(two_qubit_cmd_i, 0, two_qubit_cmd_i < length, two_qubit_cmd_i + 1):
            with switch_(two_qubit_cmds[two_qubit_cmd_i], unsafe=unsafe):
                for case_id, b in self._bakers["two_qubit_gates"].values():
                    with case_(case_id):
                        b.run()
        align()
