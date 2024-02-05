import copy
import json
from typing import Callable, Dict, Optional, List

import cirq
from cirq import GateOperation
from qm.qua import switch_, case_, declare, align, for_
from qualang_tools.bakery.bakery import Baking, baking
from tqdm import tqdm

from .gates import GateGenerator, gate_db
from .verification.command_registry import CommandRegistry


class RBBaker:
    def __init__(
        self,
        config,
        single_qubit_gate_generator: Callable,
        two_qubit_gate_generators: Dict[str, Callable],
        interleaving_gate: Optional[List[cirq.GateOperation]] = None,
        command_registry: Optional[CommandRegistry] = None,
    ):
        self._command_registry = command_registry
        self._config = copy.deepcopy(config)
        self._single_qubit_gate_generator = single_qubit_gate_generator
        self._two_qubit_gate_generators = two_qubit_gate_generators
        self._interleaving_gate = interleaving_gate
        self._symplectic_generator = GateGenerator(set(two_qubit_gate_generators.keys()))
        self._all_elements = self._collect_all_elements()
        self._cmd_to_op = {}
        self._op_to_baking = {}

    @property
    def all_elements(self):
        return self._all_elements

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

    def _collect_all_elements(self):
        config = copy.deepcopy(self._config)
        qes = set()
        for cmd_id, command in enumerate(gate_db.commands):
            if self._command_registry is not None:
                self._command_registry.set_current_command_id(cmd_id)
            with baking(config) as b:
                self._update_baking_from_cmd_id(b, cmd_id)
                qes.update(b.get_qe_set())
                b.update_config = False
            if self._interleaving_gate is not None:
                with baking(config) as b:
                    self._update_baking_from_gates(b, self._interleaving_gate)
                    qes.update(b.get_qe_set())
                    b.update_config = False
        if self._command_registry is not None:
            self._command_registry.finish()
        return qes

    def _update_baking_from_gates(self, b: Baking, gate_ops, elements=None):
        prev_gate_qubits = []
        for gate_op in gate_ops:
            gate_qubits = self._get_qubits(gate_op)
            if len(gate_qubits) != len(prev_gate_qubits) and elements is not None:
                b.align(*elements)
            prev_gate_qubits = gate_qubits
            self._gen_gate(b, gate_op)
        if elements is not None:
            b.align(*elements)

    def gates_from_cmd_id(self, cmd_id):
        if 0 <= cmd_id < len(gate_db.commands):
            gate_ops = self._symplectic_generator.generate(cmd_id)
        elif self._interleaving_gate is not None and cmd_id == len(gate_db.commands):  # Interleaving gate
            gate_ops = self._interleaving_gate
        else:
            raise RuntimeError("command out of range")
        return gate_ops

    def _update_baking_from_cmd_id(self, b: Baking, cmd_id, elements=None):
        gate_ops = self.gates_from_cmd_id(cmd_id)
        return self._update_baking_from_gates(b, gate_ops, elements)

    @staticmethod
    def _unique_baker_identifier_for_qe(b: Baking, qe: str):
        identifier = {"samples": b._samples_dict[qe], "info": b._qe_dict[qe]}
        return json.dumps(identifier)

    def _bake_all_ops(self, config: dict):
        waveform_id_per_qe = {qe: 0 for qe in self._all_elements}
        waveform_to_baking = {qe: {} for qe in self._all_elements}
        cmd_to_op = {qe: {} for qe in self._all_elements}
        op_to_baking = {qe: [] for qe in self._all_elements}
        num_of_commands = len(gate_db.commands) + (0 if self._interleaving_gate is None else 1)
        for cmd_id in tqdm(range(num_of_commands), desc="Pre-baking pulses for combinations of gates", unit="command"):
            with baking(config) as b:
                self._update_baking_from_cmd_id(b, cmd_id, self._all_elements)
                any_qe_used = False
                for qe in self._all_elements:
                    key = self._unique_baker_identifier_for_qe(b, qe)
                    if key not in waveform_to_baking[qe]:
                        waveform_to_baking[qe][key] = waveform_id_per_qe[qe], b
                        op_to_baking[qe].append(b)
                        waveform_id_per_qe[qe] += 1
                        any_qe_used = True
                    cmd_to_op[qe][cmd_id] = waveform_to_baking[qe][key][0]
                b.update_config = any_qe_used
        return cmd_to_op, op_to_baking

    def bake(self) -> dict:
        config = copy.deepcopy(self._config)
        self._cmd_to_op, self._op_to_baking = self._bake_all_ops(config)
        return config

    def decode(self, cmd_id, element):
        return self._cmd_to_op[element][cmd_id]

    @staticmethod
    def _run_baking_for_qe(b: Baking, qe: str):
        orig_get_qe_set = b.get_qe_set
        b.get_qe_set = lambda: {qe}
        b.run()
        b.get_qe_set = orig_get_qe_set

    def run(self, op_list_per_qe: dict, length, unsafe=True):
        if set(op_list_per_qe.keys()) != self._all_elements:
            raise RuntimeError(f"must specify ops for all elements: {', '.join(self._all_elements)} ")

        align()
        for qe, op_list in op_list_per_qe.items():
            cmd_i = declare(int)
            with for_(cmd_i, 0, cmd_i < length, cmd_i + 1):
                with switch_(op_list[cmd_i], unsafe=unsafe):
                    for op_id, b in enumerate(self._op_to_baking[qe]):
                        with case_(op_id):
                            self._run_baking_for_qe(b, qe)
        align()
