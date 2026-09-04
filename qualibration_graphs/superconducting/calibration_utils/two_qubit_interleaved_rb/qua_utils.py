from typing import Literal

import numpy as np
from more_itertools import flatten
from qm.qua import *
from qm.qua._expressions import QuaArrayVariable, QuaVariable
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from quam_config import Quam


def reset_qubits(node, control: Quam.qubit_type, target: Quam.qubit_type, thermalization_time: float | None = None):

    control.reset(reset_type=node.parameters.reset_type)
    target.reset(reset_type=node.parameters.reset_type)


def play_gate(
    gate: QuaVariable,
    qubit_pair: Quam.qubit_pair_type,
    state: QuaVariable,
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st: "_ResultSource",
    reset_type: Literal["thermal", "active"],
    cz_operation: str = "cz_unipolar"
):
    with switch_(gate, unsafe=True):

        with case_(0):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(1):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(2):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(3):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(4):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(5):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(6):
            qubit_pair.qubit_control.xy.play("x90")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(7):
            qubit_pair.qubit_control.xy.play("x90")
        with case_(8):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(9):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(10):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(11):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(12):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(13):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(14):
            qubit_pair.qubit_control.xy.play("x180")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(15):
            qubit_pair.qubit_control.xy.play("x180")
        with case_(16):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(17):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(18):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(19):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(20):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(21):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(22):
            qubit_pair.qubit_control.xy.play("y90")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(23):
            qubit_pair.qubit_control.xy.play("y90")
        with case_(24):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x90")
        with case_(25):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("x180")
        with case_(26):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y90")
        with case_(27):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.play("y180")
        with case_(28):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(29):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(30):
            qubit_pair.qubit_control.xy.play("y180")
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(31):
            qubit_pair.qubit_control.xy.play("y180")
        with case_(32):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(33):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(34):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(35):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(36):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(37):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(38):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(39):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi / 2)
        with case_(40):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(41):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(42):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(43):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(44):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(45):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(46):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(47):
            qubit_pair.qubit_control.xy.frame_rotation(np.pi)
        with case_(48):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("x90")
        with case_(49):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("x180")
        with case_(50):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("y90")
        with case_(51):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.play("y180")
        with case_(52):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(53):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(54):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(55):
            qubit_pair.qubit_control.xy.frame_rotation(3 * np.pi / 2)
        with case_(56):
            qubit_pair.qubit_target.xy.play("x90")
        with case_(57):
            qubit_pair.qubit_target.xy.play("x180")
        with case_(58):
            qubit_pair.qubit_target.xy.play("y90")
        with case_(59):
            qubit_pair.qubit_target.xy.play("y180")
        with case_(60):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi / 2)
        with case_(61):
            qubit_pair.qubit_target.xy.frame_rotation(np.pi)
        with case_(62):
            qubit_pair.qubit_target.xy.frame_rotation(3 * np.pi / 2)
        with case_(63):
            qubit_pair.qubit_control.wait(20)
            qubit_pair.qubit_target.wait(20)
        with case_(64):  # CZ
            qubit_pair.macros[cz_operation].apply()
        with case_(65):  # idle_2q
            qubit_pair.qubit_control.wait(4)
            qubit_pair.qubit_target.wait(4)

        with case_(66):

            align()
            qubit_pair.qubit_control.readout_state(state_control)
            qubit_pair.qubit_target.readout_state(state_target)

            assign(state, state_control * 2 + state_target)
            save(state, state_st)

            # Initialize the qubits
            qubit_pair.qubit_control.reset(reset_type=reset_type)
            qubit_pair.qubit_target.reset(reset_type=reset_type)

            # Reset the frame of the qubits in order not to accumulate rotations
            reset_frame(qubit_pair.qubit_control.xy.name, qubit_pair.qubit_target.xy.name)

            align()


def play_sequence(
    sequence: QuaArrayVariable,
    depth: int,
    qubit_pair: Quam.qubit_pair_type,
    state: list[QuaVariable],
    state_control: QuaVariable,
    state_target: QuaVariable,
    state_st,
    reset_type: Literal["thermal", "active"],
    cz_operation: str = "cz_unipolar"
):

    i = declare(int)
    with for_(i, 0, i < depth, i + 1):
        play_gate(sequence[i], qubit_pair, state, state_control, state_target, state_st, reset_type, cz_operation)


class QuaProgramHandler:

    def __init__(
        self,
        node: QualibrationNode,
        num_pairs: int,
        circuits_as_ints: list[int],
        machine: Quam,
        qubit_pairs: list[Quam.qubit_pair_type],
        max_sequence_length: int = 6000,
    ):

        self.u = unit(coerce_to_integer=True)
        self.node = node
        self.num_pairs = num_pairs
        self.circuits_as_ints = circuits_as_ints
        self.machine = machine
        self.qubit_pairs = qubit_pairs
        self.max_sequence_length = max_sequence_length

        if self.node.parameters.use_input_stream:
            self.circuits_as_ints_batched = split_list_by_integer_count(self.circuits_as_ints, self.max_sequence_length)
            self.circuits_as_ints_batched = [list(flatten(batch)) for batch in self.circuits_as_ints_batched]
            self.sequence_lengths = [len(batch) for batch in self.circuits_as_ints_batched]
            self.max_current_sequence_length = max(len(seq) for seq in self.circuits_as_ints_batched)

    def _get_qua_program_with_input_stream(self):

        with program() as rb:

            n = declare(int)
            n_st = declare_stream()

            sequence = declare_input_stream(int, name="sequence", size=self.max_current_sequence_length)

            # The relevant streams
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            for i, qubit_pair in enumerate(self.qubit_pairs):

                # Bring the active qubits to the desired frequency point
                self.machine.set_all_fluxes(
                    flux_point=self.node.parameters.flux_point_joint_or_independent, target=qubit_pair.qubit_control
                )

                # Initialize the qubits
                if self.node.parameters.reset_type_thermal_or_active == "active":
                    active_reset(qubit_pair.qubit_control, "readout")
                    active_reset(qubit_pair.qubit_target, "readout")
                else:
                    # qubit_pair.qubit_control.resonator.wait(4)
                    qubit_pair.qubit_control.resonator.wait(qubit_pair.qubit_control.thermalization_time * self.u.ns)
                    qubit_pair.qubit_target.resonator.wait(qubit_pair.qubit_target.thermalization_time * self.u.ns)

                # Align the two elements to play the sequence after qubit initialization
                align()

                for l in self.sequence_lengths:
                    advance_input_stream(sequence)

                    with for_(n, 0, n < self.node.parameters.num_averages, n + 1):

                        play_sequence(
                            sequence,
                            l,
                            qubit_pair,
                            state,
                            state_control,
                            state_target,
                            state_st[i],
                            self.node.parameters.reset_type_thermal_or_active,
                        )

                        save(n, n_st)

            with stream_processing():
                n_st.save("iteration")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(
                        len(self.node.parameters.circuit_lengths)
                    ).buffer(self.node.parameters.num_averages).save(f"state{i + 1}")
        return rb

    def _get_qua_program_without_input_stream(self):

        job_sequence = list(flatten(self.circuits_as_ints))
        sequence_length = len(job_sequence)

        with program() as rb:

            n = declare(int)
            n_st = declare_stream()

            job_sequence_qua = declare(int, value=job_sequence)

            # The relevant streams
            state_control = declare(int)
            state_target = declare(int)
            state = declare(int)
            state_st = [declare_stream() for _ in range(self.num_pairs)]

            # Bring the active qubits to the desired frequency point
            for qp in self.qubit_pairs:
                self.node.machine.initialize_qpu(target=qp.qubit_control)
                self.node.machine.initialize_qpu(target=qp.qubit_target)

            for i, qubit_pair in enumerate(self.qubit_pairs):

                # Initialize the qubits
                qubit_pair.qubit_control.reset(reset_type=self.node.parameters.reset_type)
                qubit_pair.qubit_target.reset(reset_type=self.node.parameters.reset_type)

                # Align the two elements to play the sequence after qubit initialization
                align()

                with for_(n, 0, n < self.node.parameters.num_shots, n + 1):

                    play_sequence(
                        job_sequence_qua,
                        sequence_length,
                        qubit_pair,
                        state,
                        state_control,
                        state_target,
                        state_st[i],
                        self.node.parameters.reset_type,
                        self.node.parameters.operation,
                    )

                    save(n, n_st)

            with stream_processing():
                n_st.save("n")
                for i in range(len(self.qubit_pairs)):
                    state_st[i].buffer(self.node.parameters.num_circuits_per_length).buffer(
                        len(self.node.parameters.circuit_lengths)
                    ).buffer(self.node.parameters.num_shots).save(f"state{i + 1}")
        return rb

    def get_qua_program(self):

        if self.node.parameters.use_input_stream:
            return self._get_qua_program_with_input_stream()
        else:
            return self._get_qua_program_without_input_stream()


