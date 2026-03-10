"""Regression tests for the simulated video-mode QuAM factory."""

from pathlib import Path
import sys

from qm.qua import Cast, align, assign, declare, fixed, program, wait

ROOT = Path(__file__).resolve().parents[5]
QUANTUM_DOTS_DIR = ROOT / "qualibration_graphs" / "quantum_dots"
if str(QUANTUM_DOTS_DIR) not in sys.path:
    sys.path.insert(0, str(QUANTUM_DOTS_DIR))

from calibration_utils.run_video_mode.simulated_video_mode import (  # noqa: E402
    save_simulated_video_mode_quam,
)
from quam_config import Quam  # noqa: E402


def _compile_power_rabi_sequence(machine: Quam) -> None:
    qubit = machine.qubits["Q1"]
    with program():
        amplitude = declare(fixed)
        p1 = declare(int)
        p2 = declare(int)

        align()
        qubit.empty()
        align()
        assign(p1, Cast.to_int(qubit.measure()))
        align()
        qubit.initialize()
        align()
        qubit.x180(amplitude_scale=amplitude)
        align()
        assign(p2, Cast.to_int(qubit.measure()))
        align()
        qubit.voltage_sequence.apply_compensation_pulse()


def _compile_time_rabi_sequence(machine: Quam) -> None:
    qubit = machine.qubits["Q1"]
    with program():
        duration = declare(int)
        p1 = declare(int)
        p2 = declare(int)

        align()
        qubit.empty()
        align()
        assign(p1, Cast.to_int(qubit.measure()))
        align()
        qubit.initialize()
        align()
        qubit.x180(duration=duration)
        align()
        assign(p2, Cast.to_int(qubit.measure()))
        align()
        qubit.voltage_sequence.apply_compensation_pulse()


def _compile_ramsey_sequence(machine: Quam) -> None:
    qubit = machine.qubits["Q1"]
    with program():
        duration = declare(int)
        detuning = declare(int)
        p1 = declare(int)
        p2 = declare(int)

        qubit.xy.update_frequency(qubit.xy.intermediate_frequency + detuning)
        align()
        qubit.empty()
        align()
        assign(p1, Cast.to_int(qubit.measure()))
        align()
        qubit.initialize()
        align()
        qubit.x90()
        align()
        wait(duration)
        qubit.voltage_sequence.step_to_voltages({}, duration=duration * 4)
        align()
        qubit.x90()
        align()
        assign(p2, Cast.to_int(qubit.measure()))
        align()
        qubit.voltage_sequence.apply_compensation_pulse()


def _compile_hahn_sequence(machine: Quam) -> None:
    qubit = machine.qubits["Q1"]
    with program():
        duration = declare(int)
        p1 = declare(int)
        p2 = declare(int)

        align()
        qubit.empty()
        align()
        assign(p1, Cast.to_int(qubit.measure()))
        align()
        qubit.initialize()
        align()
        qubit.x90()
        align()
        wait(duration)
        qubit.voltage_sequence.step_to_voltages({}, duration=duration * 4)
        align()
        qubit.x180()
        align()
        wait(duration)
        qubit.voltage_sequence.step_to_voltages({}, duration=duration * 4)
        align()
        qubit.x90()
        align()
        assign(p2, Cast.to_int(qubit.measure()))
        align()
        qubit.voltage_sequence.apply_compensation_pulse()


def test_simulated_video_mode_quam_supports_single_qubit_nodes(tmp_path):
    state_path = save_simulated_video_mode_quam(tmp_path / "quam_state")
    machine = Quam.load(str(state_path))

    assert list(machine.qubits.keys()) == ["Q1", "Q2"]
    assert machine.active_qubit_names == ["Q1", "Q2"]

    q1 = machine.qubits["Q1"]
    assert {"x180", "x90", "-x90", "y90", "-y90"}.issubset(q1.xy.operations.keys())
    assert {"x180", "x90", "measure"}.issubset(q1.macros.keys())

    _compile_power_rabi_sequence(machine)
    _compile_time_rabi_sequence(machine)
    _compile_ramsey_sequence(machine)
    _compile_hahn_sequence(machine)

    machine.generate_config()
