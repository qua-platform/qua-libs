"""LRU macro (pump + virtual Z) and the node 26c phase fit."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.lru_phase_compensation import fit_frame_oscillation, fit_raw_data, wrap_turns  # noqa: E402
from calibration_utils.lru_calibration import add_lru_element  # noqa: E402
from quam_config import Quam  # noqa: E402
from quam_config.lru_components import LRU_OPERATION, LruMacro  # noqa: E402
from tests.lru_machine import build_machine  # noqa: E402

FRAMES = np.arange(25) / 25


def _ramsey(psi, repetitions=1, contrast=0.4, noise=0.0, seed=0):
    """P(e) of x90 - N LRUs - Z(frame) - x90 with a residual psi per LRU, as the analysis models it."""
    rng = np.random.default_rng(seed)
    return 0.5 + contrast * np.cos(2 * np.pi * (FRAMES + repetitions * psi)) + noise * rng.standard_normal(FRAMES.size)


@pytest.mark.parametrize("psi", [-0.4, -0.11, 0.0, 0.07, 0.31])
def test_fit_recovers_the_phase(psi):
    fitted, stderr, contrast, offset, _ = fit_frame_oscillation(FRAMES, _ramsey(psi, noise=0.02))
    assert fitted == pytest.approx(psi, abs=0.01)
    assert 0 < stderr < 0.01
    assert contrast == pytest.approx(0.4, abs=0.03)
    assert offset == pytest.approx(0.5, abs=0.02)


def _node(previous, repetitions=1):
    return SimpleNamespace(
        namespace={"qubits": [SimpleNamespace(name="q1")], "previous_phase_shifts": {"q1": previous}},
        parameters=SimpleNamespace(num_lru_repetitions=repetitions),
    )


def _ds(pe):
    return xr.Dataset({"state": (["qubit", "frame"], [pe])}, coords={"qubit": ["q1"], "frame": FRAMES})


@pytest.mark.parametrize("repetitions", [1, 3])
def test_new_correction_cancels_the_residual(repetitions):
    # The LRU leaves +0.1 turn; 0.03 is already corrected, so 0.07 remains per LRU
    stored = 0.97  # -0.03 turn
    residual = 0.1 + wrap_turns(stored)
    ds_fit, results = fit_raw_data(_ds(_ramsey(residual, repetitions)), _node(stored, repetitions))
    result = results["q1"]
    assert result.success
    assert result.residual_phase == pytest.approx(residual, abs=1e-6)
    # With the new correction the LRU and its virtual Z add up to zero
    assert wrap_turns(0.1 + result.phase_shift) == pytest.approx(0.0, abs=1e-6)
    assert float(ds_fit.phase_shift.sel(qubit="q1")) == pytest.approx(result.phase_shift)


def test_flat_data_fails():
    _, results = fit_raw_data(_ds(_ramsey(0.2, contrast=0.02, noise=0.01)), _node(0.0))
    assert not results["q1"].success
    assert any("contrast" in issue for issue in results["q1"].fit_issues)


def test_lru_macro_round_trips_and_plays_pump_then_virtual_z(tmp_path, monkeypatch):
    from qm import generate_qua_script
    from qm.qua import program

    machine = build_machine(Quam, tmp_path, monkeypatch)
    qubit = machine.qubits["q1"]
    add_lru_element(qubit)
    qubit.macros[LRU_OPERATION] = LruMacro(phase_shift=0.125)
    machine.save()

    loaded = Quam.load()
    macro = loaded.qubits["q1"].macros[LRU_OPERATION]
    assert isinstance(macro, LruMacro)
    assert macro.phase_shift == 0.125
    assert macro.inferred_duration == pytest.approx(loaded.qubits["q1"].lru.operations[LRU_OPERATION].length * 1e-9)

    with program() as prog:
        macro.apply()
    script = generate_qua_script(prog)
    pump, rotation = script.index("play('lru', 'q1.lru')"), script.index("frame_rotation_2pi(0.125, 'q1.xy')")
    assert pump < rotation


def test_zero_phase_shift_plays_no_frame_rotation(tmp_path, monkeypatch):
    from qm import generate_qua_script
    from qm.qua import program

    machine = build_machine(Quam, tmp_path, monkeypatch)
    qubit = machine.qubits["q1"]
    add_lru_element(qubit)
    qubit.macros[LRU_OPERATION] = LruMacro()
    with program() as prog:
        qubit.macros[LRU_OPERATION].apply()
    assert "frame_rotation" not in generate_qua_script(prog)


def test_flux_lru_macro_round_trips_and_plays_tone_then_virtual_z(tmp_path, monkeypatch):
    from qm import generate_qua_script
    from qm.qua import program

    from quam_config.lru_components import FLUX_LRU_MACRO, FluxLru, FluxLruMacro

    machine = build_machine(Quam, tmp_path, monkeypatch)
    qubit = machine.qubits["q1"]
    qubit.flux_lru = FluxLru(transition_frequency=1.49e9, amplitude_v=0.12, duration_ns=1000)
    qubit.macros[FLUX_LRU_MACRO] = FluxLruMacro(phase_shift=0.25)
    machine.save()

    loaded = Quam.load()
    macro = loaded.qubits["q1"].macros[FLUX_LRU_MACRO]
    assert isinstance(macro, FluxLruMacro)
    assert macro.phase_shift == 0.25
    assert macro.inferred_duration == pytest.approx(1e-6)

    with program() as prog:
        macro.apply()
        macro.apply(amplitude_scale=0.0, phase_shift=0.0)  # the control: no tone, no virtual Z
    script = generate_qua_script(prog)
    tone, rotation = script.index("play('flux_lru', 'q1.flux_lru')"), script.index("frame_rotation_2pi(0.25, 'q1.xy')")
    assert tone < rotation
    assert script.count("frame_rotation") == 1
    assert "play('flux_lru', 'q1.flux_lru', amplitude_scale=0.0)" in script
    # The tone is aligned with the drive and the flux line it shares the port with
    assert "align('q1.xy', 'q1.z', 'q1.flux_lru')" in script
