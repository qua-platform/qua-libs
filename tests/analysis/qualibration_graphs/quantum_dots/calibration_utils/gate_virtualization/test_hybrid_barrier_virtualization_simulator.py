"""Unit tests for hybrid barrier virtualization simulator physics assumptions."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


def _repo_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "tests").is_dir() and (current / "qualibration_graphs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate repository root.")


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


REPO_ROOT = _repo_root(Path(__file__).resolve())
TEST_QD_ROOT = REPO_ROOT / "tests" / "analysis" / "qualibration_graphs" / "quantum_dots"

_sim_mod = _load_module(
    TEST_QD_ROOT / "calibration_utils" / "gate_virtualization" / "hybrid_barrier_virtualization_simulator.py",
    "gate_virtualization_hybrid_simulator_unit_test",
)

HybridBarrierSimulationConfig = _sim_mod.HybridBarrierSimulationConfig
HybridBarrierVirtualizationSimulator = _sim_mod.HybridBarrierVirtualizationSimulator
finite_temperature_excess_charge = _sim_mod.finite_temperature_excess_charge


@pytest.mark.analysis
def test_effective_gamma_enforces_nearest_neighbor_locality() -> None:
    barrier_names = ["b12", "b23", "b34", "b45"]
    gamma = np.array(
        [
            [2.0, -0.9, 0.7, -0.5],
            [0.8, 1.8, -0.6, 0.4],
            [-0.3, 0.5, 1.7, -0.7],
            [0.2, -0.4, 0.8, 1.9],
        ],
        dtype=float,
    )
    cfg = HybridBarrierSimulationConfig(
        barrier_names=barrier_names,
        barrier_exponent_matrix=gamma,
        base_tunnel_couplings=np.array([0.08, 0.09, 0.10, 0.11], dtype=float),
        drive_values=np.linspace(-0.01, 0.01, 5, dtype=float),
        detuning_values=np.linspace(-1.0, 1.0, 121, dtype=float),
        nearest_neighbor_gamma_ratio_max=0.25,
        zero_non_nearest_neighbor_gamma=True,
        use_qarray_background=False,
    )
    sim = HybridBarrierVirtualizationSimulator(cfg)
    effective = sim.effective_barrier_exponent_matrix

    assert np.allclose(np.diag(effective), np.diag(gamma))

    for i in range(len(barrier_names)):
        for j in range(len(barrier_names)):
            distance = abs(i - j)
            if distance > 1:
                assert effective[i, j] == 0.0
            elif distance == 1:
                limit = 0.25 * abs(gamma[i, i])
                assert abs(effective[i, j]) <= limit + 1e-12


@pytest.mark.analysis
def test_pair_scan_uses_eq2_tunnel_and_paper_signal_model() -> None:
    barrier_names = ["b12", "b23", "b34"]
    gamma = np.array(
        [
            [7.5, -2.5, 0.1],
            [-2.2, 7.0, -1.8],
            [0.1, -2.0, 7.3],
        ],
        dtype=float,
    )
    offsets = np.array([0.004, -0.003, 0.002], dtype=float)
    drive = np.array([-0.01, 0.0, 0.01], dtype=float)
    detuning = np.linspace(-0.3, 0.3, 121, dtype=float)
    cfg = HybridBarrierSimulationConfig(
        barrier_names=barrier_names,
        barrier_exponent_matrix=gamma,
        base_tunnel_couplings=np.array([0.08, 0.09, 0.10], dtype=float),
        drive_values=drive,
        detuning_values=detuning,
        barrier_dc_offsets=offsets,
        thermal_energy=0.16,
        detuning_center=0.01,
        detuning_center_drive_factor=0.0,
        use_paper_signal_model=True,
        paper_signal_v0=0.30,
        paper_signal_delta_v=1.15,
        paper_signal_s0=0.04,
        paper_signal_s1=-0.01,
        qarray_background_weight=0.0,
        analytic_background_weight=0.0,
        noise_std=0.0,
        use_qarray_background=False,
    )
    sim = HybridBarrierVirtualizationSimulator(cfg)
    ds, truth = sim.generate_pair_scan(target_barrier="b23", drive_barrier="b12")

    i = 1
    j = 0
    gamma_row = sim.effective_barrier_exponent_matrix[i, :]
    dB = np.repeat(offsets[None, :], drive.size, axis=0)
    dB[:, j] = offsets[j] + drive
    expected_tunnel = cfg.base_tunnel_couplings[i] * np.exp(dB @ gamma_row)
    assert np.allclose(ds["tunnel_truth"].values, expected_tunnel, atol=1e-12)

    for row, drive_value in enumerate(drive):
        center = cfg.detuning_center + cfg.detuning_center_drive_factor * float(drive_value)
        transition = finite_temperature_excess_charge(
            detuning,
            tunnel_coupling=expected_tunnel[row],
            thermal_energy=cfg.thermal_energy,
            center=center,
        )
        detuning_centered = detuning - center
        expected_signal = (
            cfg.paper_signal_v0
            + cfg.paper_signal_delta_v * transition
            + (cfg.paper_signal_s0 + (cfg.paper_signal_s1 - cfg.paper_signal_s0) * transition) * detuning_centered
        )
        assert np.allclose(ds["amplitude_truth"].values[row], expected_signal, atol=1e-12)

    expected_t_at_zero = cfg.base_tunnel_couplings[i] * np.exp(float(np.dot(gamma_row, offsets)))
    expected_slope = float(gamma_row[j]) * expected_t_at_zero
    assert np.isclose(float(truth["dt_dB_at_zero"]), expected_slope, atol=1e-12)
    assert float(truth["uses_paper_signal_model"]) == 1.0
