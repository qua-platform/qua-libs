"""Offline analysis tests for gate_virtualization barrier-barrier method."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import xarray as xr


def _repo_root(start: Path) -> Path:
    current = start
    while current != current.parent:
        if (current / "tests").is_dir() and (current / "qualibration_graphs").is_dir():
            return current
        current = current.parent
    raise FileNotFoundError("Could not locate repository root.")


REPO_ROOT = _repo_root(Path(__file__).resolve())
QD_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots"
if str(QD_ROOT) not in sys.path:
    sys.path.insert(0, str(QD_ROOT))


def _load_module(module_path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_analysis_mod = _load_module(
    QD_ROOT / "calibration_utils" / "gate_virtualization" / "analysis.py",
    "gate_virtualization_analysis_test",
)
_barrier_mod = _load_module(
    QD_ROOT / "calibration_utils" / "gate_virtualization" / "barrier_compensation_analysis.py",
    "gate_virtualization_barrier_analysis_test",
)

update_compensation_submatrix = _analysis_mod.update_compensation_submatrix
assemble_slope_matrix = _barrier_mod.assemble_slope_matrix
calibrate_stepwise_barrier_virtualization = _barrier_mod.calibrate_stepwise_barrier_virtualization
compute_residual_crosstalk_ratios = _barrier_mod.compute_residual_crosstalk_ratios
extract_barrier_compensation_coefficients = _barrier_mod.extract_barrier_compensation_coefficients
finite_temperature_excess_charge = _barrier_mod.finite_temperature_excess_charge
fit_finite_temperature_two_level = _barrier_mod.fit_finite_temperature_two_level
fit_barrier_cross_talk = _barrier_mod.fit_barrier_cross_talk


class _DummyLayer:
    def __init__(self, layer_id: str, gates: list[str], matrix: np.ndarray):
        self.id = layer_id
        self.source_gates = list(gates)
        self.target_gates = list(gates)
        self.matrix = np.asarray(matrix, dtype=float).tolist()


class _DummyGateSet:
    def __init__(self, gate_set_id: str, layer: _DummyLayer):
        self.id = gate_set_id
        self.layers = [layer]


class _DummyMachine:
    def __init__(self, gate_set: _DummyGateSet):
        self.virtual_gate_sets = {gate_set.id: gate_set}


class _DummyNode:
    def __init__(self, matrix: np.ndarray, gates: list[str]):
        layer = _DummyLayer("layer0", gates, matrix)
        gate_set = _DummyGateSet("main_qpu", layer)
        self.machine = _DummyMachine(gate_set)
        self.parameters = SimpleNamespace(virtual_gate_set_id="main_qpu", matrix_layer_id=None)


def _make_synthetic_pair_dataset(
    slope: float,
    detuning: np.ndarray,
    drive_values: np.ndarray,
    t0: float = 0.08,
    thermal_energy: float = 0.15,
    center: float = 0.0,
) -> xr.Dataset:
    traces = []
    for drive in drive_values:
        tunnel = t0 + slope * drive
        base = finite_temperature_excess_charge(detuning, tunnel, thermal_energy, center)
        traces.append(0.2 + 1.1 * base)
    signal = np.asarray(traces, dtype=float)
    return xr.Dataset(
        data_vars={"amplitude": (("x_volts", "y_volts"), signal)},
        coords={"x_volts": drive_values, "y_volts": detuning},
    )


@pytest.mark.analysis
def test_finite_temperature_fit_recovers_tunnel_coupling() -> None:
    detuning = np.linspace(-1.5, 1.5, 501, dtype=float)
    true_t = 0.11
    true_kbt = 0.17
    true_center = 0.03

    base = finite_temperature_excess_charge(detuning, true_t, true_kbt, true_center)
    rng = np.random.default_rng(123)
    signal = 0.3 + 1.4 * base + rng.normal(0.0, 0.004, size=base.shape)

    fit = fit_finite_temperature_two_level(detuning, signal)
    assert fit["success"], fit
    assert np.isfinite(fit["tunnel_coupling"])
    assert abs(fit["tunnel_coupling"] - true_t) < 0.04
    assert fit["fit_quality"] > 0.9


@pytest.mark.analysis
def test_barrier_slope_fit_recovers_dt_dB() -> None:
    detuning = np.linspace(-1.0, 1.0, 321, dtype=float)
    drive = np.linspace(-0.02, 0.02, 11, dtype=float)
    true_slope = 0.65

    ds = _make_synthetic_pair_dataset(slope=true_slope, detuning=detuning, drive_values=drive)
    fit = fit_barrier_cross_talk(ds, "x_volts", "y_volts")

    assert fit["success"], fit
    assert np.isfinite(fit["coefficient"])
    assert abs(fit["coefficient"] - true_slope) < 0.2
    assert fit["fit_quality"] > 0.7
    assert fit["n_points"] >= 8


@pytest.mark.analysis
def test_stepwise_transform_reduces_residual_crosstalk() -> None:
    barrier_names = ["barrier_12", "barrier_23", "barrier_34"]
    slope_matrix = np.array(
        [
            [1.0, -0.45, 0.15],
            [-0.35, 1.0, -0.28],
            [0.20, -0.42, 1.0],
        ],
        dtype=float,
    )

    initial = compute_residual_crosstalk_ratios(slope_matrix, barrier_names)
    result = calibrate_stepwise_barrier_virtualization(
        slope_matrix_raw=slope_matrix,
        barrier_names=barrier_names,
        calibration_order=barrier_names,
        residual_target=0.10,
        max_refinement_rounds=3,
    )

    final_transform = np.asarray(result["barrier_transform_final"], dtype=float)
    assert np.allclose(np.diag(final_transform), 1.0, atol=1e-12)
    assert result["max_residual_crosstalk"] <= initial["max_ratio"]
    assert result["barrier_transform_history"][-1]["label"] == "B†"


@pytest.mark.analysis
def test_update_compensation_submatrix_updates_only_barrier_block() -> None:
    gates = ["plunger_1", "barrier_12", "barrier_23", "barrier_34", "sensor_1"]
    base = np.eye(5, dtype=float)
    base[0, 4] = 0.123
    base[4, 0] = -0.321

    node = _DummyNode(matrix=base, gates=gates)
    barrier_names = ["barrier_12", "barrier_23", "barrier_34"]
    barrier_block = np.array(
        [
            [1.0, -0.80, 0.19],
            [-0.36, 1.0, -0.24],
            [0.23, -0.64, 1.0],
        ],
        dtype=float,
    )

    update_meta = update_compensation_submatrix(
        node=node,
        row_names=barrier_names,
        col_names=barrier_names,
        values=barrier_block,
        layer_id="layer0",
    )

    updated = np.asarray(node.machine.virtual_gate_sets["main_qpu"].layers[0].matrix, dtype=float)
    assert update_meta["layer_id"] == "layer0"

    barrier_idx = [gates.index(name) for name in barrier_names]
    assert np.allclose(updated[np.ix_(barrier_idx, barrier_idx)], barrier_block)

    # Non-barrier entries are preserved.
    assert np.isclose(updated[0, 4], 0.123)
    assert np.isclose(updated[4, 0], -0.321)


@pytest.mark.analysis
def test_offline_pipeline_synthetic_pair_scans() -> None:
    barrier_names = ["barrier_12", "barrier_23", "barrier_34"]
    slope_truth = np.array(
        [
            [1.00, -0.40, 0.12],
            [-0.30, 1.00, -0.20],
            [0.18, -0.35, 1.00],
        ],
        dtype=float,
    )

    detuning = np.linspace(-1.1, 1.1, 241, dtype=float)
    drive_values = np.linspace(-0.02, 0.02, 9, dtype=float)

    fit_results: dict[str, dict[str, float]] = {}
    for i, target in enumerate(barrier_names):
        for j, drive in enumerate(barrier_names):
            ds = _make_synthetic_pair_dataset(
                slope=float(slope_truth[i, j]),
                detuning=detuning,
                drive_values=drive_values,
                t0=0.09,
                thermal_energy=0.14,
                center=0.01,
            )
            pair_key = f"{target}_vs_{drive}"
            fit = extract_barrier_compensation_coefficients(ds, drive, target)
            fit["target_barrier"] = target
            fit["drive_barrier"] = drive
            fit_results[pair_key] = fit

    slope_matrix_raw = assemble_slope_matrix(fit_results, barrier_names)
    assert slope_matrix_raw.shape == (3, 3)
    assert np.all(np.isfinite(np.diag(slope_matrix_raw)))

    result = calibrate_stepwise_barrier_virtualization(
        slope_matrix_raw=slope_matrix_raw,
        barrier_names=barrier_names,
        calibration_order=barrier_names,
        residual_target=0.10,
        max_refinement_rounds=3,
    )

    assert np.isfinite(result["max_residual_crosstalk"])
    assert result["max_residual_crosstalk"] <= 0.10
    assert len(result["barrier_transform_history"]) >= len(barrier_names) + 1


@pytest.mark.analysis
def test_failure_on_near_zero_self_slope() -> None:
    barrier_names = ["barrier_12", "barrier_23"]
    slope_matrix = np.array([[0.0, 0.1], [0.2, 1.0]], dtype=float)

    with pytest.raises(ValueError, match="Invalid self slope"):
        calibrate_stepwise_barrier_virtualization(
            slope_matrix_raw=slope_matrix,
            barrier_names=barrier_names,
            calibration_order=barrier_names,
            min_abs_self_slope=1e-9,
        )
