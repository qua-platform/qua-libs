"""Fixtures for gate-virtualization analysis tests using qarray.

These tests generate synthetic 2D scan datasets via the qarray
``ChargeSensedDotArray`` model and then run the node's ``analyse_data``,
``plot_data``, ``update_state``, and ``save_results`` actions against
that data -- without requiring a real QOP or QDAC connection.

Uses the unified wiring-based QuAM factory (``create_qd_quam``) and
shared test helpers from ``shared_fixtures``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

# qualibrate 1.1.x moved parameters into core.parameters; qualibration-libs
# still imports from qualibrate.core.parameters.
if "qualibrate.parameters" not in sys.modules:
    try:
        import qualibrate.core.parameters as _cp

        sys.modules["qualibrate.parameters"] = _cp
    except ImportError:
        pass

os.environ.setdefault("QUAM_STATE_PATH", "/tmp/quam_test_state")

import numpy as np
import pytest
import xarray as xr

DEFAULT_NOISE_STD = 0.01
DEFAULT_PINK_STD = 0.01
DEFAULT_BROWN_STD = 0.01


def _colored_noise(
    rng: np.random.Generator,
    shape: tuple,
    pink_std: float = 0.0,
    brown_std: float = 0.0,
) -> np.ndarray:
    """Generate pink (1/f) and brown (1/f²) noise via FFT spectral shaping."""
    n = int(np.prod(shape))
    noise = np.zeros(n)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0
    if pink_std > 0.0:
        white = rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        spectrum /= np.sqrt(freqs)
        spectrum[0] = 0.0
        pink = np.fft.irfft(spectrum, n=n)
        std = pink.std()
        if std > 0.0:
            noise += pink_std * pink / std
    if brown_std > 0.0:
        white = rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        spectrum /= freqs
        spectrum[0] = 0.0
        brown = np.fft.irfft(spectrum, n=n)
        std = brown.std()
        if std > 0.0:
            noise += brown_std * brown / std
    return noise.reshape(shape)


def apply_sensor_noise(
    signal: np.ndarray,
    *,
    noise_std: float = DEFAULT_NOISE_STD,
    pink_std: float = DEFAULT_PINK_STD,
    brown_std: float = DEFAULT_BROWN_STD,
    seed: int = 42,
) -> np.ndarray:
    """Apply white + colored noise to a sensor signal array."""
    if noise_std > 0 or pink_std > 0 or brown_std > 0:
        rng = np.random.default_rng(seed=seed)
        if noise_std > 0:
            signal = signal + rng.normal(0, noise_std, size=signal.shape)
        if pink_std > 0 or brown_std > 0:
            signal = signal + _colored_noise(rng, signal.shape, pink_std, brown_std)
    return signal


CURRENT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = CURRENT_DIR.parents[3]  # tests/analysis/

# ── Shared helpers ─────────────────────────────────────────────────────
_SHARED_DIR = (
    Path(__file__).resolve().parents[5]
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
)
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_fixtures import (  # noqa: E402
    REPO_ROOT,
    apply_param_overrides,
    call_node_action,
    ensure_qua_dashboards_stub,
    make_save_analysis_plot,
    patch_action_manager_register_only,
    patch_qualibrate_logger,
    reimport_node_to_register_actions,
    setup_test_cache,
)
from quam_factory import create_qd_quam  # noqa: E402

# Backward-compatible aliases used by test modules importing from .conftest
_call_node_action = call_node_action
_patch_action_manager_register_only = patch_action_manager_register_only
_reimport_node_to_register_actions = reimport_node_to_register_actions

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = ANALYSIS_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

import matplotlib  # noqa: E402

matplotlib.use("Agg")

# ── Paths ──────────────────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "gate_virtualization"
)
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

_QUANTUM_DOTS_DIR = REPO_ROOT / "qualibration_graphs" / "quantum_dots"
if str(_QUANTUM_DOTS_DIR) not in sys.path:
    sys.path.insert(0, str(_QUANTUM_DOTS_DIR))

from validation_utils.charge_stability.default import init_dot_model  # noqa: E402
from .simulation_helpers import (
    simulate_sensor_device_scan,
    sweep_voltages_mV,
)  # noqa: E402

# ── Calibrated sensor compensation coefficients ───────────────────────
# Extracted from node 01 analysis with the asymmetric model below
# (Cds dot_0=0.003, dot_1=0.0015; Cgs gate_0=0.0015, gate_1=0.001).
CALIBRATED_SENSOR_COMP = {0: -0.0290778656, 1: -0.0184242463}


# ── qarray model fixtures ─────────────────────────────────────────────


@pytest.fixture
def dot_model():
    """Return a fully configured qarray ``ChargeSensedDotArray`` (6 dots + 1 sensor).

    Dot-sensor couplings are asymmetric so the two device dots produce
    visibly different sensor responses (dot_0 couples more strongly).
    """
    return init_dot_model(
        Cds=[[0.003, 0.0015, 0.002, 0.002, 0.002, 0.002]],
        Cgs=[[0.0015, 0.001, 0.000, 0.000, 0.000, 0.000, 0.100]],
    )


# ── Simulation helpers ─────────────────────────────────────────────────


def simulate_plunger_plunger_scan(
    model,
    v_plunger_x: np.ndarray,
    v_plunger_y: np.ndarray,
    plunger_x_gate_idx: int = 0,
    plunger_y_gate_idx: int = 1,
    *,
    sensor_gate_idx: int = 6,
    sensor_operating_point: float = 14.5,
    base_voltages: Optional[np.ndarray] = None,
    sensor_compensation: Optional[Dict[int, float]] = None,
    noise_std: float = DEFAULT_NOISE_STD,
    pink_std: float = DEFAULT_PINK_STD,
    brown_std: float = DEFAULT_BROWN_STD,
    seed: int = 42,
) -> xr.Dataset:
    """Simulate a 2D plunger-plunger charge stability scan using qarray."""
    n_gates = max(plunger_x_gate_idx, plunger_y_gate_idx, sensor_gate_idx) + 1
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    cx = (v_plunger_x[0] + v_plunger_x[-1]) / 2
    cy = (v_plunger_y[0] + v_plunger_y[-1]) / 2

    sensor_is_x = sensor_gate_idx == plunger_x_gate_idx
    sensor_is_y = sensor_gate_idx == plunger_y_gate_idx

    rows = []
    for vy in v_plunger_y:
        for vx in v_plunger_x:
            v = base_voltages.copy()
            v[plunger_x_gate_idx] = vx
            v[plunger_y_gate_idx] = vy

            if sensor_is_x or sensor_is_y:
                # Sensor IS one of the sweep axes.  The sweep value is already
                # set above; add sensor compensation for the *other* (non-sensor)
                # axis so the Coulomb peak tracks plunger movement.
                if sensor_compensation:
                    if sensor_is_y:
                        alpha_x = sensor_compensation.get(plunger_x_gate_idx, 0.0)
                        v[sensor_gate_idx] += alpha_x * (vx - cx)
                    else:
                        alpha_y = sensor_compensation.get(plunger_y_gate_idx, 0.0)
                        v[sensor_gate_idx] += alpha_y * (vy - cy)
            else:
                s_v = sensor_operating_point
                if sensor_compensation:
                    alpha_x = sensor_compensation.get(plunger_x_gate_idx, 0.0)
                    alpha_y = sensor_compensation.get(plunger_y_gate_idx, 0.0)
                    s_v += alpha_x * (vx - cx) + alpha_y * (vy - cy)
                v[sensor_gate_idx] = s_v

            rows.append(v)

    voltage_array = np.array(rows)
    z, _ = model.charge_sensor_open(-voltage_array)
    z = np.asarray(z).squeeze()
    signal_2d = z.reshape(len(v_plunger_y), len(v_plunger_x))

    if noise_std > 0 or pink_std > 0 or brown_std > 0:
        rng = np.random.default_rng(seed=seed)
        if noise_std > 0:
            signal_2d = signal_2d + rng.normal(0, noise_std, size=signal_2d.shape)
        if pink_std > 0 or brown_std > 0:
            signal_2d = signal_2d + _colored_noise(rng, signal_2d.shape, pink_std, brown_std)

    v_x_V = v_plunger_x * 1e-3
    v_y_V = v_plunger_y * 1e-3
    I_data = signal_2d[np.newaxis, :, :]
    Q_data = np.zeros_like(I_data)

    return xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_x_V, "y_volts": v_y_V},
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_x_V, "y_volts": v_y_V},
            ),
        }
    )


def simulate_sensor_sweep(
    model,
    v_sensor_mV: np.ndarray,
    *,
    sensor_gate_idx: int = 6,
    base_voltages: Optional[np.ndarray] = None,
    noise_std: float = DEFAULT_NOISE_STD,
    pink_std: float = DEFAULT_PINK_STD,
    brown_std: float = DEFAULT_BROWN_STD,
    seed: int = 42,
) -> xr.Dataset:
    """Simulate a 1D sensor gate sweep with other gates at *base_voltages*."""
    n_gates = sensor_gate_idx + 1
    if base_voltages is None:
        base_voltages = np.zeros(n_gates)

    voltage_array = np.tile(base_voltages, (len(v_sensor_mV), 1))
    voltage_array[:, sensor_gate_idx] = v_sensor_mV

    z, _ = model.charge_sensor_open(-voltage_array)
    z = np.asarray(z).squeeze()

    if noise_std > 0 or pink_std > 0 or brown_std > 0:
        rng = np.random.default_rng(seed=seed)
        if noise_std > 0:
            z = z + rng.normal(0, noise_std, size=z.shape)
        if pink_std > 0 or brown_std > 0:
            z = z + _colored_noise(rng, z.shape, pink_std, brown_std)
    z = np.clip(z, 0.0, 1.0)

    v_V = v_sensor_mV * 1e-3
    I_data = z[np.newaxis, :]
    Q_data = np.zeros_like(I_data)

    return xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "x_volts"],
                coords={"sensors": ["sensor_1"], "x_volts": v_V},
            ),
        }
    )


# ── QuAM factory fixture ──────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a Stage-1 ``BaseQuamQD``."""

    def _factory():
        return create_qd_quam()

    return _factory


# ── Analysis runner fixture ────────────────────────────────────────────


@pytest.fixture
def save_analysis_plot():
    return make_save_analysis_plot()


@pytest.fixture
def analysis_runner(minimal_quam_factory, save_analysis_plot):
    """Run an analysis e2e test for a gate-virtualization node."""

    def _run(
        node_name: str,
        ds_raw_all: Dict[str, xr.Dataset],
        *,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
    ) -> Any:
        from quam_config import Quam

        ensure_qua_dashboards_stub()
        machine = minimal_quam_factory()

        with (
            patch.object(Quam, "load", return_value=machine),
            patch_action_manager_register_only(),
        ):
            node = reimport_node_to_register_actions(
                node_name, CALIBRATION_LIBRARY_ROOT
            )
        if node is None:
            pytest.fail(
                f"Could not load node '{node_name}' from {CALIBRATION_LIBRARY_ROOT}"
            )

        node.machine = machine

        overrides = dict(param_overrides) if param_overrides else {}
        overrides["simulate"] = False
        apply_param_overrides(node, overrides)

        node.results["ds_raw_all"] = ds_raw_all

        call_node_action(node, "analyse_data")
        call_node_action(node, "plot_data")
        action_names = set(
            getattr(getattr(node, "_action_manager", None), "actions", {}).keys()
        )
        if "update_state" in action_names:
            call_node_action(node, "update_state")
        call_node_action(node, "save_results")

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        figures = node.results.get("figures", {})
        if len(figures) == 1:
            save_analysis_plot(next(iter(figures.values())), artifacts_dir)
        else:
            for pair_key, fig in figures.items():
                safe_name = pair_key.replace("/", "_")
                save_analysis_plot(fig, artifacts_dir, filename=f"{safe_name}.png")

        return node

    return _run


# ── Pytest hooks ───────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "analysis: mark test as an analysis test using qarray simulation",
    )
