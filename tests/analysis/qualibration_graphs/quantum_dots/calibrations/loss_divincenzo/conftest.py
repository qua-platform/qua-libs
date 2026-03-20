"""Fixtures for Loss-DiVincenzo analysis tests using virtual_qpu.

These tests generate synthetic ``ds_raw`` datasets via physics simulation
(virtual_qpu / dynamiqs) and then run the node's ``analyse_data``,
``plot_data``, and ``update_state`` actions -- without requiring a real
QOP connection.

Uses the unified wiring-based QuAM factory (``create_ld_quam``) and
shared test helpers from ``shared_fixtures``.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

CURRENT_DIR = Path(__file__).resolve().parent
ANALYSIS_ROOT = CURRENT_DIR.parents[3]  # tests/analysis/

# ── Shared helpers ─────────────────────────────────────────────────────
_SHARED_DIR = Path(__file__).resolve().parents[5] / "qualibration_graphs" / "quantum_dots" / "calibrations"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))

from shared_fixtures import (  # noqa: E402
    REPO_ROOT,
    apply_param_overrides,
    call_node_action,
    ensure_quam_config_stub,
    get_parameters_dict,
    load_library_node,
    make_save_analysis_plot,
    patch_action_manager_register_only,
    patch_qualibrate_logger,
    reimport_node_to_register_actions,
    setup_test_cache,
)
from quam_factory import create_ld_quam  # noqa: E402

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = ANALYSIS_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = REPO_ROOT / "qualibration_graphs" / "quantum_dots" / "calibrations" / "loss_divincenzo"
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

QUBIT_NAMES: list[str] = ["q1", "q2", "q3", "q4"]
ANALYSE_QUBITS: list[str] = ["q1"]

# ── virtual_qpu path setup ────────────────────────────────────────────
VIRTUAL_QPU_ROOT = REPO_ROOT.parent / "virtual_qpu"
_vqpu_src = str(VIRTUAL_QPU_ROOT / "src")
_vqpu_platforms = str(VIRTUAL_QPU_ROOT / "platforms")
if _vqpu_src not in sys.path:
    sys.path.insert(0, _vqpu_src)
if _vqpu_platforms not in sys.path:
    sys.path.insert(0, _vqpu_platforms)

# ── virtual_qpu imports (lazy) ─────────────────────────────────────────

import jax.numpy as jnp  # noqa: E402

from virtual_qpu.dynamics import simulate as _simulate  # noqa: E402
from virtual_qpu.operators import expval as _expval  # noqa: E402
from virtual_qpu.sweep import sweep as _sweep  # noqa: E402

from quantum_dots.device import LossDiVincenzoDevice  # noqa: E402
from quantum_dots.params import ExchangeModel, LossDiVincenzoParams, MU_B_OVER_H  # noqa: E402

_VIRTUAL_QPU_AVAILABLE = True

# ── Default device configuration ──────────────────────────────────────

DEFAULT_LD_PARAMS = LossDiVincenzoParams(
    n_qubits=2,
    g_factors=[2.0, 2.04],
    magnetic_field=10.0 / (2.0 * MU_B_OVER_H),
    exchange_models=[ExchangeModel(J_0=0.001, V_ref=0.0, lever_arm=0.050)],
    ref_freqs=None,
    frame="rot",
    use_rwa=True,
    t1=[1000.0, 1000.0],
    t2=[400.0, 400.0],
)

DEFAULT_SOLVER = "me"
DEFAULT_NOISE_STD = 0.1
DEFAULT_DRIVE_AMP_GHZ = 0.008
DEFAULT_PULSE_DURATION_NS = 100.0


# ── Simulation helpers ─────────────────────────────────────────────────


def simulate_sweep(
    device: LossDiVincenzoDevice,
    make_schedule: Any,
    tsave: Any,
    *,
    observable_qubit: int = 0,
    observable_state: int = 1,
    solver: str = DEFAULT_SOLVER,
    noise_std: float = DEFAULT_NOISE_STD,
    seed: int = 42,
    **sweep_axes: Any,
) -> np.ndarray:
    """Run a vectorised parameter sweep and return expectation values."""
    dim = 2
    psi0 = device.ground_state()
    jump_ops = device.collapse_operators() if solver == "me" else None
    tsave_is_callable = callable(tsave)

    local_proj = jnp.zeros((dim, dim), dtype=jnp.complex64)
    local_proj = local_proj.at[observable_state, observable_state].set(1.0)
    observable = device.embed(local_proj, mode=observable_qubit)

    def _inner(**kwargs):
        resolved = make_schedule(**kwargs)
        H_t = device.hamiltonian(resolved)
        ts = tsave(**kwargs) if tsave_is_callable else tsave
        sol = _simulate(H_t, psi0, ts, solver=solver, jump_ops=jump_ops)
        return _expval(sol.states, observable)

    result = np.asarray(_sweep(_inner, **sweep_axes))

    if noise_std > 0:
        rng = np.random.default_rng(seed=seed)
        result = result + rng.normal(0, noise_std, size=result.shape)
        result = np.clip(result, 0.0, 1.0)
    return result


def build_parity_ds_raw(
    coords: Dict[str, tuple],
    pdiff_per_qubit: Dict[str, np.ndarray],
    qubit_names: Optional[list[str]] = None,
) -> xr.Dataset:
    """Build an ``xarray.Dataset`` in ``execute_qua_program`` parity-diff format."""
    qubit_names = qubit_names or QUBIT_NAMES
    dim_names = list(coords.keys())
    shape = tuple(len(coords[d][0]) for d in dim_names)

    data_vars: Dict[str, Any] = {}
    for qname in qubit_names:
        pd = pdiff_per_qubit.get(qname, np.zeros(shape))
        data_vars[f"p1_{qname}"] = xr.DataArray(np.zeros_like(pd), dims=dim_names)
        data_vars[f"p2_{qname}"] = xr.DataArray(pd, dims=dim_names)
        data_vars[f"pdiff_{qname}"] = xr.DataArray(pd, dims=dim_names)

    xr_coords = {
        name: xr.DataArray(vals, dims=name, attrs={"long_name": long, "units": units})
        for name, (vals, long, units) in coords.items()
    }

    return xr.Dataset(data_vars, coords=xr_coords, attrs={"qubit_names": qubit_names})


# ── Fixtures ───────────────────────────────────────────────────────────


@pytest.fixture
def minimal_quam_factory():
    """Factory fixture returning a ``LossDiVincenzoQuam`` with default macros."""

    def _factory():
        return create_ld_quam()

    return _factory


@pytest.fixture
def ld_device():
    """A pre-configured ``LossDiVincenzoDevice`` with default parameters."""
    device = LossDiVincenzoDevice(params=DEFAULT_LD_PARAMS)
    jump_ops = device.collapse_operators()
    n_expected = 2 * DEFAULT_LD_PARAMS.n_qubits
    assert len(jump_ops) == n_expected, f"Expected {n_expected} collapse ops, got {len(jump_ops)}"
    return device


@pytest.fixture(scope="session")
def calibrated_pi_half_amp():
    """Calibrate the pi/2 pulse amplitude via a quick power-Rabi sweep."""
    from virtual_qpu.pulse import GaussianIQPulse
    from virtual_qpu.schedule import Schedule

    device = LossDiVincenzoDevice(params=DEFAULT_LD_PARAMS)
    qubit_freq_ghz = device.params.qubit_freqs[0]

    amp_prefactors = jnp.linspace(0.1, 3.0, 200, dtype=jnp.float32)

    def make_schedule(amp):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=DEFAULT_PULSE_DURATION_NS,
                amplitude=DEFAULT_DRIVE_AMP_GHZ * amp,
                frequency=qubit_freq_ghz,
                sigma=DEFAULT_PULSE_DURATION_NS / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    result = simulate_sweep(
        device,
        make_schedule,
        tsave=jnp.array([0.0, DEFAULT_PULSE_DURATION_NS], dtype=jnp.float32),
        noise_std=0.0,
        amp=amp_prefactors,
    )
    parity = np.asarray(result[..., -1])
    pi_idx = int(np.argmax(parity))
    pi_amp = float(DEFAULT_DRIVE_AMP_GHZ * amp_prefactors[pi_idx])
    return pi_amp / 2.0


@pytest.fixture(scope="session")
def rabi_chevron_calibration():
    """Run a Rabi chevron simulation + FFT analysis to calibrate pulse parameters."""
    if not _VIRTUAL_QPU_AVAILABLE:
        pytest.skip("virtual_qpu (dynamiqs) not installed")

    from virtual_qpu.pulse import GaussianIQPulse
    from virtual_qpu.schedule import Schedule
    from calibration_utils.time_rabi_chevron_parity_diff.analysis import (
        _fft_analyse_single_qubit,
    )

    device = LossDiVincenzoDevice(params=DEFAULT_LD_PARAMS)
    qubit_freq_ghz = device.params.qubit_freqs[0]

    max_dur_ns = 800
    n_dur = 200
    freq_span_mhz = 100.0
    freq_step_mhz = 1.0

    durations = jnp.linspace(4, max_dur_ns, n_dur, dtype=jnp.float32)
    span_ghz = freq_span_mhz * 1e-3
    step_ghz = freq_step_mhz * 1e-3
    drive_freqs = jnp.arange(
        qubit_freq_ghz - span_ghz / 2,
        qubit_freq_ghz + span_ghz / 2,
        step_ghz,
        dtype=jnp.float32,
    )

    def make_schedule(freq, dur):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=dur,
                amplitude=DEFAULT_DRIVE_AMP_GHZ,
                frequency=freq,
                sigma=dur / 5,
            ),
            channel="drive_q0",
        )
        return sched.resolve()

    result = simulate_sweep(
        device,
        make_schedule,
        tsave=lambda dur, **_: jnp.array([0.0, dur]),
        noise_std=0.0,
        freq=drive_freqs,
        dur=durations,
    )
    pdiff = result[..., -1]

    freqs_hz = np.asarray(drive_freqs) * 1e9
    durations_ns = np.asarray(durations)
    nominal_freq_hz = float(qubit_freq_ghz * 1e9)

    fit_result, _ = _fft_analyse_single_qubit(pdiff, freqs_hz, durations_ns, nominal_freq_hz)
    assert fit_result["success"], f"Rabi chevron calibration failed: {fit_result}"
    return fit_result


# ── Markdown generator ─────────────────────────────────────────────────


@pytest.fixture
def markdown_generator():
    """Fixture that generates README.md documentation for a node."""

    def _generate(node, parameters_dict, artifacts_dir):
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        params_table = [
            "| Parameter | Value | Description |",
            "|-----------|-------|-------------|",
        ]
        for name, value in parameters_dict.items():
            doc = ""
            node_params = getattr(node, "parameters", None)
            if node_params is not None and hasattr(node_params, "__class__"):
                for cls in node_params.__class__.__mro__:
                    if hasattr(cls, "__annotations__") and name in cls.__annotations__:
                        if hasattr(cls, "__pydantic_fields__"):
                            fi = cls.__pydantic_fields__.get(name)
                            if fi and fi.description:
                                doc = fi.description
                                break
            params_table.append(f"| `{name}` | `{value}` | {doc} |")

        fit_results = node.results.get("fit_results", {})
        fit_section = ""
        if fit_results:
            fit_rows = [
                "| Qubit | f_res (GHz) | t_pi (ns) | Omega_R (rad/ns) | gamma (1/ns) | T2* (ns) | success |",
                "|-------|-------------|----------|--------------|----------|----------|--------|",
            ]
            for qname, r in sorted(fit_results.items()):
                f_ghz = r.get("optimal_frequency", 0) * 1e-9
                t_pi = r.get("optimal_duration", float("nan"))
                omega = r.get("rabi_frequency", float("nan"))
                gamma = r.get("decay_rate", float("nan"))
                t2_star = 1.0 / gamma if gamma > 0 else float("inf")
                succ = r.get("success", False)
                fit_rows.append(
                    f"| {qname} | {f_ghz:.4f} | {t_pi:.1f} | {omega:.6f} | {gamma:.5f} | {t2_star:.0f} | {succ} |"
                )
            fit_section = "\n## Fit Results\n\n" + "\n".join(fit_rows)

        state_section = ""
        if fit_results:
            op_name = getattr(getattr(node, "parameters", None), "operation", "x180")
            state_rows = [
                f"| Qubit | intermediate_frequency (Hz) | xy.operations.{op_name}.length (ns) |",
                "|-------|-----------------------------|-----------------------------------------|",
            ]
            for qname, r in sorted(fit_results.items()):
                if not r.get("success", False):
                    continue
                f_hz = r.get("optimal_frequency", 0)
                t_pi = r.get("optimal_duration", float("nan"))
                state_rows.append(f"| {qname} | {f_hz:.0f} | {t_pi:.1f} |")
            if len(state_rows) > 2:
                state_section = "\n## Updated State\n\n" + "\n".join(state_rows)

        content = (
            f"# {getattr(node, 'name', 'Unknown Node')}\n\n"
            f"## Description\n\n"
            f"{getattr(node, 'description', 'No description available')}\n\n"
            f"## Parameters\n\n" + "\n".join(params_table) + f"\n{fit_section}\n{state_section}\n\n"
            "## Analysis Output\n\n"
            "![Analysis simulation](simulation.png)\n\n"
            "---\n*Generated by analysis test infrastructure (virtual_qpu)*\n"
        )

        output_path = artifacts_dir / "README.md"
        output_path.write_text(content, encoding="utf-8")
        return output_path

    return _generate


# ── Analysis runner fixture ────────────────────────────────────────────


@pytest.fixture
def save_analysis_plot():
    return make_save_analysis_plot()


@pytest.fixture
def analysis_runner(minimal_quam_factory, save_analysis_plot, markdown_generator):
    """Run an analysis test: inject synthetic ds_raw, execute analyse/plot/update."""

    def _run(
        node_name: str,
        ds_raw: Any,
        fig: Any = None,
        param_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        library_root: Optional[Path] = None,
        analyse_qubits: Optional[list[str]] = None,
    ) -> Any:
        if analyse_qubits is None:
            analyse_qubits = ANALYSE_QUBITS
        overrides = dict(param_overrides) if param_overrides else {}
        overrides.setdefault("qubits", analyse_qubits)

        machine = minimal_quam_factory()
        library_root = library_root or CALIBRATION_LIBRARY_ROOT

        ensure_quam_config_stub(machine)
        from quam_config import Quam

        with (
            patch.object(Quam, "load", return_value=machine),
            patch_action_manager_register_only(),
        ):
            node = reimport_node_to_register_actions(node_name, library_root)
            if node is None:
                node = load_library_node(node_name, library_root)
        node.machine = machine

        overrides["simulate"] = False
        apply_param_overrides(node, overrides)

        try:
            from calibration_utils.common_utils.experiment import get_qubits

            node.namespace["qubits"] = get_qubits(node)
        except Exception:
            if hasattr(machine, "qubits"):
                qubits = machine.qubits
                node.namespace["qubits"] = list(qubits.values()) if isinstance(qubits, dict) else list(qubits)

        try:
            from calibration_utils.common_utils.experiment import get_sensors

            node.namespace["sensors"] = get_sensors(node)
        except Exception:
            if hasattr(machine, "sensor_dots"):
                sensors = machine.sensor_dots
                node.namespace["sensors"] = list(sensors.values()) if isinstance(sensors, dict) else list(sensors)

        if analyse_qubits:
            keep_vars = []
            for q in analyse_qubits:
                for prefix in ("p1_", "p2_", "pdiff_"):
                    v = f"{prefix}{q}"
                    if v in ds_raw.data_vars:
                        keep_vars.append(v)
            if keep_vars:
                ds_raw = ds_raw[[v for v in ds_raw.data_vars if v in keep_vars]]

        node.results["ds_raw"] = ds_raw

        call_node_action(node, "analyse_data")
        call_node_action(node, "plot_data")

        if "fit_results" in node.results:
            call_node_action(node, "update_state")

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        fig_to_save = node.results.get("figure") or fig
        if fig_to_save is not None:
            save_analysis_plot(fig_to_save, artifacts_dir, "simulation.png")

        markdown_generator(node, get_parameters_dict(node), artifacts_dir)

        if fig_to_save is not None:
            assert (artifacts_dir / "simulation.png").exists(), "simulation.png not created"
        assert (artifacts_dir / "README.md").exists(), "README.md not created"

        return node

    return _run


# ── Pytest hooks ───────────────────────────────────────────────────────


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "analysis: mark test as an analysis test using virtual_qpu",
    )


def pytest_collection_modifyitems(config, items):
    for item in items:
        if "tests/analysis/" in str(item.fspath) and not item.get_closest_marker("analysis"):
            item.add_marker(pytest.mark.analysis)
