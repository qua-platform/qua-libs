"""Fixtures for Loss-DiVincenzo analysis tests using virtual_qpu.

These tests generate synthetic ``ds_raw`` datasets via physics simulation
(virtual_qpu / dynamiqs) and then run the node's ``analyse_data``,
``plot_data``, and ``update_state`` actions -- without requiring a real
QOP connection.

Uses the unified wiring-based QuAM factory (``create_ld_quam``) and
shared test helpers from ``shared_fixtures``.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Optional
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

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

# Same-directory helper module ``virtual_ld_defaults`` (tests-only defaults).
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

# ── Cache setup ────────────────────────────────────────────────────────
_cache_base = ANALYSIS_ROOT / ".pytest_cache"
setup_test_cache(_cache_base)
patch_qualibrate_logger(_cache_base)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── Paths and defaults ─────────────────────────────────────────────────

CALIBRATION_LIBRARY_ROOT = (
    REPO_ROOT
    / "qualibration_graphs"
    / "quantum_dots"
    / "calibrations"
    / "loss_divincenzo"
)
ARTIFACTS_BASE = ANALYSIS_ROOT / "artifacts"

QUBIT_NAMES: list[str] = ["q1", "q2", "q3", "q4"]
QUBIT_PAIR_NAMES: list[str] = ["q1_q2"]
ANALYSE_QUBITS: list[str] = ["q1"]

# ── virtual_qpu imports (``virtual-qpu`` from git in root pyproject; ``uv sync`` / ``pip install -e .``) ─

import jax.numpy as jnp  # noqa: E402

try:
    from virtual_qpu.dynamics import simulate as _simulate  # noqa: E402
    from virtual_qpu.operators import expval as _expval  # noqa: E402
    from virtual_qpu._sweep import sweep as _sweep  # noqa: E402

    from quantum_dots.device import LossDiVincenzoDevice  # noqa: E402
    from virtual_ld_defaults import default_virtual_ld_params  # noqa: E402

    _VIRTUAL_QPU_AVAILABLE = True
except Exception:  # pragma: no cover — environment without virtual_qpu installed
    _simulate = _expval = _sweep = None  # type: ignore[assignment]
    LossDiVincenzoDevice = None  # type: ignore[assignment]
    default_virtual_ld_params = None  # type: ignore[assignment,misc]
    _VIRTUAL_QPU_AVAILABLE = False

# ── Default device configuration (see ``virtual_ld_defaults.py`` beside this conftest) ─

if _VIRTUAL_QPU_AVAILABLE:
    DEFAULT_LD_PARAMS = default_virtual_ld_params()
else:
    DEFAULT_LD_PARAMS = None  # type: ignore[assignment]


def ld_params_with_decoherence(
    t1_ns: list[float],
    t2_ns: list[float],
) -> Any:
    """Two-qubit params matching ``DEFAULT_LD_PARAMS`` with Lindblad T1/T2."""
    if DEFAULT_LD_PARAMS is None:
        raise RuntimeError(
            "ld_params_with_decoherence requires virtual_qpu / quantum_dots"
        )
    return replace(DEFAULT_LD_PARAMS, t1=t1_ns, t2=t2_ns)


def single_qubit_ld_params(t1_ns: float, t2_ns: float) -> Any:
    """One qubit using qubit-0 Zeeman settings from ``DEFAULT_LD_PARAMS``."""
    if DEFAULT_LD_PARAMS is None:
        raise RuntimeError("single_qubit_ld_params requires virtual_qpu / quantum_dots")
    return replace(
        DEFAULT_LD_PARAMS,
        n_qubits=1,
        g_factors=[DEFAULT_LD_PARAMS.g_factors[0]],
        exchange_models=[],
        t1=[t1_ns],
        t2=[t2_ns],
    )


DEFAULT_SOLVER = "me"
DEFAULT_NOISE_STD = 0.025
DEFAULT_PINK_STD = 0.025
DEFAULT_BROWN_STD = 0.025
DEFAULT_DRIVE_AMP_GHZ = 0.008
DEFAULT_PULSE_DURATION_NS = 100.0

# Parity projector onto the odd-parity (antiparallel spin) subspace.
# P_odd = |01⟩⟨01| + |10⟩⟨10| = (I − Z⊗Z)/2.
# Basis order: |00⟩, |01⟩, |10⟩, |11⟩  (mode-0 ⊗ mode-1).
# Gives 0 for (↑↑, ↓↓) and 1 for (↑↓, ↓↑), distinguishing even from odd parity.
PARITY_PROJECTOR_4x4 = (
    jnp.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0,  0.0, 0.0],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 0.0],
        ],
        dtype=jnp.complex64,
    )
    if _VIRTUAL_QPU_AVAILABLE
    else None
)


# ── Simulation helpers ─────────────────────────────────────────────────


def _colored_noise(
    rng: np.random.Generator,
    shape: tuple,
    pink_std: float = 0.0,
    brown_std: float = 0.0,
) -> np.ndarray:
    """Generate pink (1/f) and/or brown (1/f²) noise with unit-normalised std.

    The noise is generated by shaping independent white noise sequences in the
    frequency domain and transforming back.  Each color uses its own random draw
    so the two contributions are independent.

    Parameters
    ----------
    pink_std
        Target standard deviation of the pink (1/f) component.
    brown_std
        Target standard deviation of the brown (1/f²) component.
    """
    n = int(np.prod(shape))
    noise = np.zeros(n)

    freqs = np.fft.rfftfreq(n)
    freqs[0] = 1.0  # avoid DC division; DC component is zeroed after shaping

    if pink_std > 0.0:
        white = rng.standard_normal(n)
        spectrum = np.fft.rfft(white)
        spectrum /= np.sqrt(freqs)
        spectrum[0] = 0.0  # remove DC offset
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


def simulate_sweep(
    device: LossDiVincenzoDevice,
    make_schedule: Any,
    tsave: Any,
    *,
    observable_qubit: int = 0,
    observable_state: int = 1,
    observable_parity: bool = False,
    solver: str = DEFAULT_SOLVER,
    solver_options: Optional[Dict[str, Any]] = None,
    noise_std: float = DEFAULT_NOISE_STD,
    pink_std: float = DEFAULT_PINK_STD,
    brown_std: float = DEFAULT_BROWN_STD,
    seed: int = 42,
    **sweep_axes: Any,
) -> np.ndarray:
    """Run a vectorised parameter sweep and return expectation values.

    Parameters
    ----------
    observable_parity
        When True, use the two-qubit parity projector P_odd = |01⟩⟨01|+|10⟩⟨10|
        (gives 0 for ↑↑/↓↓, 1 for ↑↓/↓↑) instead of single-qubit |1⟩⟨1|.
    solver_options
        Forwarded as ``options=...`` to :func:`virtual_qpu.dynamics.simulate`
        (e.g. ``{"max_steps": 1_000_000}`` for stiff exchange Hamiltonians).
    noise_std
        Standard deviation of additive white (Gaussian) noise.
    pink_std
        Standard deviation of additive pink (1/f) noise.
    brown_std
        Standard deviation of additive brown (1/f²) noise.
    seed
        Seed for the random number generator (all noise types share the same
        seeded generator so results are reproducible).
    """
    import dynamiqs as _dq

    dim = 2
    psi0 = device.ground_state()
    jump_ops = device.collapse_operators() if solver == "me" else None
    tsave_is_callable = callable(tsave)

    if observable_parity:
        observable = _dq.asqarray(PARITY_PROJECTOR_4x4)
    else:
        local_proj = jnp.zeros((dim, dim), dtype=jnp.complex64)
        local_proj = local_proj.at[observable_state, observable_state].set(1.0)
        observable = device.embed(local_proj, mode=observable_qubit)

    def _inner(**kwargs):
        resolved = make_schedule(**kwargs)
        H_t = device.hamiltonian(resolved)
        ts = tsave(**kwargs) if tsave_is_callable else tsave
        sol = _simulate(
            H_t,
            psi0,
            ts,
            solver=solver,
            jump_ops=jump_ops,
            options=solver_options,
        )
        return _expval(sol.states, observable)

    result = np.asarray(_sweep(_inner, **sweep_axes))

    if noise_std > 0 or pink_std > 0 or brown_std > 0:
        rng = np.random.default_rng(seed=seed)
        if noise_std > 0:
            result = result + rng.normal(0, noise_std, size=result.shape)
        if pink_std > 0 or brown_std > 0:
            result = result + _colored_noise(rng, result.shape, pink_std, brown_std)
    return np.clip(result, 0.0, 1.0)


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


def build_joint_stream_analysis_ds(
    coords: Dict[str, tuple],
    signal_per_qubit: Dict[str, np.ndarray],
    analysis_signal: str = "E_p2_given_p1_0",
    qubit_names: Optional[list[str]] = None,
) -> xr.Dataset:
    """Build ``ds_raw`` in joint-outcome / conditional-expectation format (post-``process_joint_streams``)."""
    qubit_names = qubit_names or QUBIT_NAMES
    dim_names = list(coords.keys())
    shape = tuple(len(coords[d][0]) for d in dim_names)

    data_vars: Dict[str, Any] = {}
    for qname in qubit_names:
        sig = signal_per_qubit.get(qname, np.zeros(shape))
        data_vars[f"p0_p0_{qname}"] = xr.DataArray(
            np.zeros_like(sig, dtype=float), dims=dim_names
        )
        data_vars[f"{analysis_signal}_{qname}"] = xr.DataArray(sig, dims=dim_names)

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
    assert len(jump_ops) > 0, (
        "Default device should have dephasing/relaxation jump operators (T1/T2 set)"
    )
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
def calibrated_target_pi_amp():
    """Calibrate the π pulse amplitude on the target qubit (q0), same Rabi method as control.

    Node ``16_geometric_cz_calibration`` uses ``qubit_target.x90()`` / ``x180()``; this
    matches the virtual-qpu amplitude for a full π rotation on ``drive_q0``.
    """
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
        observable_qubit=0,
        noise_std=0.0,
        amp=amp_prefactors,
    )
    parity = np.asarray(result[..., -1])
    pi_idx = int(np.argmax(parity))
    return float(DEFAULT_DRIVE_AMP_GHZ * amp_prefactors[pi_idx])


@pytest.fixture(scope="session")
def calibrated_control_pi_amp():
    """Calibrate the pi pulse amplitude for the control qubit (qubit 1)."""
    from virtual_qpu.pulse import GaussianIQPulse
    from virtual_qpu.schedule import Schedule

    device = LossDiVincenzoDevice(params=DEFAULT_LD_PARAMS)
    control_freq_ghz = device.params.qubit_freqs[1]

    amp_prefactors = jnp.linspace(0.1, 3.0, 200, dtype=jnp.float32)

    def make_schedule(amp):
        sched = Schedule()
        sched.play(
            GaussianIQPulse(
                duration=DEFAULT_PULSE_DURATION_NS,
                amplitude=DEFAULT_DRIVE_AMP_GHZ * amp,
                frequency=control_freq_ghz,
                sigma=DEFAULT_PULSE_DURATION_NS / 5,
            ),
            channel="drive_q1",
        )
        return sched.resolve()

    result = simulate_sweep(
        device,
        make_schedule,
        tsave=jnp.array([0.0, DEFAULT_PULSE_DURATION_NS], dtype=jnp.float32),
        observable_qubit=1,
        noise_std=0.0,
        amp=amp_prefactors,
    )
    parity = np.asarray(result[..., -1])
    pi_idx = int(np.argmax(parity))
    return float(DEFAULT_DRIVE_AMP_GHZ * amp_prefactors[pi_idx])


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

    fit_result, _ = _fft_analyse_single_qubit(
        pdiff, freqs_hz, durations_ns, nominal_freq_hz
    )
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
            f"## Parameters\n\n"
            + "\n".join(params_table)
            + f"\n{fit_section}\n{state_section}\n\n"
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
        namespace_overrides: Optional[Dict[str, Any]] = None,
        artifacts_subdir: Optional[str] = None,
        library_root: Optional[Path] = None,
        analyse_qubits: Optional[list[str]] = None,
        analyse_qubit_pairs: Optional[list[str]] = None,
    ) -> Any:
        if analyse_qubits is None and analyse_qubit_pairs is None:
            analyse_qubits = ANALYSE_QUBITS
        overrides = dict(param_overrides) if param_overrides else {}
        if analyse_qubits:
            overrides.setdefault("qubits", analyse_qubits)
        if analyse_qubit_pairs:
            overrides.setdefault("qubit_pairs", analyse_qubit_pairs)

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
                node.namespace["qubits"] = (
                    list(qubits.values()) if isinstance(qubits, dict) else list(qubits)
                )

        try:
            from calibration_utils.common_utils.experiment import get_qubit_pairs

            node.namespace["qubit_pairs"] = get_qubit_pairs(node)
        except Exception:
            if hasattr(machine, "qubit_pairs"):
                qp = machine.qubit_pairs
                node.namespace["qubit_pairs"] = (
                    list(qp.values()) if isinstance(qp, dict) else list(qp)
                )

        try:
            from calibration_utils.common_utils.experiment import get_sensors

            node.namespace["sensors"] = get_sensors(node)
        except Exception:
            if hasattr(machine, "sensor_dots"):
                sensors = machine.sensor_dots
                node.namespace["sensors"] = (
                    list(sensors.values())
                    if isinstance(sensors, dict)
                    else list(sensors)
                )

        filter_names = analyse_qubit_pairs or analyse_qubits
        if filter_names:
            keep_vars = []
            for q in filter_names:
                for prefix in (
                    "p1_",
                    "p2_",
                    "pdiff_",
                    "p0_p0_",
                    "p0_p1_",
                    "p1_p0_",
                    "p1_p1_",
                ):
                    v = f"{prefix}{q}"
                    if v in ds_raw.data_vars:
                        keep_vars.append(v)
                v_single = f"p_{q}"
                if v_single in ds_raw.data_vars:
                    keep_vars.append(v_single)
                for sig in ("E_p2_given_p1_0", "E_p2_given_p1_1"):
                    for variant in ("", "_ctrl"):
                        v = f"{sig}{variant}_{q}"
                        if v in ds_raw.data_vars:
                            keep_vars.append(v)
            if keep_vars:
                ds_raw = ds_raw[[v for v in ds_raw.data_vars if v in keep_vars]]

        node.results["ds_raw"] = ds_raw

        if namespace_overrides:
            node.namespace.update(namespace_overrides)

        call_node_action(node, "analyse_data")
        call_node_action(node, "plot_data")

        action_manager = getattr(node, "_action_manager", None)
        has_update_state = (
            action_manager is not None
            and "update_state" in getattr(action_manager, "actions", {})
        )
        if "fit_results" in node.results and has_update_state:
            call_node_action(node, "update_state")

        artifacts_dir = ARTIFACTS_BASE / (artifacts_subdir or node_name)
        fig_to_save = (
            node.results.get("figures", {}).get("crot_spectroscopy")
            or node.results.get("figure")
            or fig
        )
        if fig_to_save is not None:
            save_analysis_plot(fig_to_save, artifacts_dir, "simulation.png")

        markdown_generator(node, get_parameters_dict(node), artifacts_dir)

        if fig_to_save is not None:
            assert (
                artifacts_dir / "simulation.png"
            ).exists(), "simulation.png not created"
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
        if "tests/analysis/" in str(item.fspath) and not item.get_closest_marker(
            "analysis"
        ):
            item.add_marker(pytest.mark.analysis)
