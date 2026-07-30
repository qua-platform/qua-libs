"""Analysis test for ``07_init_ramp_rate_calibration``.

Synthesises averaged state-assignment data for one or more qubit pairs as a
function of initialisation ramp duration, then runs the node's
``analyse_data``, ``plot_data``, and ``update_state`` actions.

The synthetic signal is a U-shaped curve: fast ramps produce mixed states
(avg ~ 0.5) because the dots cannot follow, while very slow ramps suffer from
decoherence.  The optimal ramp_duration sits at the valley of the curve.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from .conftest import (
    ARTIFACTS_BASE,
    CALIBRATION_LIBRARY_ROOT,
    QUBIT_PAIR_NAMES,
)

NODE_NAME = "07_init_ramp_rate_calibration"
PAIR_NAME = "q1_q2"


# ── Synthetic data generation ──────────────────────────────────────────────


def _synthetic_avg_state(
    ramp_durations: np.ndarray,
    optimal_ramp_ns: float,
    base: float = 0.05,
    curvature: float = 1.5e-6,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate a U-shaped average state curve with a minimum at *optimal_ramp_ns*.

    The model: avg_state = base + curvature * (t - t_opt)^2 + noise, clipped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    signal = base + curvature * (ramp_durations - optimal_ramp_ns) ** 2
    signal += rng.normal(0, noise_std, size=signal.shape)
    return np.clip(signal, 0.0, 1.0)


def _synthetic_iq(
    ramp_durations: np.ndarray,
    num_shots: int = 100,
    i_offset: float = 0.01,
    q_offset: float = -0.005,
    amplitude: float = 0.005,
    noise_std: float = 0.001,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic I (averaged) and per-shot Q signals vs ramp duration.

    I is 1-D (n_ramp_durations,); Q is 2-D (n_shots, n_ramp_durations) matching
    the hardware stream-processing output.
    """
    rng = np.random.default_rng(seed)
    t = (ramp_durations - ramp_durations[0]) / (ramp_durations[-1] - ramp_durations[0])
    I = i_offset + amplitude * np.cos(2 * np.pi * t) + rng.normal(0, noise_std, size=t.shape)
    # Per-shot Q: mean shifts smoothly with ramp duration; shot-to-shot noise models distribution
    q_mean = q_offset + amplitude * np.sin(2 * np.pi * t)  # (n_ramp_durations,)
    Q = q_mean[np.newaxis, :] + rng.normal(0, noise_std * 5, size=(num_shots, len(ramp_durations)))
    return I, Q


def _build_ds_raw(
    pair_names: list[str],
    ramp_durations: np.ndarray,
    optimal_ramp_ns: float | list[float],
    num_shots: int = 100,
    seed_base: int = 42,
) -> xr.Dataset:
    """Build a synthetic ``ds_raw`` matching the node's stream-processed output."""
    if isinstance(optimal_ramp_ns, (int, float)):
        optimal_ramp_ns = [optimal_ramp_ns] * len(pair_names)

    data_vars: Dict[str, Any] = {}
    for i, pname in enumerate(pair_names):
        avg_state = _synthetic_avg_state(
            ramp_durations,
            optimal_ramp_ns=optimal_ramp_ns[i],
            seed=seed_base + i,
        )
        I, Q = _synthetic_iq(ramp_durations, num_shots=num_shots, seed=seed_base + i + 100)
        data_vars[f"state_{pname}"] = xr.DataArray(avg_state, dims=["ramp_duration"])
        data_vars[f"I_{pname}"] = xr.DataArray(I, dims=["ramp_duration"])
        data_vars[f"Q_{pname}"] = xr.DataArray(Q, dims=["n_shots", "ramp_duration"])

    return xr.Dataset(
        data_vars,
        coords={
            "ramp_duration": xr.DataArray(
                ramp_durations,
                dims="ramp_duration",
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
        },
    )


# ── Bespoke runner (analysis_runner assumes a qubit-list node) ────────────


def _run_07_analysis(
    *,
    machine,
    ds_raw: xr.Dataset,
    param_overrides: Dict[str, Any],
    artifacts_subdir: str,
) -> Any:
    from shared_fixtures import (
        apply_param_overrides,
        call_node_action,
        ensure_quam_config_stub,
        get_parameters_dict,
        load_library_node,
        make_save_analysis_plot,
        patch_action_manager_register_only,
        reimport_node_to_register_actions,
    )

    ensure_quam_config_stub(machine)
    from quam_config import Quam

    with (
        patch.object(Quam, "load", return_value=machine),
        patch_action_manager_register_only(),
    ):
        node = reimport_node_to_register_actions(NODE_NAME, CALIBRATION_LIBRARY_ROOT)
        if node is None:
            node = load_library_node(NODE_NAME, CALIBRATION_LIBRARY_ROOT)

    node.machine = machine
    apply_param_overrides(node, {"simulate": False, **param_overrides})

    if node.parameters.qubit_pairs not in (None, ""):
        node.namespace["qubit_pairs"] = [
            machine.qubit_pairs[name] for name in node.parameters.qubit_pairs
        ]
    else:
        node.namespace["qubit_pairs"] = list(machine.qubit_pairs.values())

    node.results["ds_raw"] = ds_raw

    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    if "fit_results" in node.results:
        call_node_action(node, "update_state")

    artifacts_dir = ARTIFACTS_BASE / artifacts_subdir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save = make_save_analysis_plot()
    fig = node.results.get("figure")
    if fig is not None:
        save(fig, artifacts_dir, "simulation.png")

    figs = node.results.get("figures", {}) or {}
    saved_figs: list[str] = []
    for fname, fig_item in figs.items():
        if fig_item is None:
            continue
        save(fig_item, artifacts_dir, f"{fname}.png")
        saved_figs.append(fname)

    md = [
        f"# {NODE_NAME}",
        "",
        "## Description",
        "",
        str(getattr(node, "description", "") or "").strip(),
        "",
        "## Parameters",
        "",
        "| Parameter | Value |",
        "|-----------|-------|",
    ]
    for k, v in sorted(get_parameters_dict(node).items()):
        md.append(f"| `{k}` | `{v}` |")

    fit_results = node.results.get("fit_results", {})
    if fit_results:
        md += [
            "",
            "## Fit Results",
            "",
            "| qubit_pair | optimal_ramp_duration (ns) | optimal_avg_state | find_minimum | success |",
            "|------------|---------------------------|-------------------|--------------|---------|",
        ]
        for qp_name, r in sorted(fit_results.items()):
            md.append(
                f"| {qp_name} | {r['optimal_ramp_duration']} | "
                f"{r['optimal_avg_state']:.4f} | {r['find_minimum']} | {r['success']} |"
            )

    md += ["", "## Analysis Output", "", "![Analysis](simulation.png)", ""]
    if saved_figs:
        md += ["", "## Figures", ""] + [f"![{n}]({n}.png)" for n in saved_figs]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_07_init_ramp_rate_find_minimum(minimal_quam_factory):
    """Find the minimum average state (default behaviour)."""
    machine = minimal_quam_factory()
    assert PAIR_NAME in machine.qubit_pairs, (
        f"Test factory missing expected pair '{PAIR_NAME}'; "
        f"got {list(machine.qubit_pairs)}"
    )
    pair_names = [PAIR_NAME]

    ramp_min, ramp_max, ramp_step = 16, 2000, 20
    ramp_durations = np.arange(ramp_min, ramp_max, ramp_step)
    optimal_ramp_ns = 500

    num_shots = 100
    ds_raw = _build_ds_raw(
        pair_names=pair_names,
        ramp_durations=ramp_durations,
        optimal_ramp_ns=optimal_ramp_ns,
        num_shots=num_shots,
    )

    node = _run_07_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "ramp_duration_min": ramp_min,
            "ramp_duration_max": ramp_max,
            "ramp_duration_step": ramp_step,
            "find_minimum": True,
            "num_shots": num_shots,
        },
        artifacts_subdir=NODE_NAME,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"][PAIR_NAME]
    assert fit["success"]
    assert fit["find_minimum"] is True

    found_ramp = fit["optimal_ramp_duration"]
    assert abs(found_ramp - optimal_ramp_ns) <= ramp_step * 3, (
        f"Optimal ramp should be near {optimal_ramp_ns} ns, got {found_ramp} ns"
    )
    assert 0.0 <= fit["optimal_avg_state"] <= 0.3, (
        f"Minimum avg state should be low, got {fit['optimal_avg_state']:.4f}"
    )

    assert "figure" in node.results
    fig = node.results["figure"]
    visible_axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible_axes) == len(pair_names)
    assert visible_axes[0].get_xlabel() == "Ramp duration (ns)"
    assert visible_axes[0].get_ylabel() == "Average state assignment"

    figs = node.results.get("figures", {})
    assert "iq_vs_ramp_duration" in figs, "Expected figure 'iq_vs_ramp_duration' not produced"
    assert figs["iq_vs_ramp_duration"] is not None
    assert "q_density_vs_ramp_duration" in figs, "Expected figure 'q_density_vs_ramp_duration' not produced"
    assert figs["q_density_vs_ramp_duration"] is not None

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    assert (artifacts_dir / "README.md").exists()
    assert (artifacts_dir / "simulation.png").exists()
    assert (artifacts_dir / "iq_vs_ramp_duration.png").exists(), (
        "iq_vs_ramp_duration.png not written to artifacts"
    )


@pytest.mark.analysis
def test_07_init_ramp_rate_find_maximum(minimal_quam_factory):
    """Find the maximum average state (find_minimum=False)."""
    machine = minimal_quam_factory()
    pair_names = [PAIR_NAME]

    ramp_min, ramp_max, ramp_step = 16, 2000, 20
    ramp_durations = np.arange(ramp_min, ramp_max, ramp_step)

    rng = np.random.default_rng(99)
    optimal_ramp_ns = 800
    signal = 0.9 - 1.5e-6 * (ramp_durations - optimal_ramp_ns) ** 2
    signal += rng.normal(0, 0.02, size=signal.shape)
    signal = np.clip(signal, 0.0, 1.0)

    ds_raw = xr.Dataset(
        {f"state_{PAIR_NAME}": xr.DataArray(signal, dims=["ramp_duration"])},
        coords={
            "ramp_duration": xr.DataArray(
                ramp_durations,
                dims="ramp_duration",
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
        },
    )

    node = _run_07_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "ramp_duration_min": ramp_min,
            "ramp_duration_max": ramp_max,
            "ramp_duration_step": ramp_step,
            "find_minimum": False,
            "num_shots": 100,
        },
        artifacts_subdir=f"{NODE_NAME}_find_max",
    )

    fit = node.results["fit_results"][PAIR_NAME]
    assert fit["success"]
    assert fit["find_minimum"] is False

    found_ramp = fit["optimal_ramp_duration"]
    assert abs(found_ramp - optimal_ramp_ns) <= ramp_step * 3, (
        f"Optimal ramp should be near {optimal_ramp_ns} ns, got {found_ramp} ns"
    )
    assert fit["optimal_avg_state"] >= 0.7, (
        f"Maximum avg state should be high, got {fit['optimal_avg_state']:.4f}"
    )
