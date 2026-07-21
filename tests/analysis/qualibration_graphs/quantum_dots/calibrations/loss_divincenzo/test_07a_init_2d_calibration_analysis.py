"""Analysis test for ``07a_init_2d_calibration``.

Synthesises 2D averaged state-assignment data for one or more qubit pairs as a
function of initialisation ramp duration and wait duration (between init and
measure), then runs the node's ``analyse_data``, ``plot_data``, and
``update_state`` actions.

The synthetic signal is a 2D bowl: the average state has a minimum at a
specific (ramp_duration, wait_duration) pair.  Fast ramps and very short or
very long waits produce mixed states.
"""

from __future__ import annotations

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

NODE_NAME = "07a_init_2d_calibration"
PAIR_NAME = "q1_q2"


# ── Synthetic data generation ──────────────────────────────────────────────


def _synthetic_avg_state_2d(
    ramp_durations: np.ndarray,
    wait_durations: np.ndarray,
    optimal_ramp_ns: float,
    optimal_wait_ns: float,
    base: float = 0.05,
    curvature_ramp: float = 1.5e-6,
    curvature_wait: float = 1.5e-6,
    noise_std: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Generate a 2D bowl-shaped average state with a minimum at the given optimum.

    Returns shape ``(n_ramp, n_wait)``.
    """
    rng = np.random.default_rng(seed)
    ramp_2d, wait_2d = np.meshgrid(ramp_durations, wait_durations, indexing="ij")
    signal = (
        base
        + curvature_ramp * (ramp_2d - optimal_ramp_ns) ** 2
        + curvature_wait * (wait_2d - optimal_wait_ns) ** 2
    )
    signal += rng.normal(0, noise_std, size=signal.shape)
    return np.clip(signal, 0.0, 1.0)


def _synthetic_iq_2d(
    ramp_durations: np.ndarray,
    wait_durations: np.ndarray,
    i_offset: float = 0.01,
    amplitude: float = 0.005,
    noise_std: float = 0.001,
    seed: int = 0,
) -> np.ndarray:
    """Generate synthetic averaged I signal on a 2D grid.

    Returns shape ``(n_ramp, n_wait)``.
    """
    rng = np.random.default_rng(seed)
    t_ramp = (ramp_durations - ramp_durations[0]) / max(
        ramp_durations[-1] - ramp_durations[0], 1
    )
    t_wait = (wait_durations - wait_durations[0]) / max(
        wait_durations[-1] - wait_durations[0], 1
    )
    ramp_2d, wait_2d = np.meshgrid(t_ramp, t_wait, indexing="ij")
    I = (
        i_offset
        + amplitude * np.cos(2 * np.pi * ramp_2d) * np.cos(2 * np.pi * wait_2d)
        + rng.normal(0, noise_std, size=ramp_2d.shape)
    )
    return I


def _build_ds_raw_2d(
    pair_names: list[str],
    ramp_durations: np.ndarray,
    wait_durations: np.ndarray,
    optimal_ramp_ns: float | list[float],
    optimal_wait_ns: float | list[float],
    seed_base: int = 42,
) -> xr.Dataset:
    """Build a synthetic 2D ``ds_raw`` matching the node's stream-processed output."""
    if isinstance(optimal_ramp_ns, (int, float)):
        optimal_ramp_ns = [optimal_ramp_ns] * len(pair_names)
    if isinstance(optimal_wait_ns, (int, float)):
        optimal_wait_ns = [optimal_wait_ns] * len(pair_names)

    data_vars: Dict[str, Any] = {}
    for i, pname in enumerate(pair_names):
        avg_state = _synthetic_avg_state_2d(
            ramp_durations,
            wait_durations,
            optimal_ramp_ns=optimal_ramp_ns[i],
            optimal_wait_ns=optimal_wait_ns[i],
            seed=seed_base + i,
        )
        I = _synthetic_iq_2d(
            ramp_durations, wait_durations, seed=seed_base + i + 100
        )
        data_vars[f"state_{pname}"] = xr.DataArray(
            avg_state, dims=["ramp_duration", "wait_duration"]
        )
        data_vars[f"I_{pname}"] = xr.DataArray(
            I, dims=["ramp_duration", "wait_duration"]
        )

    return xr.Dataset(
        data_vars,
        coords={
            "ramp_duration": xr.DataArray(
                ramp_durations,
                dims="ramp_duration",
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
            "wait_duration": xr.DataArray(
                wait_durations,
                dims="wait_duration",
                attrs={"long_name": "wait duration", "units": "ns"},
            ),
        },
    )


# ── Bespoke runner ─────────────────────────────────────────────────────


def _run_07a_analysis(
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
            "| qubit_pair | optimal_ramp (ns) | optimal_wait (ns) | optimal_avg_state | find_minimum | success |",
            "|------------|-------------------|-------------------|-------------------|--------------|---------|",
        ]
        for qp_name, r in sorted(fit_results.items()):
            md.append(
                f"| {qp_name} | {r['optimal_ramp_duration']} | "
                f"{r['optimal_wait_duration']} | "
                f"{r['optimal_avg_state']:.4f} | {r['find_minimum']} | {r['success']} |"
            )

    md += ["", "## Analysis Output", "", "![Analysis](simulation.png)", ""]
    if saved_figs:
        md += ["", "## Figures", ""] + [f"![{n}]({n}.png)" for n in saved_figs]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_07a_init_2d_find_minimum(minimal_quam_factory):
    """Find the minimum average state in a 2D (ramp, wait) sweep."""
    machine = minimal_quam_factory()
    assert PAIR_NAME in machine.qubit_pairs, (
        f"Test factory missing expected pair '{PAIR_NAME}'; "
        f"got {list(machine.qubit_pairs)}"
    )
    pair_names = [PAIR_NAME]

    ramp_min, ramp_max, ramp_step = 16, 2000, 40
    wait_min, wait_max, wait_step = 16, 2000, 40
    ramp_durations = np.arange(ramp_min, ramp_max, ramp_step)
    wait_durations = np.arange(wait_min, wait_max, wait_step)
    optimal_ramp_ns = 500
    optimal_wait_ns = 800

    ds_raw = _build_ds_raw_2d(
        pair_names=pair_names,
        ramp_durations=ramp_durations,
        wait_durations=wait_durations,
        optimal_ramp_ns=optimal_ramp_ns,
        optimal_wait_ns=optimal_wait_ns,
    )

    node = _run_07a_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "ramp_duration_min": ramp_min,
            "ramp_duration_max": ramp_max,
            "ramp_duration_step": ramp_step,
            "wait_duration_min": wait_min,
            "wait_duration_max": wait_max,
            "wait_duration_step": wait_step,
            "find_minimum": True,
            "num_shots": 100,
        },
        artifacts_subdir=NODE_NAME,
    )

    assert "fit_results" in node.results
    fit = node.results["fit_results"][PAIR_NAME]
    assert fit["success"]
    assert fit["find_minimum"] is True

    found_ramp = fit["optimal_ramp_duration"]
    found_wait = fit["optimal_wait_duration"]
    assert abs(found_ramp - optimal_ramp_ns) <= ramp_step * 3, (
        f"Optimal ramp should be near {optimal_ramp_ns} ns, got {found_ramp} ns"
    )
    assert abs(found_wait - optimal_wait_ns) <= wait_step * 3, (
        f"Optimal wait should be near {optimal_wait_ns} ns, got {found_wait} ns"
    )
    assert 0.0 <= fit["optimal_avg_state"] <= 0.3, (
        f"Minimum avg state should be low, got {fit['optimal_avg_state']:.4f}"
    )

    assert "figure" in node.results
    fig = node.results["figure"]
    # 2x2 grid per pair: 4 main axes + 4 colorbars = 8 total axes
    all_axes = fig.axes
    assert len(all_axes) >= 4 * len(pair_names), (
        f"Expected at least 4 subplot axes per pair, got {len(all_axes)}"
    )

    figs = node.results.get("figures", {})
    assert "summary_2d" in figs, "Expected figure 'summary_2d' not produced"
    assert figs["summary_2d"] is not None

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    assert (artifacts_dir / "README.md").exists()
    assert (artifacts_dir / "simulation.png").exists()
    assert (artifacts_dir / "summary_2d.png").exists(), (
        "summary_2d.png not written to artifacts"
    )


@pytest.mark.analysis
def test_07a_init_2d_find_maximum(minimal_quam_factory):
    """Find the maximum average state (find_minimum=False) in a 2D sweep."""
    machine = minimal_quam_factory()
    pair_names = [PAIR_NAME]

    ramp_min, ramp_max, ramp_step = 16, 2000, 40
    wait_min, wait_max, wait_step = 16, 2000, 40
    ramp_durations = np.arange(ramp_min, ramp_max, ramp_step)
    wait_durations = np.arange(wait_min, wait_max, wait_step)

    rng = np.random.default_rng(99)
    optimal_ramp_ns = 800
    optimal_wait_ns = 600
    ramp_2d, wait_2d = np.meshgrid(ramp_durations, wait_durations, indexing="ij")
    signal = (
        0.9
        - 1.5e-6 * (ramp_2d - optimal_ramp_ns) ** 2
        - 1.5e-6 * (wait_2d - optimal_wait_ns) ** 2
    )
    signal += rng.normal(0, 0.02, size=signal.shape)
    signal = np.clip(signal, 0.0, 1.0)

    ds_raw = xr.Dataset(
        {f"state_{PAIR_NAME}": xr.DataArray(signal, dims=["ramp_duration", "wait_duration"])},
        coords={
            "ramp_duration": xr.DataArray(
                ramp_durations,
                dims="ramp_duration",
                attrs={"long_name": "ramp duration", "units": "ns"},
            ),
            "wait_duration": xr.DataArray(
                wait_durations,
                dims="wait_duration",
                attrs={"long_name": "wait duration", "units": "ns"},
            ),
        },
    )

    node = _run_07a_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "ramp_duration_min": ramp_min,
            "ramp_duration_max": ramp_max,
            "ramp_duration_step": ramp_step,
            "wait_duration_min": wait_min,
            "wait_duration_max": wait_max,
            "wait_duration_step": wait_step,
            "find_minimum": False,
            "num_shots": 100,
        },
        artifacts_subdir=f"{NODE_NAME}_find_max",
    )

    fit = node.results["fit_results"][PAIR_NAME]
    assert fit["success"]
    assert fit["find_minimum"] is False

    found_ramp = fit["optimal_ramp_duration"]
    found_wait = fit["optimal_wait_duration"]
    assert abs(found_ramp - optimal_ramp_ns) <= ramp_step * 3, (
        f"Optimal ramp should be near {optimal_ramp_ns} ns, got {found_ramp} ns"
    )
    assert abs(found_wait - optimal_wait_ns) <= wait_step * 3, (
        f"Optimal wait should be near {optimal_wait_ns} ns, got {found_wait} ns"
    )
    assert fit["optimal_avg_state"] >= 0.7, (
        f"Maximum avg state should be high, got {fit['optimal_avg_state']:.4f}"
    )
