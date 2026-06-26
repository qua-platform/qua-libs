"""Analysis test for ``06b_PSB_search_opx_fixed_detuning_measure_duration``.

Mirrors the structure of ``test_06a_PSB_search_opx_sweep_detuning_analysis``:
skips QUA program generation / hardware execution entirely, synthesises a
shot-by-shot IQ dataset over a readout-length sweep, then runs the node's
``analyse_data``, ``plot_data``, and ``update_state`` actions.  Figures and a
README are written to ``tests/analysis/artifacts/``.

The node delegates analysis to ``fit_raw_data_pca_gaussian`` (PCA + EM on
|IQ|² per sweep slice).  The synthetic data places the singlet and triplet
blobs at *different radii* from the IQ origin so that the EM can distinguish
them via |IQ|²; blob separation grows with readout length so the optimal
sweep value is at the longest integration time.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from .conftest import (
    ANALYSE_QUBITS,  # noqa: F401 — pulled in for module hygiene
    ARTIFACTS_BASE,
    CALIBRATION_LIBRARY_ROOT,
)

from calibration_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)

NODE_NAME = "06b_PSB_search_opx_fixed_detuning_measure_duration"
PAIR_NAME = "q1_q2"


# ── Synthetic data generation ─────────────────────────────────────────────────


def _simulate_psb_duration_sweep(
    *,
    sweep_values_ns: np.ndarray,
    num_shots: int,
    mu_S: tuple = (0.02e-2, 0.0),
    mu_T_max: tuple = (1.2e-2, 0.25e-2),
    sigma_I: float = 0.12e-2,
    sigma_Q: float = 0.10e-2,
    p_triplet: float = 0.5,
    tau_M: float = 1.0,
    T1: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate shot-by-shot I, Q for a readout-length sweep.

    ``mu_T`` is linearly interpolated from ``mu_S`` toward ``mu_T_max`` as
    ``readout_length`` increases, so blob separation (and fidelity) grows
    monotonically with sweep index.

    The S and T blobs are placed at *different distances* from the IQ origin
    so that ``_vmap_em_two_gaussians(I² + Q²)`` can distinguish them.
    """
    rng = np.random.default_rng(seed=seed)
    t_ref = float(sweep_values_ns.max())
    n_t = len(sweep_values_ns)
    I = np.zeros((num_shots, n_t))
    Q = np.zeros((num_shots, n_t))

    for ti, t_ns in enumerate(sweep_values_ns):
        frac = float(t_ns) / t_ref
        mu_T = (
            mu_S[0] + frac * (mu_T_max[0] - mu_S[0]),
            mu_S[1] + frac * (mu_T_max[1] - mu_S[1]),
        )
        params = SimulationParamsIQ(
            n_samples=num_shots,
            p_triplet=p_triplet,
            mu_S=mu_S,
            mu_T=mu_T,
            sigma_I=sigma_I,
            sigma_Q=sigma_Q,
            rho=0.0,
            tau_M=tau_M,
            T1=T1,
        )
        X, _ = simulate_readout_iq(params, rng=rng, return_labels=False)
        I[:, ti] = X[:, 0]
        Q[:, ti] = X[:, 1]

    return I, Q


def _build_ds_raw(
    pair_names: list[str],
    sweep_values_ns: np.ndarray,
    num_shots: int,
    sweep_name: str = "readout_length",
    seed_base: int = 42,
) -> xr.Dataset:
    I_per_pair, Q_per_pair = [], []
    for i in range(len(pair_names)):
        I, Q = _simulate_psb_duration_sweep(
            sweep_values_ns=sweep_values_ns,
            num_shots=num_shots,
            seed=seed_base + i,
        )
        I_per_pair.append(I)
        Q_per_pair.append(Q)

    I_arr = np.stack(I_per_pair, axis=0)  # (n_pairs, n_runs, n_sweep)
    Q_arr = np.stack(Q_per_pair, axis=0)

    return xr.Dataset(
        {
            "I": (["qubit_pair", "n_runs", sweep_name], I_arr),
            "Q": (["qubit_pair", "n_runs", sweep_name], Q_arr),
        },
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            sweep_name: xr.DataArray(
                sweep_values_ns,
                dims=sweep_name,
                attrs={"long_name": "readout length", "units": "ns"},
            ),
        },
    )


def _add_pair_aliases(machine, alias_names: list[str]) -> list[str]:
    """Register extra qubit-pair aliases backed by the same dot pair."""
    base_pair = machine.qubit_pairs[PAIR_NAME]
    for alias in alias_names:
        machine.qubit_pairs[alias] = type(
            "_AliasPair",
            (),
            {
                "name": alias,
                "quantum_dot_pair": base_pair.quantum_dot_pair,
            },
        )()
    return [PAIR_NAME, *alias_names]


# ── Bespoke analysis runner ───────────────────────────────────────────────────


def _run_06b_analysis(
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
    from .conftest import markdown_generator  # noqa: F401 — for fixture resolution

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
    apply_param_overrides(
        node,
        {"simulate": False, "use_simulated_data": False, **param_overrides},
    )

    # Resolve qubit pairs
    if node.parameters.qubit_pairs not in (None, ""):
        node.namespace["qubit_pairs"] = [
            machine.qubit_pairs[name] for name in node.parameters.qubit_pairs
        ]
    else:
        node.namespace["qubit_pairs"] = list(machine.qubit_pairs.values())

    node.namespace["dot_pairs"] = [
        qp.quantum_dot_pair for qp in node.namespace["qubit_pairs"]
    ]

    try:
        from calibration_utils.common_utils.experiment import get_sensors
        node.namespace["sensors"] = get_sensors(node)
    except Exception:
        pass

    # No hardware patches were applied: nothing to revert
    node.namespace["tracked_resonators"] = []
    node.namespace["tracked_original_detunings"] = {}

    node.results["ds_raw"] = ds_raw

    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    if "fit_results" in node.results:
        call_node_action(node, "update_state")

    # ── Artifact generation ──────────────────────────────────────────────
    artifacts_dir = ARTIFACTS_BASE / artifacts_subdir
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    save = make_save_analysis_plot()
    figs: Dict[str, Any] = node.results.get("figures", {}) or {}
    saved: list[str] = []
    for fname, fig in figs.items():
        if fig is None:
            continue
        save(fig, artifacts_dir, f"{fname}.png")
        saved.append(fname)

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
            "| qubit_pair | optimal_length_ns | F* @ length | V* @ length | F (%) | V | success |",
            "|------------|-------------------|-------------|-------------|-------|---|---------|",
        ]
        for qp_name, r in sorted(fit_results.items()):
            md.append(
                f"| {qp_name} | {r['optimal_sweep_value']:.4g} | "
                f"{r['optimal_sweep_value_fidelity']:.4g} | "
                f"{r['optimal_sweep_value_visibility']:.4g} | "
                f"{r['readout_fidelity']:.1f} | {r['visibility']:.3f} | {r['success']} |"
            )

    md += ["", "## Figures", ""] + [f"![{n}]({n}.png)" for n in saved]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


# ── Test ─────────────────────────────────────────────────────────────────────


@pytest.mark.analysis
def test_06b_psb_measure_duration_analysis(minimal_quam_factory):
    machine = minimal_quam_factory()
    assert PAIR_NAME in machine.qubit_pairs, (
        f"Test factory missing expected pair '{PAIR_NAME}'; "
        f"got {list(machine.qubit_pairs)}"
    )

    sweep_values_ns = np.array([100, 200, 400, 800, 1200, 1600], dtype=float)
    num_shots = 1000
    pair_names = [PAIR_NAME]

    ds_raw = _build_ds_raw(
        pair_names=pair_names,
        sweep_values_ns=sweep_values_ns,
        num_shots=num_shots,
    )

    node = _run_06b_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "num_shots": num_shots,
            "readout_length_min": int(sweep_values_ns[0]),
            "readout_length_max": int(sweep_values_ns[-1]),
            "readout_length_points": len(sweep_values_ns),
            "sweep_name": "readout_length",
            "optimization_metric": "fidelity",
            "labeled_states": False,
        },
        artifacts_subdir=NODE_NAME,
    )

    # ── Structural assertions ─────────────────────────────────────────────
    assert "fit_results" in node.results
    assert set(node.results["fit_results"]) == set(pair_names)

    for qp_name, r in node.results["fit_results"].items():
        assert r["success"], f"Analysis failed for {qp_name}: {r}"
        assert r["sweep_name"] == "readout_length"
        assert len(r["sweep_values"]) == len(sweep_values_ns)
        assert np.isfinite(r["optimal_sweep_value"]), (
            f"optimal_sweep_value not finite for {qp_name}"
        )
        assert sweep_values_ns.min() <= r["optimal_sweep_value"] <= sweep_values_ns.max()
        assert np.isfinite(r["I_threshold"]), f"I_threshold not finite for {qp_name}"
        assert r["readout_fidelity"] > 50.0, (
            f"Fidelity unexpectedly low for {qp_name}: {r['readout_fidelity']:.1f}%"
        )

    # ── Figure assertions ─────────────────────────────────────────────────
    figs = node.results.get("figures", {})
    for expected_key in (
        "fidelity_vs_readout_length",
        "visibility_vs_readout_length",
        "sweep_summary",
        "rotated_iq_density",
    ):
        assert expected_key in figs, f"Expected figure {expected_key!r} not produced"
        assert figs[expected_key] is not None

    # ── Artifact assertions ───────────────────────────────────────────────
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    assert (artifacts_dir / "README.md").exists(), "README.md not written to artifacts"
    assert any(
        (artifacts_dir / f"{k}.png").exists()
        for k in ("fidelity_vs_readout_length", "visibility_vs_readout_length", "sweep_summary")
    ), "No figure PNG written to artifacts"
    assert (artifacts_dir / "rotated_iq_density.png").exists(), (
        "rotated_iq_density.png not written to artifacts"
    )