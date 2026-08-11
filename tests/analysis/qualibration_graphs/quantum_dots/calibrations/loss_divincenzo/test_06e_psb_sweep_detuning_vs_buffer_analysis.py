"""Analysis test for ``06e_PSB_search_opx_sweep_detuning_vs_buffer``.

Skips QUA program generation / hardware execution entirely, synthesises a
shot-by-shot IQ dataset over a 2D detuning/buffer sweep, then runs the node's
``process_raw_data``, ``analyse_data``, ``plot_data``, and ``update_state``
actions. Figures and a README are written to ``tests/analysis/artifacts/``.

The synthetic data makes the singlet/triplet separation peak near a chosen
detuning and at the longest buffer duration, so the exploratory PCA metric has
a clear optimum to recover and persist back into the QUAM state.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from calibration_utils.iq_utils.iq_blobs.readout_barthel.simulate import (
    SimulationParamsIQ,
    simulate_readout_iq,
)

from .conftest import (
    ANALYSE_QUBITS,  # noqa: F401 -- pulled in for module hygiene
    ARTIFACTS_BASE,
    CALIBRATION_LIBRARY_ROOT,
)

NODE_NAME = "06e_PSB_search_opx_sweep_detuning_vs_buffer"
PAIR_NAME = "q1_q2"


def _simulate_psb_2d_sweep(
    *,
    detuning_values: np.ndarray,
    buffer_values_ns: np.ndarray,
    num_shots: int,
    optimal_detuning: float,
    width: float = 0.02,
    mu_S: tuple = (0.0, 0.0),
    mu_T_max: tuple = (1.5e-2, 0.375e-2),
    sigma_I: float = 0.18e-2,
    sigma_Q: float = 0.15e-2,
    p_triplet: float = 0.5,
    tau_M: float = 1.0,
    T1: float = 2.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate shot-by-shot I/Q for a detuning-vs-buffer sweep."""
    rng = np.random.default_rng(seed=seed)
    n_detuning = len(detuning_values)
    n_buffer = len(buffer_values_ns)
    I = np.zeros((num_shots, n_detuning, n_buffer))
    Q = np.zeros((num_shots, n_detuning, n_buffer))

    buffer_scale = float(buffer_values_ns.max())
    for di, detuning in enumerate(detuning_values):
        detuning_weight = float(
            np.exp(-((detuning - optimal_detuning) ** 2) / (2.0 * width**2))
        )

        for bi, buffer_ns in enumerate(buffer_values_ns):
            buffer_weight = float(buffer_ns) / buffer_scale
            separation = detuning_weight * buffer_weight
            mu_T = (
                mu_S[0] + (mu_T_max[0] - mu_S[0]) * separation,
                mu_S[1] + (mu_T_max[1] - mu_S[1]) * separation,
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
            I[:, di, bi] = X[:, 0]
            Q[:, di, bi] = X[:, 1]

    I += detuning_values[None, :, None] * 0.1 + 0.3
    Q += detuning_values[None, :, None] * 0.15 + 0.2
    return I, Q


def _build_ds_raw(
    pair_names: list[str],
    detuning_values: np.ndarray,
    buffer_values_ns: np.ndarray,
    num_shots: int,
    optimal_detuning: float,
    seed_base: int = 42,
) -> xr.Dataset:
    I_per_pair = []
    Q_per_pair = []
    for i in range(len(pair_names)):
        I, Q = _simulate_psb_2d_sweep(
            detuning_values=detuning_values,
            buffer_values_ns=buffer_values_ns,
            num_shots=num_shots,
            optimal_detuning=optimal_detuning,
            seed=seed_base + i,
        )
        I_per_pair.append(I)
        Q_per_pair.append(Q)

    I_arr = np.stack(I_per_pair, axis=0)
    Q_arr = np.stack(Q_per_pair, axis=0)

    return xr.Dataset(
        {
            "I": (["qubit_pair", "n_runs", "detuning", "buffer_duration"], I_arr),
            "Q": (["qubit_pair", "n_runs", "detuning", "buffer_duration"], Q_arr),
        },
        coords={
            "qubit_pair": pair_names,
            "n_runs": np.arange(num_shots),
            "detuning": xr.DataArray(
                detuning_values,
                dims="detuning",
                attrs={"long_name": "detuning", "units": "V"},
            ),
            "buffer_duration": xr.DataArray(
                buffer_values_ns,
                dims="buffer_duration",
                attrs={"long_name": "buffer duration", "units": "ns"},
            ),
        },
    )


def _run_06e_analysis(
    *,
    machine,
    ds_raw: xr.Dataset,
    param_overrides: Dict[str, Any],
    artifacts_subdir: str,
) -> Any:
    from tests.shared_fixtures import (
        apply_param_overrides,
        call_node_action,
        ensure_quam_config_stub,
        get_parameters_dict,
        load_library_node,
        make_save_analysis_plot,
        patch_action_manager_register_only,
        reimport_node_to_register_actions,
    )
    from .conftest import markdown_generator  # noqa: F401 -- for fixture resolution

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

    if node.parameters.qubit_pairs not in (None, ""):
        node.namespace["qubit_pairs"] = [
            machine.qubit_pairs[name] for name in node.parameters.qubit_pairs
        ]
    else:
        node.namespace["qubit_pairs"] = list(machine.qubit_pairs.values())

    node.results["ds_raw"] = ds_raw

    call_node_action(node, "process_raw_data")
    call_node_action(node, "analyse_data")
    call_node_action(node, "plot_data")
    if "fit_results" in node.results:
        call_node_action(node, "update_state")

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
            "| qubit_pair | optimal_detuning | optimal_buffer_duration | metric_name | max_metric_value | success |",
            "|------------|------------------|-------------------------|-------------|------------------|---------|",
        ]
        for qp_name, r in sorted(fit_results.items()):
            md.append(
                f"| {qp_name} | {r['optimal_detuning']:.4g} | "
                f"{r['optimal_buffer_duration']} | {r['metric_name']} | "
                f"{r['max_metric_value']:.4g} | {r['success']} |"
            )

    md += ["", "## Figures", ""] + [f"![{n}]({n}.png)" for n in saved]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


@pytest.mark.analysis
def test_06e_psb_sweep_detuning_vs_buffer_analysis(minimal_quam_factory):
    machine = minimal_quam_factory()
    assert PAIR_NAME in machine.qubit_pairs, (
        f"Test factory missing expected pair '{PAIR_NAME}'; "
        f"got {list(machine.qubit_pairs)}"
    )

    detuning_values = np.linspace(-0.05, 0.05, 9)
    buffer_values_ns = np.arange(16, 96, 16, dtype=float)
    optimal_detuning = 0.025
    num_shots = 1500
    pair_names = [PAIR_NAME]

    dot_pair = machine.qubit_pairs[PAIR_NAME].quantum_dot_pair
    point_name = dot_pair._create_point_name("measure")
    point = dot_pair.voltage_sequence.gate_set.get_macros()[point_name]
    measure_macro = dot_pair.macros.get("measure")

    ds_raw = _build_ds_raw(
        pair_names=pair_names,
        detuning_values=detuning_values,
        buffer_values_ns=buffer_values_ns,
        num_shots=num_shots,
        optimal_detuning=optimal_detuning,
    )

    node = _run_06e_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubit_pairs": pair_names,
            "num_shots": num_shots,
            "detuning_min": float(detuning_values[0]),
            "detuning_max": float(detuning_values[-1]),
            "detuning_points": len(detuning_values),
            "buffer_duration_min": int(buffer_values_ns[0]),
            "buffer_duration_max": int(buffer_values_ns[-1] + 16),
            "buffer_duration_step": 16,
            "pca_metric": "pc1_std",
        },
        artifacts_subdir=NODE_NAME,
    )

    assert "fit_results" in node.results
    assert set(node.results["fit_results"]) == set(pair_names)

    fit = node.results["fit_results"][PAIR_NAME]
    assert fit["success"], f"Analysis failed for {PAIR_NAME}: {fit}"
    assert fit["metric_name"] == "pc1_std"
    assert np.isfinite(fit["optimal_detuning"])
    assert fit["optimal_buffer_duration"] in buffer_values_ns
    assert fit["optimal_buffer_duration"] == int(buffer_values_ns[-1])

    nearest_gap = float(np.min(np.abs(detuning_values - optimal_detuning)))
    assert abs(float(fit["optimal_detuning"]) - optimal_detuning) <= nearest_gap + 1e-9

    figs = node.results.get("figures", {})
    assert "detuning_vs_buffer_pca_map" in figs
    assert figs["detuning_vs_buffer_pca_map"] is not None

    assert point.voltages[dot_pair.name] == pytest.approx(fit["optimal_detuning"])
    assert measure_macro is not None
    assert measure_macro.buffer_duration == int(fit["optimal_buffer_duration"])

    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    assert (artifacts_dir / "README.md").exists(), "README.md not written to artifacts"
    assert (artifacts_dir / "detuning_vs_buffer_pca_map.png").exists(), (
        "detuning_vs_buffer_pca_map.png not written to artifacts"
    )
