"""Analysis test for ``06d_PSB_search_opx_fixed_detuning``.

Mirrors the structure of ``test_06a_PSB_search_opx_sweep_detuning_analysis``:
skips QUA program generation / hardware execution entirely, synthesises a
labeled two-arm IQ dataset (I_no_pi/Q_no_pi and I_pi/Q_pi per qubit),
then runs the node's ``analyse_data``, ``plot_data``, and ``update_state``
actions.  Figures and a README are written to ``tests/analysis/artifacts/``.

Runs for both the Barthel and GMM analysis models.
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

NODE_NAME = "06d_PSB_search_opx_fixed_detuning"
QUBIT_NAME = "q1"


# ── Synthetic data generation ─────────────────────────────────────────────────


def _build_ds_raw(
    qubit_names: list[str],
    num_shots: int,
    seed_base: int = 42,
) -> xr.Dataset:
    """Build ds_raw with I_no_pi, Q_no_pi, I_pi, Q_pi per qubit.

    No-pi arm: S (singlet/no-decay) shots — tight Gaussian blob at mu_S.
    Pi arm:    T (triplet/decay) shots — well-separated blob at mu_T.
    Blobs are deliberately far apart so Barthel MCMC and GMM both converge.
    """
    mu_S = (3.0e-2, 1.0e-2)
    mu_T = (1.0e-2, 0.25e-2)
    sigma_I, sigma_Q = 0.12e-2, 0.10e-2

    I_no_pi_list, Q_no_pi_list = [], []
    I_pi_list,    Q_pi_list    = [], []

    for i, _ in enumerate(qubit_names):
        rng_s = np.random.default_rng(seed=seed_base + i)
        rng_t = np.random.default_rng(seed=seed_base + i + 100)

        # no-pi arm → S state (singlet, ground)
        I_s = rng_s.normal(mu_S[0], sigma_I, num_shots)
        Q_s = rng_s.normal(mu_S[1], sigma_Q, num_shots)
        # pi arm → T state (triplet, decay/excited)
        I_t = rng_t.normal(mu_T[0], sigma_I, num_shots)
        Q_t = rng_t.normal(mu_T[1], sigma_Q, num_shots)

        I_no_pi_list.append(I_s)
        Q_no_pi_list.append(Q_s)
        I_pi_list.append(I_t)
        Q_pi_list.append(Q_t)

    I_no_pi = np.stack(I_no_pi_list, axis=0)
    Q_no_pi = np.stack(Q_no_pi_list, axis=0)
    I_pi    = np.stack(I_pi_list,    axis=0)
    Q_pi    = np.stack(Q_pi_list,    axis=0)

    return xr.Dataset(
        {
            "I_no_pi": (["qubit", "n_runs"], I_no_pi),
            "Q_no_pi": (["qubit", "n_runs"], Q_no_pi),
            "I_pi":    (["qubit", "n_runs"], I_pi),
            "Q_pi":    (["qubit", "n_runs"], Q_pi),
        },
        coords={
            "qubit":  qubit_names,
            "n_runs": np.arange(num_shots),
        },
    )


# ── Bespoke analysis runner ───────────────────────────────────────────────────


def _run_06d_analysis(
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
        {"simulate": False, **param_overrides},
    )

    # Resolve qubits from the machine
    from calibration_utils.common_utils.experiment import get_qubits
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Resolve (qubit, dot_pair) tuples via preferred_readout_quantum_dot
    qubit_dot_pairs = []
    for qubit in qubits:
        preferred_dot_id = getattr(qubit, "preferred_readout_quantum_dot", None)
        if preferred_dot_id is not None:
            pair_name = machine.find_quantum_dot_pair(qubit.quantum_dot.id, preferred_dot_id)
            if pair_name is not None:
                qubit_dot_pairs.append((qubit, machine.quantum_dot_pairs[pair_name]))
    node.namespace["qubit_dot_pairs"] = qubit_dot_pairs
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
            "| qubit | I_threshold | iw_angle | F (%) | success |",
            "|-------|-------------|----------|-------|---------|",
        ]
        for qname, r in sorted(fit_results.items()):
            md.append(
                f"| {qname} | {r['I_threshold']:.4g} | "
                f"{r.get('iw_angle', float('nan')):.4g} | "
                f"{r['readout_fidelity']:.1f} | {r['success']} |"
            )

    md += ["", "## Figures", ""] + [f"![{n}]({n}.png)" for n in saved]
    (artifacts_dir / "README.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return node


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.analysis
@pytest.mark.parametrize("analysis_model", ["barthel", "gmm"])
def test_06d_psb_fixed_detuning_analysis(minimal_quam_factory, analysis_model):
    machine = minimal_quam_factory()
    assert QUBIT_NAME in machine.qubits, (
        f"Test factory missing expected qubit '{QUBIT_NAME}'; "
        f"got {list(machine.qubits)}"
    )

    num_shots = 1000
    qubit_names = [QUBIT_NAME]

    ds_raw = _build_ds_raw(
        qubit_names=qubit_names,
        num_shots=num_shots,
    )

    artifacts_subdir = f"{NODE_NAME}_{analysis_model}"
    node = _run_06d_analysis(
        machine=machine,
        ds_raw=ds_raw,
        param_overrides={
            "qubits": qubit_names,
            "num_shots": num_shots,
            "analysis_model": analysis_model,
            "init_state_label": "no_decay",
        },
        artifacts_subdir=artifacts_subdir,
    )

    # ── Structural assertions ─────────────────────────────────────────────
    assert "fit_results" in node.results
    assert set(node.results["fit_results"]) == set(qubit_names)

    # ── Figure assertions ─────────────────────────────────────────────────
    figs = node.results.get("figures", {})
    assert "iq_blobs" in figs, "Expected figure 'iq_blobs' not produced"
    assert figs["iq_blobs"] is not None
    assert "histogram" in figs, "Expected figure 'histogram' not produced"
    assert figs["histogram"] is not None

    # ── Artifact assertions ───────────────────────────────────────────────
    artifacts_dir = ARTIFACTS_BASE / artifacts_subdir
    assert (artifacts_dir / "README.md").exists(), "README.md not written to artifacts"