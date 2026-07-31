"""Analysis test for 21_xeb (Cross-Entropy Benchmarking).

Two virtual qubits with different depolarizing noise rates are simulated
numerically.  Random gate sequences are generated, ideal probabilities
computed via unitary multiplication, and depolarizing noise applied to
create synthetic measured data.

No ``virtual_qpu`` / dynamiqs dependency is needed: XEB data generation
is purely mathematical.  We compute ideal probabilities via the XEB gate
unitaries (NumPy), apply a depth-dependent depolarizing channel
``p_meas = r^d · p_ideal + (1 − r^d)/dim``, and add shot noise via
multinomial sampling.  This produces datasets identical in structure to
what the OPX stream processing would output.

The test verifies that:
  1. The analysis pipeline (analyse_data, plot_data, update_state) runs
     without error on the synthetic dataset.
  2. Both qubits produce a valid exponential decay fit.
  3. The qubit with lower noise (higher layer fidelity) has a higher
     fitted ``r`` than the qubit with more noise.
  4. All expected figures are generated and saved to the artifacts directory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from calibration_utils.xeb import (
    NUM_XEB_GATES,
    Parameters,
    calc_ideal_probs_1q,
)

from .conftest import ARTIFACTS_BASE, create_ld_quam as create_minimal_quam

# =============================================================================
# Constants
# =============================================================================

NODE_NAME = "21_xeb"
N_SEQUENCES = 30
N_SHOTS = 500
DEPTH_MIN = 5
DEPTH_MAX = 55
DEPTH_STEP = 5
RNG_SEED = 42

R_HIGH = 0.998  # qubit A: high layer fidelity (low noise)
R_LOW = 0.95  # qubit B: low layer fidelity (high noise)


# =============================================================================
# Synthetic data generation
# =============================================================================


def _generate_gate_indices(
    n_sequences: int,
    max_depth: int,
    num_gates: int,
    seed: int,
) -> np.ndarray:
    """Generate random gate indices with no consecutive repeats.

    Returns ndarray of shape (n_sequences, max_depth) with values in [0, num_gates).
    """
    rng = np.random.default_rng(seed)
    indices = np.zeros((n_sequences, max_depth), dtype=int)

    for s in range(n_sequences):
        indices[s, 0] = rng.integers(0, num_gates)
        for d in range(1, max_depth):
            g = rng.integers(0, num_gates)
            while g == indices[s, d - 1]:
                g = rng.integers(0, num_gates)
            indices[s, d] = g

    return indices


def _apply_depolarizing_noise(
    ideal_probs: np.ndarray,
    depths: np.ndarray,
    layer_fidelity: float,
    n_shots: int,
    dim: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply a depolarizing channel and shot noise to ideal probabilities.

    For each (sequence, depth), the measured distribution is:
        p_measured = r^depth * p_ideal + (1 - r^depth) / dim
    then shot-sampled via multinomial and divided by n_shots.

    Returns ndarray of shape (n_sequences, n_depths) — averaged state
    (probability of |1⟩), matching the node's 1Q stream processing.
    """
    n_sequences, n_depths, _ = ideal_probs.shape
    state_avg = np.zeros((n_sequences, n_depths))

    for s in range(n_sequences):
        for d_idx in range(n_depths):
            r_d = layer_fidelity ** depths[d_idx]
            p_noisy = r_d * ideal_probs[s, d_idx] + (1 - r_d) / dim
            p_noisy = np.clip(p_noisy, 0, 1)
            p_noisy /= p_noisy.sum()
            counts = rng.multinomial(n_shots, p_noisy)
            state_avg[s, d_idx] = counts[1] / n_shots

    return state_avg


def _generate_xeb_dataset(
    gate_indices: np.ndarray,
    depths: np.ndarray,
    layer_fidelity: float,
    n_shots: int,
    gate_set: str,
    qubit_name: str,
    q_idx: int,
    seed: int,
) -> xr.Dataset:
    """Generate a synthetic 1Q XEB dataset for one qubit.

    Returns a Dataset containing gate_indices_{q_idx} and state_{qubit_name}.
    """
    rng = np.random.default_rng(seed)
    ideal_probs = calc_ideal_probs_1q(gate_indices, depths, gate_set)
    state_avg = _apply_depolarizing_noise(
        ideal_probs, depths, layer_fidelity, n_shots, dim=2, rng=rng
    )
    max_depth = gate_indices.shape[1]

    return xr.Dataset(
        {
            f"gate_indices_{q_idx}": xr.DataArray(
                gate_indices,
                dims=["sequence", "gate_depth"],
                coords={
                    "sequence": np.arange(gate_indices.shape[0]),
                    "gate_depth": np.arange(max_depth),
                },
            ),
            f"state_{qubit_name}": xr.DataArray(
                state_avg,
                dims=["sequence", "depth"],
                coords={
                    "sequence": np.arange(state_avg.shape[0]),
                    "depth": depths,
                },
            ),
        }
    )


# =============================================================================
# Test
# =============================================================================


def _save_xeb_figures(node, artifacts_dir: Path) -> list[str]:
    """Save all XEB figures from node.results to the artifacts directory."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, val in node.results.items():
        if key.startswith("figure") and hasattr(val, "savefig"):
            safe_name = key.replace("/", "_").replace(" ", "_")
            fname = f"{safe_name}.png"
            val.savefig(artifacts_dir / fname, dpi=200)
            plt.close(val)
            saved.append(fname)
    return saved


@pytest.mark.analysis
def test_21_xeb_analysis(analysis_runner):
    """1Q XEB with synthetic depolarizing-noise data.

    Two qubits are given different layer fidelities (R_HIGH vs R_LOW).
    The analysis pipeline must recover that qubit A (high fidelity) has
    a higher fitted layer fidelity ``r`` than qubit B (low fidelity).
    """
    machine = create_minimal_quam()
    qubit_name_1 = machine.qubits["q1"].name
    qubit_name_2 = machine.qubits["q2"].name

    params = Parameters(
        depth_min=DEPTH_MIN,
        depth_max=DEPTH_MAX,
        depth_step=DEPTH_STEP,
        n_sequences=N_SEQUENCES,
        n_shots=N_SHOTS,
        gate_set="sw",
        apply_two_qubit_gate=True,
    )
    depths = params.get_depths()
    max_depth = int(depths[-1])

    gi_a = _generate_gate_indices(N_SEQUENCES, max_depth, NUM_XEB_GATES, seed=RNG_SEED)
    gi_b = _generate_gate_indices(
        N_SEQUENCES, max_depth, NUM_XEB_GATES, seed=RNG_SEED + 1
    )

    ds_a = _generate_xeb_dataset(
        gi_a, depths, R_HIGH, N_SHOTS, "sw", qubit_name_1, q_idx=0, seed=RNG_SEED + 10
    )
    ds_b = _generate_xeb_dataset(
        gi_b, depths, R_LOW, N_SHOTS, "sw", qubit_name_2, q_idx=1, seed=RNG_SEED + 20
    )
    ds_raw = xr.merge([ds_a, ds_b])

    for var in [f"state_{qubit_name_1}", f"state_{qubit_name_2}"]:
        assert var in ds_raw.data_vars, f"Missing variable {var}"
    assert "gate_indices_0" in ds_raw.data_vars
    assert "gate_indices_1" in ds_raw.data_vars

    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        analyse_qubits=["q1", "q2"],
        param_overrides={
            "depth_min": DEPTH_MIN,
            "depth_max": DEPTH_MAX,
            "depth_step": DEPTH_STEP,
            "n_sequences": N_SEQUENCES,
            "n_shots": N_SHOTS,
            "gate_set": "sw",
            "apply_two_qubit_gate": False,
        },
    )

    # ── Save all XEB figures to artifacts ──────────────────────────────
    artifacts_dir = ARTIFACTS_BASE / NODE_NAME
    saved_figs = _save_xeb_figures(node, artifacts_dir)
    assert len(saved_figs) > 0, "No XEB figures were generated by plot_data"
    for fname in saved_figs:
        assert (artifacts_dir / fname).exists(), f"Figure not saved: {fname}"

    # ── Fit result assertions ─────────────────────────────────────────
    assert "fit_results" in node.results, "analyse_data must populate fit_results"

    fit_a = node.results["fit_results"][qubit_name_1]
    fit_b = node.results["fit_results"][qubit_name_2]

    r_a = fit_a["linear_fit"]["r"]
    r_b = fit_b["linear_fit"]["r"]

    assert not np.isnan(r_a), f"Fit for high-fidelity qubit failed: {fit_a}"
    assert not np.isnan(r_b), f"Fit for low-fidelity qubit failed: {fit_b}"

    assert 0.0 < r_a <= 1.0, f"r_a = {r_a:.4f} out of range"
    assert 0.0 < r_b <= 1.0, f"r_b = {r_b:.4f} out of range"

    assert r_a > r_b, (
        f"Expected r_a > r_b (high vs low fidelity), "
        f"got r_a={r_a:.4f}, r_b={r_b:.4f}"
    )

    assert node.outcomes[qubit_name_1] == "successful"
    assert node.outcomes[qubit_name_2] == "successful"
