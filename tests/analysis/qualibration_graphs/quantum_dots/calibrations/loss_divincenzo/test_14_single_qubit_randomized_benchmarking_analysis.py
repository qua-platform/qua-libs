"""Analysis test for 14_single_qubit_randomized_benchmarking.

Synthetic RB data is generated using the same Clifford group algebra tables
as the QUA PPU (CAYLEY, INVERSES, decomposition), combined with a classical
Bloch sphere simulation and a depolarizing noise model.  No quantum simulation
(virtual_qpu) is required.

The test verifies:
1. Clifford algebra correctness (group law + Bloch sphere round-trip).
2. Dataset structure and decay trend of the synthetic data.
3. Successful exponential decay fit (alpha, gate_fidelity, error_per_clifford).
4. State update consistency.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from calibration_utils.single_qubit_randomized_benchmarking.clifford_tables import (
    CAYLEY,
    INVERSES,
    compose_sequence,
    invert,
    decomposition,
)
from calibration_utils.single_qubit_randomized_benchmarking.parameters import Parameters

from .conftest import ANALYSE_QUBITS
from .quam_factory import create_minimal_quam

# =============================================================================
# Constants
# =============================================================================

NODE_NAME = "14_single_qubit_randomized_benchmarking"
MAX_CIRCUIT_DEPTH = 512  # log-scale → depths [2, 4, 8, 16, 32, 64, 128, 256, 512]
NUM_CIRCUITS = 20  # circuits per depth
NUM_SHOTS = 100  # shots per circuit (for Bernoulli sampling)
DEPOL_PER_PHYSICAL_GATE = 0.01  # depolarizing error per physical native gate
RNG_SEED = 42

# =============================================================================
# Bloch sphere rotation matrices
# =============================================================================

# Virtual Z gates are noiseless frame rotations; physical gates are the rest.
_VIRTUAL_GATES = {"z90", "z180", "zm90"}


def _Rx(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def _Ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def _Rz(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


GATE_BLOCH_MATRIX: dict[str, np.ndarray] = {
    "x90": _Rx(+np.pi / 2),
    "x180": _Rx(np.pi),
    "xm90": _Rx(-np.pi / 2),
    "y90": _Ry(+np.pi / 2),
    "y180": _Ry(np.pi),
    "ym90": _Ry(-np.pi / 2),
    "z90": _Rz(+np.pi / 2),
    "z180": _Rz(np.pi),
    "zm90": _Rz(-np.pi / 2),
}

# =============================================================================
# Simulation helpers
# =============================================================================


def simulate_rb_circuit(cliff_sequence: list, noise_prob: float) -> float:
    """Simulate one RB circuit on the Bloch sphere.

    Applies each Clifford gate in ``cliff_sequence`` (left = first applied)
    via its native-gate decomposition.  Physical gates shrink the Bloch vector
    by the depolarizing factor ``(1 - 4/3 * noise_prob)``; virtual Z gates are
    noiseless frame rotations.

    Parameters
    ----------
    cliff_sequence : list of int
        Clifford indices applied in chronological order.
    noise_prob : float
        Depolarizing error probability per physical gate.  0 for noiseless.

    Returns
    -------
    float
        Survival probability P(|0⟩) after the circuit.
    """
    bloch = np.array([0.0, 0.0, 1.0])  # |0⟩ = north pole
    for cliff_idx in cliff_sequence:
        for gate in decomposition(int(cliff_idx)):
            bloch = GATE_BLOCH_MATRIX[gate] @ bloch
            if gate not in _VIRTUAL_GATES and noise_prob > 0:
                bloch *= 1.0 - 4.0 / 3.0 * noise_prob
    return float((1.0 + bloch[2]) / 2.0)


def generate_rb_ds_raw(
    depths: np.ndarray,
    num_circuits: int,
    num_shots: int,
    noise_prob: float,
    seed: int,
    qubit_name: str = "Q1",
) -> xr.Dataset:
    """Generate a synthetic RB dataset via Clifford algebra + Bloch sphere simulation.

    Uses the same incremental composition algorithm as the QUA PPU:
    - One random circuit of maximum length is generated per circuit index.
    - Shorter depths are truncations of the same random sequence (standard RB).
    - The recovery Clifford is computed from INVERSES[cumulative_clifford].

    Parameters
    ----------
    depths : 1-D ndarray
        Sorted array of circuit depths (e.g. [2, 4, 8, 16, 32]).
    num_circuits : int
        Number of random circuits per depth.
    num_shots : int
        Shots per circuit (Bernoulli sampling to match experimental data format).
    noise_prob : float
        Depolarizing error per physical gate.
    seed : int
        RNG seed for reproducibility.
    qubit_name : str
        Name used for the data variable (``state_{qubit_name}``).  Should
        match ``qubit.name`` as returned by the QuAM for the target qubit.

    Returns
    -------
    xr.Dataset
        Dataset with data variable ``state_{qubit_name}`` of shape
        ``[num_circuits, len(depths)]``, values in [0, 1].
    """
    rng = np.random.default_rng(seed)
    state_data = np.zeros((num_circuits, len(depths)))
    max_random = int(depths[-1]) - 1  # depth d → d-1 random Cliffords

    for ci in range(num_circuits):
        # Pre-generate a random Clifford sequence of maximum length
        random_seq = rng.integers(0, 24, size=max_random)

        # Incrementally compose and snapshot recovery Clifford at each depth
        total_clifford = 0  # identity
        prev_count = 0

        for di, depth in enumerate(depths):
            num_random = int(depth) - 1

            # Extend cumulative composition from last checkpoint
            for i in range(prev_count, num_random):
                cliff = int(random_seq[i])
                total_clifford = CAYLEY[cliff][total_clifford]
            prev_count = num_random

            inverse_cliff = INVERSES[total_clifford]
            full_seq = list(random_seq[:num_random]) + [inverse_cliff]

            p_survive = simulate_rb_circuit(full_seq, noise_prob)
            # Binomial sampling to mimic shot-averaged experimental measurement
            state_data[ci, di] = rng.binomial(num_shots, np.clip(p_survive, 0.0, 1.0)) / num_shots

    return xr.Dataset(
        {f"state_{qubit_name}": xr.DataArray(state_data, dims=["circuit", "depth"])},
        coords={
            "circuit": np.arange(num_circuits),
            "depth": depths,
        },
    )


# =============================================================================
# Test
# =============================================================================


@pytest.mark.analysis
def test_14_single_qubit_rb_analysis(analysis_runner):
    """Single-qubit RB analysis using Clifford-algebra synthetic data."""

    # 1. Generate depth schedule from Parameters
    params = Parameters(
        max_circuit_depth=MAX_CIRCUIT_DEPTH,
        log_scale=True,
        num_circuits_per_length=NUM_CIRCUITS,
        num_shots=NUM_SHOTS,
    )
    depths = params.get_depths()

    # 2. Clifford algebra sanity checks (no noise, ideal circuit)
    rng_check = np.random.default_rng(RNG_SEED)
    check_seq = list(rng_check.integers(0, 24, size=7))
    total = compose_sequence(check_seq)
    inv_cliff = invert(total)

    # Group law: C_inv . C_total = identity (index 0)
    assert CAYLEY[inv_cliff][total] == 0, "inverse must undo the cumulative Clifford"

    # Bloch sphere round-trip: noiseless (seq + inverse) leaves qubit in |0⟩
    survival_ideal = simulate_rb_circuit(check_seq + [inv_cliff], noise_prob=0.0)
    assert abs(survival_ideal - 1.0) < 1e-9, (
        f"Ideal RB circuit survival should be exactly 1.0, got {survival_ideal}"
    )

    # 3. Determine the actual qubit name used by the analysis pipeline.
    #    The QuAM factory stores qubits at machine.qubits["Q1"] but the qubit's
    #    .name property returns the underlying quantum-dot id (e.g. "virtual_dot_1").
    #    fit_raw_data() looks up state_{qubit.name}, so the dataset variable must
    #    match that name.
    _machine = create_minimal_quam()
    qubit_name = _machine.qubits[ANALYSE_QUBITS[0]].name  # e.g. "virtual_dot_1"
    state_var = f"state_{qubit_name}"

    # 4. Generate synthetic RB dataset with the correct variable name
    ds_raw = generate_rb_ds_raw(
        depths=depths,
        num_circuits=NUM_CIRCUITS,
        num_shots=NUM_SHOTS,
        noise_prob=DEPOL_PER_PHYSICAL_GATE,
        seed=RNG_SEED,
        qubit_name=qubit_name,
    )

    # 5. Dataset structure assertions
    assert state_var in ds_raw.data_vars, f"ds_raw must contain {state_var}"
    assert ds_raw[state_var].shape == (NUM_CIRCUITS, len(depths))
    assert np.all(ds_raw[state_var].values >= 0.0)
    assert np.all(ds_raw[state_var].values <= 1.0)

    # 6. Survival probability should decay with increasing depth on average
    mean_survival = ds_raw[state_var].mean("circuit").values
    assert mean_survival[0] > mean_survival[-1], (
        "Mean survival probability should decrease from shortest to longest depth"
    )

    # 7. Run the full analysis pipeline (analyse_data → plot_data → update_state)
    node = analysis_runner(
        node_name=NODE_NAME,
        ds_raw=ds_raw,
        param_overrides={
            "qubits": ANALYSE_QUBITS,
            "max_circuit_depth": MAX_CIRCUIT_DEPTH,
            "log_scale": True,
            "num_circuits_per_length": NUM_CIRCUITS,
            "num_shots": NUM_SHOTS,
        },
    )

    # 8. Fit result assertions (keyed by qubit.name, not the dict key)
    assert "fit_results" in node.results, "analyse_data must populate node.results['fit_results']"
    fit = node.results["fit_results"][qubit_name]
    assert fit["success"], f"RB exponential decay fit should succeed: {fit}"
    assert 0.9 < fit["alpha"] < 1.0, (
        f"Depolarising parameter alpha should be in (0.9, 1.0), got {fit['alpha']:.4f}"
    )
    assert 0.9 < fit["gate_fidelity"] < 1.0, (
        f"Gate fidelity should be in (0.9, 1.0), got {fit['gate_fidelity']:.4f}"
    )
    assert np.isfinite(fit["error_per_clifford"]), "error_per_clifford should be a finite number"

    # 9. State update: gate_fidelity["averaged"] must match the fit result
    qubit_obj = node.machine.qubits[ANALYSE_QUBITS[0]]
    if (
        hasattr(qubit_obj, "gate_fidelity")
        and isinstance(qubit_obj.gate_fidelity, dict)
        and "averaged" in qubit_obj.gate_fidelity
    ):
        assert abs(qubit_obj.gate_fidelity["averaged"] - fit["gate_fidelity"]) < 1e-6
