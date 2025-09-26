import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Literal
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V
from ..data_process_utils import reshape_control_target_val2dim

# ------------ Results container ------------


@dataclass
class FitParameters:
    """Stores the relevant tomography results for a single qubit pair."""

    fidelity: float
    success: bool


def log_fitted_results(fit_results: Dict[str, FitParameters], log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qp, fr in fit_results.items():
        log_callable(f"[{qp}] Bell={fr.bell_state}  Fidelity={fr.fidelity:.4f}  Success={fr.success}")


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    if node.parameters.use_state_discrimination:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
    else:
        ds = reshape_control_target_val2dim(ds, state_discrimination=node.parameters.use_state_discrimination)
        ds = convert_IQ_to_V(ds, qubits=None, qubit_pairs=node.namespace["qubit_pairs"])
    return ds


# ------------ Public API ------------


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    """
    Perform Bell-state tomography and compute fidelity for each qubit pair.
    """
    ds_fit = ds
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(ds_fit: xr.Dataset, node: QualibrationNode):
    """Add tomography metadata (probs, Pauli tensor, density matrix, fidelity) and build fit_results."""
    qubit_pairs = node.namespace["qubit_pairs"]
    num_shots = int(node.parameters.num_shots)

    # Pipeline
    raw_probs = _compute_raw_probs(ds_fit, num_shots)
    corrected_probs = _apply_confusion(raw_probs, qubit_pairs)
    pauli_T = _pauli_tensor(corrected_probs)
    rho = _rho_from_T(pauli_T)
    rho_ideal = _ideal_bell_density(str(node.parameters.bell_state))
    fidelity = _fidelity_pure(rho, rho_ideal)

    # Attach to dataset
    ds_out = xr.merge(
        [
            ds_fit,
            xr.Dataset(
                {
                    "raw_probs": raw_probs,
                    "corrected_probs": corrected_probs,
                    "pauli_T": pauli_T,
                    "density_matrix": rho,
                    "fidelity": fidelity,
                    "bell_state": str(node.parameters.bell_state),
                }
            ),
        ],
        compat="override",
        join="outer",
    )

    # FitParameters for each qubit pair
    fit_results = {qp.name: FitParameters(fidelity=float(fidelity.sel(qubit_pair=qp.name)), success=True) for qp in qubit_pairs}
    return ds_out, fit_results


# ------------ Helpers (top-level) ------------

_PAULIS = ["I", "X", "Y", "Z"]
_AXIS_TO_LABEL = {0: "X", 1: "Y", 2: "Z"}
_EIG = {0: +1.0, 1: -1.0}  # measurement outcome mapping: 0→+1, 1→−1


def _detect_shot_dim(ds: xr.Dataset) -> str:
    """Heuristically detect the shot dimension name."""
    for cand in ("n_shots", "N", "shots", "shot"):
        if cand in ds.dims:
            return cand
    # Fallback: use first >1 sized dim from state along control_target='c'
    if "state" in ds and "control_target" in ds.coords:
        ctl = ds["state"].sel(control_target="c")
        for d in ctl.dims:
            if ds.sizes[d] > 1 and d not in ("qubit_pair", "tomography_basis_control", "tomography_basis_target"):
                return d
    raise ValueError("Could not detect shot dimension in dataset.")


def _compute_raw_probs(ds: xr.Dataset, num_shots: int) -> xr.DataArray:
    """
    Build empirical probabilities p(s) for s ∈ {00,01,10,11}
    grouped by (qubit_pair, tomography_basis_target, tomography_basis_control).
    """
    shot_dim = _detect_shot_dim(ds)

    # pair dimension name
    pair_dim = "qubit_pair" if "qubit_pair" in ds.dims else ("qubit" if "qubit" in ds.dims else None)
    if pair_dim is None:
        raise ValueError("Expected a pair dimension ('qubit_pair' or 'qubit') in dataset.")

    # tomography dims (your newer naming)
    t_ctl = "tomography_basis_control"
    t_tgt = "tomography_basis_target"
    if not (t_ctl in ds.dims and t_tgt in ds.dims):
        raise ValueError(f"Expected dims '{t_ctl}' and '{t_tgt}' in dataset.")

    basis_labels = ["00", "01", "10", "11"]

    state_c = ds["state"].sel(control_target="c")  # (qubit_pair, n_shots, tomo_c, tomo_t)
    state_t = ds["state"].sel(control_target="t")  # same dims

    joint_idx = (state_c * 2 + state_t).astype(int)  # values in {0,1,2,3}

    eye4 = np.eye(4, dtype=int)

    def _one_hot_sum(arr):
        # arr shape: (shots,)
        return eye4[arr].sum(axis=0)

    counts = xr.apply_ufunc(
        _one_hot_sum,
        joint_idx,
        input_core_dims=[[shot_dim]],
        output_core_dims=[["outcome"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    ).assign_coords(outcome=("outcome", basis_labels))

    # ---- normalize to probabilities ----
    probs = (counts / float(num_shots)).astype(float)

    # order dims: (pair, target, control, outcome)
    lead = [d for d in (pair_dim, t_tgt, t_ctl) if d in probs.dims]
    probs = probs.transpose(*lead, "outcome")

    probs.name = "raw_probs"
    probs.attrs["basis_labels"] = basis_labels
    return probs


def _apply_confusion(probs: xr.DataArray, qubit_pairs) -> xr.DataArray:
    """
    Apply a 4x4 confusion matrix inverse to each outcome vector.
    Clips negatives to 0, then renormalizes. Works with outcome dim named
    'outcome' (preferred) or 'state' (fallback).
    """
    # dims / coords detection
    outcome_dim = "outcome" if "outcome" in probs.dims else ("state" if "state" in probs.dims else None)
    if outcome_dim is None:
        raise ValueError("Expected an outcome axis named 'outcome' (preferred) or 'state'.")
    if probs.sizes[outcome_dim] != 4:
        raise ValueError(f"Expected 4 outcomes on '{outcome_dim}', got {probs.sizes[outcome_dim]}.")

    state_labels = probs.coords[outcome_dim].values  # e.g., ['00','01','10','11']

    pair_dim = "qubit_pair" if "qubit_pair" in probs.dims else ("qubit" if "qubit" in probs.dims else None)
    if pair_dim is None:
        raise ValueError("Expected a pair dimension ('qubit_pair' or 'qubit') in probs.")

    t_ctl = "tomography_basis_control"
    t_tgt = "tomography_basis_target"

    corrected_blocks = []
    for qp in qubit_pairs:
        # Select this pair (drops the pair dim)
        sel = probs.sel({pair_dim: qp.name}).astype(float)

        # Confusion matrix for this pair
        C = getattr(qp, "confusion", None)
        C = np.asarray(C, dtype=float) if C is not None else np.eye(4, dtype=float)
        if C.shape != (4, 4):
            C = np.eye(4, dtype=float)

        # Invert (or pseudo-invert) once per pair
        try:
            M = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(C)

        # Prepare matvec with distinct core dims to avoid duplicate-dim issues
        M_da = xr.DataArray(
            M,
            dims=("outcome_out", "outcome_in"),
            coords={"outcome_out": state_labels, "outcome_in": state_labels},
        )
        sel_in = sel.rename({outcome_dim: "outcome_in"})

        # q = M @ p  (vectorized over remaining dims)
        q = xr.apply_ufunc(
            lambda A, x: A @ x,
            M_da,
            sel_in,
            input_core_dims=[["outcome_out", "outcome_in"], ["outcome_in"]],
            output_core_dims=[["outcome_out"]],
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )

        # Clip negatives, renormalize per setting
        q = xr.where(q > 0.0, q, 0.0)
        sums = q.sum(dim="outcome_out")
        uniform = xr.zeros_like(q) + 0.25
        q = xr.where(sums > 0, q / sums, uniform)

        # Restore canonical outcome dim name used by input
        q = q.rename({"outcome_out": outcome_dim}).assign_coords({outcome_dim: state_labels})

        # Reattach pair dim/coord and collect
        block = q.expand_dims({pair_dim: [qp.name]})
        corrected_blocks.append(block)

    corrected = xr.concat(corrected_blocks, dim=pair_dim)
    corrected = corrected.transpose(pair_dim, t_tgt, t_ctl, outcome_dim)
    corrected.name = "corrected_probs"
    corrected.attrs.update(probs.attrs)
    return corrected


def _pauli_tensor(probs: xr.DataArray) -> xr.DataArray:
    """
    Compute ⟨σ_i ⊗ σ_j⟩ for i,j in {I,X,Y,Z} using eigenbasis measurements.
    Assumes tomography_basis_target/control ∈ {0(X),1(Y),2(Z)} and
    outcome mapping 0→+1, 1→−1.

    Convention here: i ≡ CONTROL, j ≡ TARGET (matches your earlier code paths).
    """
    # ---- dim detection ----
    outcome_dim = "outcome" if "outcome" in probs.dims else ("state" if "state" in probs.dims else None)
    if outcome_dim is None:
        raise ValueError("Expected a 4-outcome axis named 'outcome' (preferred) or 'state'.")
    pair_dim = "qubit_pair" if "qubit_pair" in probs.dims else ("qubit" if "qubit" in probs.dims else None)
    if pair_dim is None:
        raise ValueError("Expected a pair dimension 'qubit_pair' or 'qubit' in probs.")
    t_ctl = "tomography_basis_control"
    t_tgt = "tomography_basis_target"
    if t_ctl not in probs.dims or t_tgt not in probs.dims:
        raise ValueError(f"Expected dims '{t_ctl}' and '{t_tgt}' in probs.")

    # ---- container for T ----
    pair_vals = probs.coords[pair_dim].values.tolist()
    T = xr.DataArray(
        np.zeros((len(pair_vals), 4, 4), dtype=float),
        dims=[pair_dim, "pauli_i", "pauli_j"],
        coords={pair_dim: pair_vals, "pauli_i": _PAULIS, "pauli_j": _PAULIS},
        name="pauli_T",
    )
    T.loc[{ "pauli_i": "I", "pauli_j": "I" }] = 1.0

    # ---- helper maps on a length-4 prob vector v = [p00,p01,p10,p11] ----
    def _targ_marg(v):  # ⟨σ_target⟩
        return _EIG[0]*(v[0]+v[2]) + _EIG[1]*(v[1]+v[3])
    def _ctrl_marg(v):  # ⟨σ_control⟩
        return _EIG[0]*(v[0]+v[1]) + _EIG[1]*(v[2]+v[3])
    def _corr(v):       # ⟨σ_target ⊗ σ_control⟩ with weights (+1,-1,-1,+1)
        p00,p01,p10,p11 = v
        return _EIG[0]*_EIG[0]*p00 + _EIG[0]*_EIG[1]*p01 + _EIG[1]*_EIG[0]*p10 + _EIG[1]*_EIG[1]*p11

    # ---- single-qubit terms, fixing the other qubit to Z (basis=2) ----
    for a_axis in (0, 1, 2):
        # <I ⊗ σ_target> : vary TARGET basis = a_axis, CONTROL fixed at Z (2)
        block_t = probs.sel({t_tgt: a_axis, t_ctl: 2})
        e_t = xr.apply_ufunc(
            _targ_marg, block_t,
            input_core_dims=[[outcome_dim]],
            output_core_dims=[[]],
            vectorize=True,
        )
        T.loc[{ "pauli_i": "I", "pauli_j": _AXIS_TO_LABEL[a_axis] }] = e_t

        # <σ_control ⊗ I> : vary CONTROL basis = a_axis, TARGET fixed at Z (2)
        block_c = probs.sel({t_tgt: 2, t_ctl: a_axis})
        e_c = xr.apply_ufunc(
            _ctrl_marg, block_c,
            input_core_dims=[[outcome_dim]],
            output_core_dims=[[]],
            vectorize=True,
        )
        T.loc[{ "pauli_i": _AXIS_TO_LABEL[a_axis], "pauli_j": "I" }] = e_c

    # ---- two-qubit correlations from each (TARGET axis, CONTROL axis) ----
    for t_axis in (0, 1, 2):
        for c_axis in (0, 1, 2):
            block = probs.sel({t_tgt: t_axis, t_ctl: c_axis})
            e_ct = xr.apply_ufunc(
                _corr, block,
                input_core_dims=[[outcome_dim]],
                output_core_dims=[[]],
                vectorize=True,
            )
            # Note: i ≡ CONTROL, j ≡ TARGET
            T.loc[{ "pauli_i": _AXIS_TO_LABEL[c_axis], "pauli_j": _AXIS_TO_LABEL[t_axis] }] = e_ct

    return T


def _rho_from_T(T: xr.DataArray) -> xr.DataArray:
    """Reconstruct density matrix via ρ = ¼ Σ_{i,j} T_ij (σ_control_i ⊗ σ_target_j)."""
    # detect pair dim to stay compatible with either 'qubit_pair' or 'qubit'
    pair_dim = "qubit_pair" if "qubit_pair" in T.dims else ("qubit" if "qubit" in T.dims else None)
    if pair_dim is None:
        raise ValueError("Expected a pair dimension 'qubit_pair' or 'qubit' in T.")

    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mats = {"I": I, "X": X, "Y": Y, "Z": Z}

    rhos = []
    for qp in T.coords[pair_dim].values:
        Tij = T.sel({pair_dim: qp}).to_numpy()  # shape (4,4) over (pauli_i, pauli_j)
        rho = np.zeros((4, 4), dtype=complex)
        for ii, li in enumerate(_PAULIS):       # li acts on CONTROL
            for jj, lj in enumerate(_PAULIS):   # lj acts on TARGET
                rho += Tij[ii, jj] * np.kron(mats[li], mats[lj])
        rho /= 4.0
        rho = (rho + rho.conj().T) / 2  # hermitize
        tr = np.trace(rho).real
        if tr > 0:
            rho /= tr
        rhos.append(rho)

    return xr.DataArray(
        np.stack(rhos, axis=0),
        dims=[pair_dim, "row", "col"],
        coords={pair_dim: T.coords[pair_dim].values, "row": range(4), "col": range(4)},
        name="density_matrix",
    )


def _ideal_bell_density(label: str) -> np.ndarray:
    """
    Supported:
      - "00-11"  -> |Φ+> = (|00> + |11>)/√2
      - "01-10"  -> |Ψ+> = (|01> + |10>)/√2
    """
    if label == "00-11":
        psi = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
    elif label == "01-10":
        psi = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
    else:
        raise ValueError(f"Unsupported bell_state '{label}'.")
    return np.outer(psi, psi.conj())


def _fidelity_pure(rho_da: xr.DataArray, rho_ideal: np.ndarray) -> xr.DataArray:
    """For pure target |ψ>, F = ⟨ψ|ρ|ψ⟩ = Tr(ρ ρ_ideal)."""
    def _fid(r):
        return float(np.real(np.trace(r @ rho_ideal)))
    F = xr.apply_ufunc(
        _fid, rho_da, input_core_dims=[["row", "col"]], output_core_dims=[[]], vectorize=True
    ).clip(min=0.0, max=1.0)
    return F.rename("fidelity")
