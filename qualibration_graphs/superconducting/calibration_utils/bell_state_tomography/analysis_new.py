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
                }
            ),
        ],
        compat="override",
        join="outer",
    )

    # FitParameters for each qubit pair
    fit_results = {qp.name: FitParameters(fidelity=float(fidelity.sel(qubit=qp.name)), success=True) for qp in qubit_pairs}
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
        output_core_dims=[["state"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[int],
    ).assign_coords(state=("state", basis_labels))

    # Normalize to probabilities using node.parameters.num_shots (assumed constant across settings)
    probs = (counts / float(num_shots)).astype(float)

    # Order dims as (qubit_pair, tomography_basis_target, tomography_basis_control, state)
    lead = [d for d in ("qubit_pair", "tomography_basis_target", "tomography_basis_control") if d in probs.dims]
    probs = probs.transpose(*lead, "state")

    probs.name = "raw_probs"
    probs.attrs["basis_labels"] = basis_labels
    return probs


def _apply_confusion(probs: xr.DataArray, qubit_pairs) -> xr.DataArray:
    """
    Apply 4x4 confusion matrix inverse to the length-4 outcome vector for each setting.
    Clips negatives to 0, then renormalizes.
    """
    assert probs.sizes["state"] == 4, "Expected 4 outcomes (00,01,10,11)."
    state_labels = probs.coords["state"].values  # e.g. ['00','01','10','11']

    corrected_blocks = []
    for qp in qubit_pairs:
        # Select this pair: (tomography_basis_target, tomography_basis_control, state)
        sel = probs.sel(qubit_pair=qp.name).astype(float)

        # Get/validate confusion matrix
        C = getattr(qp, "confusion", None)
        C = np.asarray(C, dtype=float) if C is not None else np.eye(4, dtype=float)
        if C.shape != (4, 4):
            C = np.eye(4, dtype=float)

        # Inverse (or pseudo-inverse) once per pair
        try:
            M = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(C)

        # Wrap as DA with distinct core dims for matvec: (state_out, state_in)
        M_da = xr.DataArray(
            M,
            dims=("state_out", "state_in"),
            coords={"state_out": state_labels, "state_in": state_labels},
        )

        # Rename vector's core dim to 'state_in' so apply_ufunc can contract it
        sel_in = sel.rename({"state": "state_in"})

        # q = M @ p  (vectorized over remaining dims)
        q = xr.apply_ufunc(
            lambda M_, p_: M_ @ p_,
            M_da,
            sel_in,
            input_core_dims=[["state_out", "state_in"], ["state_in"]],
            output_core_dims=[["state_out"]],
            vectorize=True,
            dask="parallelized",
            keep_attrs=True,
        )

        # Clean up: clip negatives and renormalize along 'state_out'
        q = xr.where(q > 0.0, q, 0.0)
        sums = q.sum(dim="state_out")
        uniform = xr.zeros_like(q) + 0.25
        q = xr.where(sums > 0, q / sums, uniform)

        # Restore standard dim name 'state'
        q = q.rename({"state_out": "state"}).assign_coords(state=("state", state_labels))

        # Reattach qubit_pair and collect
        block = q.assign_coords(qubit_pair=qp.name).expand_dims("qubit_pair")
        corrected_blocks.append(block)

    corrected = xr.concat(corrected_blocks, dim="qubit_pair")
    corrected = corrected.transpose("qubit_pair", "tomography_basis_target", "tomography_basis_control", "state")
    corrected.name = "corrected_probs"
    corrected.attrs.update(probs.attrs)
    return corrected


def _pauli_tensor(probs: xr.DataArray) -> xr.DataArray:
    """
    Compute ⟨σ_i ⊗ σ_j⟩ for i,j in {I,X,Y,Z} using eigenbasis measurements.
    Assumes tomography_basis_target/control ∈ {0(X),1(Y),2(Z)} and
    outcome mapping 0→+1, 1→−1.
    """
    qp_vals = probs["qubit_pair"].values.tolist()
    T = xr.DataArray(
        np.zeros((len(qp_vals), 4, 4), dtype=float),
        dims=["qubit_pair", "pauli_i", "pauli_j"],
        coords={"qubit_pair": qp_vals, "pauli_i": _PAULIS, "pauli_j": _PAULIS},
        name="pauli_T",
    )
    T.loc[dict(pauli_i="I", pauli_j="I")] = 1.0

    # Single-qubit averages using settings with the other qubit in Z
    for a_axis in (0, 1, 2):
        # <I ⊗ σ_target> : target basis = a_axis, control fixed at Z (2)
        block_t = probs.sel(tomography_basis_target=a_axis, tomography_basis_control=2)
        e_t = xr.apply_ufunc(
            lambda v: _EIG[0]*(v[0]+v[2]) + _EIG[1]*(v[1]+v[3]),
            block_t, input_core_dims=[["state"]], output_core_dims=[[]], vectorize=True
        )
        T.loc[dict(pauli_i="I", pauli_j=_AXIS_TO_LABEL[a_axis])] = e_t

        # <σ_control ⊗ I> : control basis = a_axis, target fixed at Z (2)
        block_c = probs.sel(tomography_basis_target=2, tomography_basis_control=a_axis)
        e_c = xr.apply_ufunc(
            lambda v: _EIG[0]*(v[0]+v[1]) + _EIG[1]*(v[2]+v[3]),
            block_c, input_core_dims=[["state"]], output_core_dims=[[]], vectorize=True
        )
        T.loc[dict(pauli_i=_AXIS_TO_LABEL[a_axis], pauli_j="I")] = e_c

    # Two-qubit correlations from settings (target_axis, control_axis)
    def _corr(v):
        p00, p01, p10, p11 = v
        return _EIG[0]*_EIG[0]*p00 + _EIG[0]*_EIG[1]*p01 + _EIG[1]*_EIG[0]*p10 + _EIG[1]*_EIG[1]*p11

    for t_axis in (0, 1, 2):
        for c_axis in (0, 1, 2):
            block = probs.sel(tomography_basis_target=t_axis, tomography_basis_control=c_axis)
            e_ct = xr.apply_ufunc(_corr, block, input_core_dims=[["state"]], output_core_dims=[[]], vectorize=True)
            # Note: i ≡ control, j ≡ target
            T.loc[dict(pauli_i=_AXIS_TO_LABEL[c_axis], pauli_j=_AXIS_TO_LABEL[t_axis])] = e_ct

    return T


def _rho_from_T(T: xr.DataArray) -> xr.DataArray:
    """Reconstruct density matrix via ρ = ¼ Σ_{i,j} T_ij (σ_i ⊗ σ_j)."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    mats = {"I": I, "X": X, "Y": Y, "Z": Z}

    rhos = []
    for qp in T["qubit_pair"].values:
        Tij = T.sel(qubit_pair=qp).to_numpy()
        rho = np.zeros((4, 4), dtype=complex)
        for ii, li in enumerate(_PAULIS):
            for jj, lj in enumerate(_PAULIS):
                rho += Tij[ii, jj] * np.kron(mats[li], mats[lj])
        rho /= 4.0
        rho = (rho + rho.conj().T) / 2  # hermitize
        tr = np.trace(rho).real
        if tr > 0:
            rho /= tr
        rhos.append(rho)

    return xr.DataArray(
        np.stack(rhos, axis=0),
        dims=["qubit_pair", "row", "col"],
        coords={"qubit_pair": T["qubit_pair"].values, "row": range(4), "col": range(4)},
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
