"""Analysis module for Bell state tomography calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.linalg import sqrtm

from .parameters import require_bell_tomography_prerequisites

# Pauli matrices for two-qubit tomography (0=I, 1=X, 2=Y, 3=Z)
_PAULI_0 = np.array([[1, 0], [0, 1]])
_PAULI_X = np.array([[0, 1], [1, 0]])
_PAULI_Y = np.array([[0, -1j], [1j, 0]])
_PAULI_Z = np.array([[1, 0], [0, -1]])
_PAULIS = [_PAULI_0, _PAULI_X, _PAULI_Y, _PAULI_Z]


@dataclass
class FitResults:
    """Stores the relevant Bell state tomography experiment fit parameters for a single qubit pair."""

    fidelity: float
    """State fidelity with respect to the ideal Bell state"""
    purity: float
    """Purity of the reconstructed density matrix, Tr(ρ²)"""
    success: bool
    """Whether the tomography analysis completed successfully"""

def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None)-> None:
    """
    Logs the node-specific fitted results for all qubit pairs.

    Parameters:
    -----------
    fit_results : Dict[str, FitResults]
        Dictionary containing FitResults for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "
        s_fidelity = f"\tFidelity: {fit_result.fidelity:.3f}"
        s_purity = f"\tPurity: {fit_result.purity:.3f}"

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_message = s_qubit + s_fidelity + "\n" + s_purity
        log_callable(log_message)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process the raw dataset to ensure it has a 'state' variable suitable for tomography.

    If the dataset has state_control and state_target only, derives state = state_control * 2 + state_target.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment
    node : QualibrationNode
        The calibration node containing qubit pairs information

    Returns:
    --------
    xr.Dataset
        Processed dataset with 'state' variable
    """
    if "state" in ds.data_vars:
        return ds

    if "state_control" in ds.data_vars and "state_target" in ds.data_vars:
        return ds.assign(state=ds.state_control * 2 + ds.state_target)

    return ds


def get_pauli_data(corrected_results_xr_sel: xr.DataArray) -> xr.Dataset:
    """
    Map measurement probabilities to Pauli expectation values for two-qubit tomography.

    For each of the 9 tomography settings (tomo_axis_control, tomo_axis_target in {0,1,2} for X,Y,Z),
    the expectation value <sigma_i otimes sigma_j> = P(00) - P(01) - P(10) + P(11).

    Single-qubit Paulis (IX, IY, IZ, XI, YI, ZI) are obtained by marginalizing over the other qubit.

    Parameters:
    -----------
    corrected_results_xr_sel : xr.DataArray
        Corrected measurement probabilities with dims (tomo_axis_control, tomo_axis_target, state)
        and state coords ['00','01','10','11']

    Returns:
    --------
    xr.Dataset
        Pauli expectation values with coord 'pauli_op' for the 16 Pauli operators (II through ZZ),
        stored as Dataset with variable name "pauli" for JSON serialization compatibility.
    """
    pauli_ops = [
        "II",
        "IX",
        "IY",
        "IZ",
        "XI",
        "XX",
        "XY",
        "XZ",
        "YI",
        "YX",
        "YY",
        "YZ",
        "ZI",
        "ZX",
        "ZY",
        "ZZ",
    ]

    def _get_probs(tc, tt):
        p = corrected_results_xr_sel.sel(tomo_axis_control=tc, tomo_axis_target=tt)
        if "tomo_axis" in p.dims:
            p = p.unstack("tomo_axis")
        p_arr = p.values.flatten()
        if len(p_arr) >= 4:
            return p_arr[0], p_arr[1], p_arr[2], p_arr[3]
        return (
            float(p.sel(state="00").values),
            float(p.sel(state="01").values),
            float(p.sel(state="10").values),
            float(p.sel(state="11").values),
        )

    def _two_qubit(p00, p01, p10, p11):
        return p00 - p01 - p10 + p11

    def _marginalize_control(p00, p01, p10, p11):
        return (p00 + p10) - (p01 + p11)

    def _marginalize_target(p00, p01, p10, p11):
        return (p00 + p01) - (p10 + p11)

    def _iz(p00, p01, p10, p11):
        return (p00 - p01) + (p10 - p11)

    pauli_vals = [1.0]  # II

    # Row 1: IX, IY, IZ (control=I, target=X,Y,Z)
    p = _get_probs(2, 0)
    pauli_vals.append(_marginalize_control(*p))
    p = _get_probs(2, 1)
    pauli_vals.append(_marginalize_control(*p))
    p = _get_probs(2, 2)
    pauli_vals.append(_iz(*p))

    # Row 2: XI, XX, XY, XZ (control=X)
    p = _get_probs(0, 2)
    pauli_vals.append(_marginalize_target(*p))
    p = _get_probs(0, 0)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(0, 1)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(0, 2)
    pauli_vals.append(_two_qubit(*p))

    # Row 3: YI, YX, YY, YZ (control=Y)
    p = _get_probs(1, 2)
    pauli_vals.append(_marginalize_target(*p))
    p = _get_probs(1, 0)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(1, 1)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(1, 2)
    pauli_vals.append(_two_qubit(*p))

    # Row 4: ZI, ZX, ZY, ZZ (control=Z)
    p = _get_probs(2, 2)
    pauli_vals.append(_marginalize_target(*p))
    p = _get_probs(2, 0)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(2, 1)
    pauli_vals.append(_two_qubit(*p))
    p = _get_probs(2, 2)
    pauli_vals.append(_two_qubit(*p))

    pauli_arr = xr.DataArray(
        pauli_vals,
        dims=["pauli_op"],
        coords={"pauli_op": pauli_ops},
    )
    return pauli_arr.to_dataset(name="pauli")


def get_density_matrix(paulis_data: xr.Dataset | xr.DataArray) -> np.ndarray:
    """
    Reconstruct the 4x4 density matrix from Pauli expectation values.

    rho = (1/4) * sum_{i,j} <sigma_i otimes sigma_j> * (sigma_i otimes sigma_j)

    Parameters:
    -----------
    paulis_data : xr.Dataset | xr.DataArray
        Pauli expectation values with coord 'pauli_op' (II, IX, ..., ZZ).
        If Dataset, expects a "pauli" data variable.

    Returns:
    --------
    np.ndarray
        4x4 density matrix (complex)
    """
    if isinstance(paulis_data, xr.Dataset):
        pauli_arr = paulis_data["pauli"]
    else:
        pauli_arr = paulis_data
    pauli_ops = pauli_arr.coords["pauli_op"].values
    pauli_vals = pauli_arr.values

    rho = np.zeros((4, 4), dtype=complex)
    idx = 0
    for i in range(4):
        for j in range(4):
            val = pauli_vals[idx]
            pauli_ij = np.kron(_PAULIS[i], _PAULIS[j])
            rho += val * pauli_ij
            idx += 1
    rho /= 4

    return rho


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, xr.Dataset], Dict[str, np.ndarray], Dict[str, FitResults]]:
    """
    Reconstruct density matrices from tomography data and compute fidelity and purity.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset from the experiment (with 'state' variable)
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs

    Returns:
    --------
    Tuple containing:
        - corrected_results_ds: Dataset of corrected state probabilities (variable "corrected_results")
        - paulis_data: Dict of Pauli expectation values per qubit pair (each a Dataset with "pauli" variable)
        - rhos: Dict of density matrices per qubit pair
        - fit_results: Dict of FitResults per qubit pair
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    num_shots = node.parameters.num_shots
    require_bell_tomography_prerequisites(qubit_pairs, node.parameters.operation)

    # Normalize dimension names
    pair_dim = "qubit_pair" if "qubit_pair" in ds.dims else "qubit"
    shot_dim = "n" if "n" in ds.dims else "N"

    states = [0, 1, 2, 3]
    results_list = []
    for state in states:
        results_list.append((ds.state == state).sum(dim=shot_dim) / num_shots)

    results_xr = xr.concat(results_list, dim=xr.DataArray(states, name="state"))
    if "dim_0" in results_xr.dims:
        results_xr = results_xr.rename({"dim_0": "state"})

    # Apply confusion matrix correction
    corrected_results = []
    for qp in qubit_pairs:
        corrected_results_qp = []
        for tomo_axis_control in [0, 1, 2]:
            corrected_results_control = []
            for tomo_axis_target in [0, 1, 2]:
                results_sel = results_xr.sel(
                    tomo_axis_control=tomo_axis_control,
                    tomo_axis_target=tomo_axis_target,
                    **{pair_dim: qp.name},
                )
                probs = np.array(results_sel.values.flatten()[:4], dtype=float)
                try:
                    confusion_inv = np.linalg.inv(qp.confusion)
                except np.linalg.LinAlgError as exc:
                    raise ValueError(
                        f"Qubit pair {qp.name!r} has a singular confusion matrix. "
                        "Re-run node 35_two_qubit_confusion_matrix."
                    ) from exc
                probs = confusion_inv @ probs
                probs = probs * (probs > 0)
                probs = probs / probs.sum()
                corrected_results_control.append(probs)
            corrected_results_qp.append(corrected_results_control)
        corrected_results.append(corrected_results_qp)

    corrected_results_xr = xr.DataArray(
        corrected_results,
        dims=[pair_dim, "tomo_axis_control", "tomo_axis_target", "state"],
        coords={
            pair_dim: [qp.name for qp in qubit_pairs],
            "tomo_axis_control": [0, 1, 2],
            "tomo_axis_target": [0, 1, 2],
            "state": ["00", "01", "10", "11"],
        },
    )

    ideal_dat = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2
    s_ideal = sqrtm(ideal_dat)

    paulis_data = {}
    rhos = {}
    fit_results = {}

    for qp in qubit_pairs:
        try:
            sel = corrected_results_xr.sel({pair_dim: qp.name})
            paulis_data[qp.name] = get_pauli_data(sel)
            rhos[qp.name] = get_density_matrix(paulis_data[qp.name])

            fidelity = float(np.abs(np.trace(sqrtm(s_ideal @ rhos[qp.name] @ s_ideal))) ** 2)
            purity = float(np.abs(np.trace(rhos[qp.name] @ rhos[qp.name])))

            fit_results[qp.name] = FitResults(fidelity=fidelity, purity=purity, success=True)
        except Exception:
            fit_results[qp.name] = FitResults(fidelity=np.nan, purity=np.nan, success=False)
            pauli_ops = [
                "II",
                "IX",
                "IY",
                "IZ",
                "XI",
                "XX",
                "XY",
                "XZ",
                "YI",
                "YX",
                "YY",
                "YZ",
                "ZI",
                "ZX",
                "ZY",
                "ZZ",
            ]
            pauli_arr = xr.DataArray([np.nan] * 16, dims=["pauli_op"], coords={"pauli_op": pauli_ops})
            paulis_data[qp.name] = pauli_arr.to_dataset(name="pauli")
            rhos[qp.name] = np.zeros((4, 4), dtype=complex)

    corrected_results_ds = corrected_results_xr.to_dataset(name="corrected_results")
    return corrected_results_ds, paulis_data, rhos, fit_results
