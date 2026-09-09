"""Analysis module for Bell state tomography calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from calibration_utils.common_utils.confusion_matrix import (
    compute_kron_confusion_matrices,
    recover_prepared_probs,
)
from calibration_utils.common_utils.fidelity import are_tomography_metrics_valid, fidelity_with_state, purity
from qualibrate import QualibrationNode

from .parameters import require_bell_tomography_prerequisites

# Pauli matrices for two-qubit tomography (0=I, 1=X, 2=Y, 3=Z)
_PAULI_0 = np.array([[1, 0], [0, 1]])
_PAULI_X = np.array([[0, 1], [1, 0]])
_PAULI_Y = np.array([[0, -1j], [1j, 0]])
_PAULI_Z = np.array([[1, 0], [0, -1]])
_PAULIS = [_PAULI_0, _PAULI_X, _PAULI_Y, _PAULI_Z]

_IDEAL_BELL_RHO = np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 1]]) / 2


@dataclass
class FitResults:
    """Stores the relevant Bell state tomography experiment fit parameters for a single qubit pair."""

    fidelity_kron: float
    """State fidelity after uncorrelated (Kronecker) readout mitigation."""
    purity_kron: float
    """Purity after uncorrelated readout mitigation."""
    fidelity_joint: Optional[float] = None
    """State fidelity after correlated (joint 2Q) readout mitigation."""
    purity_joint: Optional[float] = None
    """Purity after correlated readout mitigation."""
    joint_mitigation_valid: Optional[bool] = None
    """``True``/``False`` when a joint matrix was applied; ``None`` when unavailable."""
    success: bool = True
    """Whether the tomography analysis completed successfully."""


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None) -> None:
    """
    Log the node-specific fitted results for all qubit pairs.

    Reports uncorrelated (Kronecker) and correlated (joint 2Q) readout mitigation
    metrics when available.

    Parameters
    ----------
    fit_results : Dict[str, FitResults]
        Dictionary containing FitResults for each qubit pair.
    log_callable : callable, optional
        Logger for logging the fitted results. If None, a default logger is used.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        status = "SUCCESS!\n" if fit_result.success else "FAIL!\n"
        lines = [f"Results for qubit pair {qp_name}: {status}"]
        lines.append(f"\t[kron] Fidelity: {fit_result.fidelity_kron:.3f}")
        lines.append(f"\t[kron] Purity: {fit_result.purity_kron:.3f}")
        if fit_result.fidelity_joint is not None and fit_result.purity_joint is not None:
            lines.append(f"\t[joint] Fidelity: {fit_result.fidelity_joint:.3f}")
            lines.append(f"\t[joint] Purity: {fit_result.purity_joint:.3f}")
        elif fit_result.joint_mitigation_valid is False:
            lines.append("\tWarning: joint readout mitigation failed validation (fidelity, purity, or trace)")
        log_callable("\n".join(lines))


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """
    Process the raw dataset to ensure it has a 'state' variable suitable for tomography.

    If the dataset has state_control and state_target only, derives
    ``state = state_control * 2 + state_target``.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from the experiment.
    node : QualibrationNode
        The calibration node containing qubit pairs information.

    Returns
    -------
    xr.Dataset
        Processed dataset with a ``state`` variable.
    """
    if "state" in ds.data_vars:
        return ds

    if "state_control" in ds.data_vars and "state_target" in ds.data_vars:
        return ds.assign(state=ds.state_control * 2 + ds.state_target)

    return ds


def get_pauli_data(corrected_results_xr_sel: xr.DataArray) -> xr.Dataset:
    """
    Map measurement probabilities to Pauli expectation values for two-qubit tomography.

    For each of the 9 tomography settings (``tomo_axis_control``, ``tomo_axis_target``
    in ``{0, 1, 2}`` for X, Y, Z), the two-qubit expectation value is

    ``<sigma_i ⊗ sigma_j> = P(00) - P(01) - P(10) + P(11)``.

    Single-qubit Paulis (IX, IY, IZ, XI, YI, ZI) are obtained by marginalizing over
    the other qubit.

    Parameters
    ----------
    corrected_results_xr_sel : xr.DataArray
        Corrected measurement probabilities with dims
        ``(tomo_axis_control, tomo_axis_target, state)`` and state coords
        ``['00', '01', '10', '11']``.

    Returns
    -------
    xr.Dataset
        Pauli expectation values with coord ``pauli_op`` for the 16 Pauli operators
        (II through ZZ), stored as a Dataset with variable name ``pauli`` for JSON
        serialization compatibility.
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

    ``rho = (1/4) * sum_{i,j} <sigma_i ⊗ sigma_j> * (sigma_i ⊗ sigma_j)``

    Parameters
    ----------
    paulis_data : xr.Dataset | xr.DataArray
        Pauli expectation values with coord ``pauli_op`` (II, IX, ..., ZZ).
        If Dataset, expects a ``pauli`` data variable.

    Returns
    -------
    np.ndarray
        4x4 density matrix (complex).
    """
    if isinstance(paulis_data, xr.Dataset):
        pauli_arr = paulis_data["pauli"]
    else:
        pauli_arr = paulis_data
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


def _measured_probabilities(ds: xr.Dataset, qubit_pairs, num_shots: int) -> Tuple[xr.DataArray, str]:
    """
    Convert raw tomography counts into unmitigated outcome probabilities.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from the experiment (with a ``state`` variable).
    qubit_pairs
        Active qubit pairs from the node namespace.
    num_shots : int
        Number of shots used to normalize counts.

    Returns
    -------
    results_xr : xr.DataArray
        Measured probabilities with tomography and state dimensions.
    pair_dim : str
        Name of the qubit-pair dimension in ``results_xr`` (``qubit_pair`` or ``qubit``).
    """
    pair_dim = "qubit_pair" if "qubit_pair" in ds.dims else "qubit"
    shot_dim = "n" if "n" in ds.dims else "N"

    states = [0, 1, 2, 3]
    results_list = [(ds.state == state).sum(dim=shot_dim) / num_shots for state in states]

    results_xr = xr.concat(results_list, dim=xr.DataArray(states, name="state"))
    if "dim_0" in results_xr.dims:
        results_xr = results_xr.rename({"dim_0": "state"})
    return results_xr, pair_dim


def _mitigate_bell_probs(
    results_xr: xr.DataArray,
    qp_name: str,
    conf_mat: np.ndarray,
    pair_dim: str,
) -> xr.DataArray:
    """
    Apply readout mitigation for one qubit pair across all tomography settings.

    Parameters
    ----------
    results_xr : xr.DataArray
        Unmitigated measurement probabilities.
    qp_name : str
        Qubit pair name to select from ``results_xr``.
    conf_mat : np.ndarray
        Confusion matrix in ``conf[measured, prepared]`` layout.
    pair_dim : str
        Name of the qubit-pair dimension in ``results_xr``.

    Returns
    -------
    xr.DataArray
        Mitigated probabilities with dims
        ``(tomo_axis_control, tomo_axis_target, state)``.

    Raises
    ------
    ValueError
        If the confusion matrix is singular or otherwise invalid.
    """
    corrected_results = []
    for tomo_axis_control in [0, 1, 2]:
        corrected_results_control = []
        for tomo_axis_target in [0, 1, 2]:
            results_sel = results_xr.sel(
                tomo_axis_control=tomo_axis_control,
                tomo_axis_target=tomo_axis_target,
                **{pair_dim: qp_name},
            )
            probs = np.array(results_sel.values.flatten()[:4], dtype=float)
            try:
                probs = recover_prepared_probs(conf_mat, probs)
            except (ValueError, np.linalg.LinAlgError) as exc:
                raise ValueError(
                    f"Qubit pair {qp_name!r} has a singular or invalid confusion matrix. "
                    "Re-run node 35_two_qubit_confusion_matrix."
                ) from exc
            corrected_results_control.append(probs)
        corrected_results.append(corrected_results_control)

    return xr.DataArray(
        corrected_results,
        dims=["tomo_axis_control", "tomo_axis_target", "state"],
        coords={
            "tomo_axis_control": [0, 1, 2],
            "tomo_axis_target": [0, 1, 2],
            "state": ["00", "01", "10", "11"],
        },
    )


def _analyze_qubit_pair(
    results_xr: xr.DataArray,
    qp,
    pair_dim: str,
    kron_conf: np.ndarray,
) -> Tuple[xr.Dataset, np.ndarray, Optional[xr.Dataset], Optional[np.ndarray], FitResults]:
    """
    Reconstruct density matrices and metrics for one qubit pair.

    Applies both uncorrelated (Kronecker product of 1Q matrices) and correlated
    (joint 2Q matrix from node 35) readout mitigation, then reconstructs ρ and
    computes fidelity and purity for each path.

    Parameters
    ----------
    results_xr : xr.DataArray
        Unmitigated measurement probabilities.
    qp
        Qubit pair object with ``name``, ``confusion``, and qubit references.
    pair_dim : str
        Name of the qubit-pair dimension in ``results_xr``.
    kron_conf : np.ndarray
        Uncorrelated reference confusion matrix for this pair.

    Returns
    -------
    paulis_kron : xr.Dataset
        Pauli expectations after Kronecker mitigation.
    rho_kron : np.ndarray
        Density matrix after Kronecker mitigation.
    paulis_joint : xr.Dataset | None
        Pauli expectations after joint mitigation, if validation passed.
    rho_joint : np.ndarray | None
        Density matrix after joint mitigation, if validation passed.
    fit_result : FitResults
        Fidelity, purity, and success flags for both mitigation paths.
    """
    corrected_kron_xr = _mitigate_bell_probs(results_xr, qp.name, kron_conf, pair_dim)
    paulis_kron = get_pauli_data(corrected_kron_xr)
    rho_kron = get_density_matrix(paulis_kron)
    fidelity_kron = fidelity_with_state(rho_kron, _IDEAL_BELL_RHO)
    purity_kron = purity(rho_kron)

    fidelity_joint = None
    purity_joint = None
    joint_mitigation_valid = None
    paulis_joint = None
    rho_joint = None

    joint_conf = np.asarray(qp.confusion)
    corrected_joint_xr = _mitigate_bell_probs(results_xr, qp.name, joint_conf, pair_dim)
    paulis_joint = get_pauli_data(corrected_joint_xr)
    rho_joint = get_density_matrix(paulis_joint)
    candidate_fidelity = fidelity_with_state(rho_joint, _IDEAL_BELL_RHO)
    candidate_purity = purity(rho_joint)
    if are_tomography_metrics_valid(candidate_fidelity, candidate_purity, rho_joint):
        fidelity_joint = candidate_fidelity
        purity_joint = candidate_purity
        joint_mitigation_valid = True
    else:
        paulis_joint = None
        rho_joint = None
        joint_mitigation_valid = False

    success = are_tomography_metrics_valid(fidelity_kron, purity_kron, rho_kron)
    fit_result = FitResults(
        fidelity_kron=fidelity_kron,
        purity_kron=purity_kron,
        fidelity_joint=fidelity_joint,
        purity_joint=purity_joint,
        joint_mitigation_valid=joint_mitigation_valid,
        success=success,
    )
    return paulis_kron, rho_kron, paulis_joint, rho_joint, fit_result


def fit_raw_data(
    ds: xr.Dataset, node: QualibrationNode
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, xr.Dataset]], Dict[str, FitResults]]:
    """
    Reconstruct density matrices from tomography data and compute fidelity and purity.

    For each qubit pair, applies uncorrelated (Kronecker) and correlated (joint 2Q)
    readout mitigation, reconstructs the Bell-state density matrix from Pauli
    expectations, and extracts fidelity and purity with respect to the ideal Bell state.

    Parameters
    ----------
    ds : xr.Dataset
        Raw dataset from the experiment (with a ``state`` variable).
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns
    -------
    rhos_by_method : Dict[str, Dict[str, np.ndarray]]
        Density matrices keyed by mitigation method (``kron``, ``joint``) and pair name.
    paulis_by_method : Dict[str, Dict[str, xr.Dataset]]
        Pauli expectation datasets keyed by mitigation method and pair name.
    fit_results : Dict[str, FitResults]
        Fidelity, purity, and success flags per qubit pair.
    """
    qubit_pairs = node.namespace["qubit_pairs"]
    require_bell_tomography_prerequisites(qubit_pairs, node.parameters.operation)

    results_xr, pair_dim = _measured_probabilities(ds, qubit_pairs, node.parameters.num_shots)
    kron_confs = compute_kron_confusion_matrices(
        {qp.name: [qp.qubit_control, qp.qubit_target] for qp in qubit_pairs}
    )

    rhos_by_method: Dict[str, Dict[str, np.ndarray]] = {"kron": {}, "joint": {}}
    paulis_by_method: Dict[str, Dict[str, xr.Dataset]] = {"kron": {}, "joint": {}}
    fit_results: Dict[str, FitResults] = {}

    for qp in qubit_pairs:
        try:
            paulis_kron, rho_kron, paulis_joint, rho_joint, fit_result = _analyze_qubit_pair(
                results_xr,
                qp,
                pair_dim,
                kron_confs[qp.name],
            )
            rhos_by_method["kron"][qp.name] = rho_kron
            paulis_by_method["kron"][qp.name] = paulis_kron
            if rho_joint is not None and paulis_joint is not None:
                rhos_by_method["joint"][qp.name] = rho_joint
                paulis_by_method["joint"][qp.name] = paulis_joint
            fit_results[qp.name] = fit_result
        except Exception:
            fit_results[qp.name] = FitResults(
                fidelity_kron=np.nan,
                purity_kron=np.nan,
                fidelity_joint=None,
                purity_joint=None,
                joint_mitigation_valid=None,
                success=False,
            )

    return rhos_by_method, paulis_by_method, fit_results
