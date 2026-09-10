"""Analysis module for GHZ state tomography."""

import itertools
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.confusion_matrix import (
    compute_kron_confusion_matrices,
    get_nq_confusion_matrix,
    get_state_labels,
    recover_prepared_probs,
    resolve_target_dim,
)
from calibration_utils.common_utils.fidelity import (
    are_tomography_metrics_valid,
    fidelity_with_pure_state,
    purity,
)

from .helpers import get_density_matrix, get_pauli_data_nq, ghz_state_vector


@dataclass
class FitResults:
    """Stores GHZ tomography fit parameters for a single qubit group."""

    fidelity_kron: float
    """State fidelity after Kronecker readout mitigation."""
    purity_kron: float
    """State purity after Kronecker readout mitigation."""
    fidelity_nq: Optional[float] = None
    """State fidelity after NQ readout mitigation, if available."""
    purity_nq: Optional[float] = None
    """State purity after NQ readout mitigation, if available."""
    nq_mitigation_valid: Optional[bool] = None
    """``True``/``False`` when an NQ matrix was loaded; ``None`` when none was available."""
    success: bool = True
    """Whether the tomography analysis completed successfully."""


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None) -> None:
    """Log GHZ tomography fidelities and purities for all qubit groups."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for group_name, fit_result in fit_results.items():
        status = "SUCCESS!\n" if fit_result.success else "FAIL!\n"
        num_qubits = len(group_name.split("-"))
        lines = [f"Results for qubit group {group_name}: {status}"]
        if fit_result.success:
            lines.append(f"\t[kron] Fidelity: {fit_result.fidelity_kron:.3f}")
            lines.append(f"\t[kron] Purity: {fit_result.purity_kron:.3f}")
        else:
            lines.append("\t[kron] Fidelity: N/A")
            lines.append("\t[kron] Purity: N/A")
        if fit_result.fidelity_nq is not None and fit_result.purity_nq is not None:
            lines.append(f"\t[nq] Fidelity: {fit_result.fidelity_nq:.3f}")
            lines.append(f"\t[nq] Purity: {fit_result.purity_nq:.3f}")
        elif fit_result.nq_mitigation_valid is False:
            lines.append(f"\tWarning: {num_qubits}Q mitigation failed validation (fidelity, purity, or trace)")
        elif fit_result.success:
            lines.append(f"\tWarning: {num_qubits}Q confusion matrix not found")
        log_callable("\n".join(lines))


def _tomo_axis_names(num_qubits: int) -> list[str]:
    return [f"tomo_axis_{idx}" for idx in range(num_qubits)]


def _mitigate_tomography_probs(
    results_xr: xr.DataArray,
    group_name: str,
    conf_mat: np.ndarray,
    num_qubits: int,
) -> np.ndarray:
    """Apply readout mitigation for one qubit group across all tomography settings."""
    corrected = []

    for tomo_axes in itertools.product([0, 1, 2], repeat=num_qubits):
        # ``process_raw_dataset`` stacks ``tomo_axis_0``, ``tomo_axis_1``, ... into ``tomo_axis``.
        measured = results_xr.sel(qubit_pair=group_name, tomo_axis=tomo_axes).data
        corrected.append(recover_prepared_probs(conf_mat, measured))

    return np.array(corrected).reshape(*([3] * num_qubits), 2**num_qubits)


def _stack_corrected_tomography_xr(probs: np.ndarray, num_qubits: int) -> xr.DataArray:
    """Wrap mitigated tomography probabilities in a stacked xarray."""
    tomo_axis_names = _tomo_axis_names(num_qubits)
    corrected_xr = xr.DataArray(
        probs,
        dims=[*tomo_axis_names, "state"],
        coords={
            **{axis_name: [0, 1, 2] for axis_name in tomo_axis_names},
            "state": get_state_labels(num_qubits),
        },
    )
    return corrected_xr.stack(tomo_axis=tomo_axis_names)


def process_raw_dataset(ds: xr.Dataset, num_qubits: int, num_shots: int) -> xr.DataArray:
    """Convert raw tomography counts into unmitigated probability tensor."""
    target_dim = resolve_target_dim(ds)
    shot_dim = "n" if "n" in ds.dims else "N"
    if shot_dim not in ds.dims:
        raise ValueError("Dataset must contain a shot dimension named 'n' or 'N'.")
    if target_dim != "qubit_pair":
        ds = ds.rename({target_dim: "qubit_pair"})

    tomo_axis_names = _tomo_axis_names(num_qubits)
    states = list(range(2**num_qubits))
    results = [(ds.state == state).sum(dim=shot_dim) / num_shots for state in states]

    results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
    if "dim_0" in results_xr.dims:
        results_xr = results_xr.rename({"dim_0": "state"})
    return results_xr.stack(tomo_axis=tomo_axis_names)


def _analyze_qubit_group(
    results_xr: xr.DataArray,
    qg,
    machine,
    kron_conf: np.ndarray,
    ideal_psi: np.ndarray,
    num_qubits: int,
) -> Tuple[xr.Dataset, np.ndarray, Optional[xr.Dataset], Optional[np.ndarray], FitResults]:
    """Reconstruct density matrices and metrics for one qubit group."""
    corrected_kron_probs = _mitigate_tomography_probs(results_xr, qg.name, kron_conf, num_qubits)
    corrected_kron_xr = _stack_corrected_tomography_xr(corrected_kron_probs, num_qubits)
    paulis_kron = get_pauli_data_nq(corrected_kron_xr, num_qubits)
    rho_kron = get_density_matrix(paulis_kron, num_qubits)
    fidelity_kron = fidelity_with_pure_state(rho_kron, ideal_psi)
    purity_kron = purity(rho_kron)

    fidelity_nq = None
    purity_nq = None
    nq_mitigation_valid = None
    paulis_nq = None
    rho_nq = None

    conf_mat_nq = get_nq_confusion_matrix([q.name for q in qg.qubits], machine)
    if conf_mat_nq is not None:
        corrected_nq_probs = _mitigate_tomography_probs(results_xr, qg.name, conf_mat_nq, num_qubits)
        corrected_nq_xr = _stack_corrected_tomography_xr(corrected_nq_probs, num_qubits)
        paulis_nq = get_pauli_data_nq(corrected_nq_xr, num_qubits)
        rho_nq = get_density_matrix(paulis_nq, num_qubits)
        candidate_fidelity = fidelity_with_pure_state(rho_nq, ideal_psi)
        candidate_purity = purity(rho_nq)
        if are_tomography_metrics_valid(candidate_fidelity, candidate_purity, rho_nq):
            fidelity_nq = candidate_fidelity
            purity_nq = candidate_purity
            nq_mitigation_valid = True
        else:
            paulis_nq = None
            rho_nq = None
            nq_mitigation_valid = False

    success = are_tomography_metrics_valid(fidelity_kron, purity_kron, rho_kron)
    fit_result = FitResults(
        fidelity_kron=fidelity_kron,
        purity_kron=purity_kron,
        fidelity_nq=fidelity_nq,
        purity_nq=purity_nq,
        nq_mitigation_valid=nq_mitigation_valid,
        success=success,
    )
    return paulis_kron, rho_kron, paulis_nq, rho_nq, fit_result


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[
    Dict[str, Dict[str, np.ndarray]],
    Dict[str, Dict[str, xr.Dataset]],
    Dict[str, FitResults],
]:
    """Reconstruct density matrices and compute GHZ fidelity and purity."""
    qubit_groups = node.namespace["qubit_groups"]
    num_qubits = qubit_groups[0].num_qubits
    ideal_psi = ghz_state_vector(num_qubits)

    results_xr = process_raw_dataset(ds, num_qubits, node.parameters.num_shots)
    kron_confs = compute_kron_confusion_matrices({qg.name: qg.qubits for qg in qubit_groups})

    rhos_by_method: Dict[str, Dict[str, np.ndarray]] = {"kron": {}, "nq": {}}
    paulis_by_method: Dict[str, Dict[str, xr.Dataset]] = {"kron": {}, "nq": {}}
    fit_results: Dict[str, FitResults] = {}

    for qg in qubit_groups:
        try:
            paulis_kron, rho_kron, paulis_nq, rho_nq, fit_result = _analyze_qubit_group(
                results_xr,
                qg,
                node.machine,
                kron_confs[qg.name],
                ideal_psi,
                num_qubits,
            )
            rhos_by_method["kron"][qg.name] = rho_kron
            paulis_by_method["kron"][qg.name] = paulis_kron
            if rho_nq is not None and paulis_nq is not None:
                rhos_by_method["nq"][qg.name] = rho_nq
                paulis_by_method["nq"][qg.name] = paulis_nq
            fit_results[qg.name] = fit_result
        except Exception as exc:
            logging.getLogger(__name__).exception(
                "GHZ tomography analysis failed for qubit group %s", qg.name
            )
            if node.log is not None:
                node.log(f"Analysis failed for {qg.name}: {exc}")
            fit_results[qg.name] = FitResults(
                fidelity_kron=np.nan,
                purity_kron=np.nan,
                fidelity_nq=None,
                purity_nq=None,
                nq_mitigation_valid=None,
                success=False,
            )

    return rhos_by_method, paulis_by_method, fit_results
