"""Analysis module for GHZ Z-basis population measurement."""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode

from calibration_utils.common_utils.confusion_matrix import (
    compute_kron_confusion_matrices,
    get_nq_confusion_matrix,
    recover_prepared_probs,
)
from calibration_utils.common_utils.fidelity import (
    is_valid_probability_vector,
    is_valid_unit_metric,
    z_basis_ghz_fidelity,
)


@dataclass
class FitResults:
    """Stores Z-basis population fidelities for a single qubit group."""

    fidelity_kron: float
    """Z-basis population fidelity after Kronecker mitigation."""
    fidelity_nq: Optional[float] = None
    """Z-basis population fidelity after NQ mitigation, if available."""
    fidelity_difference: Optional[float] = None
    """NQ minus Kron Z-basis population fidelity, if NQ mitigation was applied."""
    nq_mitigation_valid: Optional[bool] = None
    """``True``/``False`` when an NQ matrix was loaded; ``None`` when none was available."""
    success: bool = True
    """Whether the analysis completed successfully."""


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None) -> None:
    """Log Z-basis population fidelities for all qubit groups."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for group_name, fit_result in fit_results.items():
        status = "SUCCESS!\n" if fit_result.success else "FAIL!\n"
        num_qubits = len(group_name.split("-"))
        lines = [f"Results for qubit group {group_name}: {status}"]
        lines.append(f"\t(Kron) Z-basis population fidelity: {fit_result.fidelity_kron:.4f}")
        if fit_result.fidelity_nq is not None:
            lines.append(f"\t({num_qubits}Q) Z-basis population fidelity: {fit_result.fidelity_nq:.4f}")
            if fit_result.fidelity_difference is not None:
                diff_sign = "+" if fit_result.fidelity_difference > 0 else ""
                lines.append(f"\tDelta (NQ - Kron): {diff_sign}{fit_result.fidelity_difference:.4f}")
        elif fit_result.nq_mitigation_valid is False:
            lines.append(f"\tWarning: {num_qubits}Q mitigation failed validation (populations or fidelity)")
        elif fit_result.success:
            lines.append(f"\tWarning: {num_qubits}Q confusion matrix not found")
        log_callable("\n".join(lines))


def _compute_z_basis_populations(
    ds: xr.Dataset,
    group_names,
    num_qubits: int,
    num_shots: int,
) -> Dict[str, np.ndarray]:
    """Compute raw Z-basis outcome probabilities for each qubit group."""
    num_states = 2**num_qubits
    populations = {}

    for group_name in group_names:
        counts = np.zeros(num_states, dtype=float)
        group_ds = ds.sel(qubit_pair=group_name)
        for state in range(num_states):
            counts[state] = (group_ds.state == state).sum().values
        populations[group_name] = counts / num_shots

    return populations


def _z_basis_analysis_success(fidelity_kron: float, corrected_kron: np.ndarray) -> bool:
    """Return True when Kron-mitigated populations and fidelity are physically valid."""
    if not is_valid_probability_vector(corrected_kron):
        return False
    return is_valid_unit_metric(fidelity_kron)


def fit_raw_data(
    ds: xr.Dataset,
    node: QualibrationNode,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, FitResults]]:
    """Apply readout mitigation and compute Z-basis population fidelities."""
    qubit_groups = node.namespace["qubit_groups"]
    num_qubits = qubit_groups[0].num_qubits

    populations = _compute_z_basis_populations(
        ds,
        [qg.name for qg in qubit_groups],
        num_qubits,
        node.parameters.num_shots,
    )

    kron_confs = compute_kron_confusion_matrices({qg.name: qg.qubits for qg in qubit_groups})
    corrected_kron: Dict[str, np.ndarray] = {}
    corrected_nq: Dict[str, np.ndarray] = {}
    fit_results: Dict[str, FitResults] = {}

    for i, qg in enumerate(qubit_groups):
        try:
            measured = populations[qg.name]
            corrected_kron[qg.name] = recover_prepared_probs(kron_confs[qg.name], measured)
            fidelity_kron = z_basis_ghz_fidelity(corrected_kron[qg.name])

            conf_mat_nq = get_nq_confusion_matrix(node.parameters.qubit_groups[i], node.machine)
            fidelity_nq = None
            fidelity_difference = None
            nq_mitigation_valid = None
            if conf_mat_nq is not None:
                corrected_nq_probs = recover_prepared_probs(conf_mat_nq, measured)
                fidelity_nq = z_basis_ghz_fidelity(corrected_nq_probs)
                if is_valid_probability_vector(corrected_nq_probs) and is_valid_unit_metric(fidelity_nq):
                    corrected_nq[qg.name] = corrected_nq_probs
                    fidelity_difference = fidelity_nq - fidelity_kron
                    nq_mitigation_valid = True
                else:
                    fidelity_nq = None
                    nq_mitigation_valid = False

            success = _z_basis_analysis_success(fidelity_kron, corrected_kron[qg.name])
            fit_results[qg.name] = FitResults(
                fidelity_kron=fidelity_kron,
                fidelity_nq=fidelity_nq,
                fidelity_difference=fidelity_difference,
                nq_mitigation_valid=nq_mitigation_valid,
                success=success,
            )
        except Exception:
            fit_results[qg.name] = FitResults(
                fidelity_kron=np.nan,
                fidelity_nq=None,
                fidelity_difference=None,
                success=False,
            )

    return corrected_kron, corrected_nq, fit_results
