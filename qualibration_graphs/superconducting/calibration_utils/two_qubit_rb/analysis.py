"""Analysis utilities for two-qubit randomized benchmarking experiments."""

# pylint: disable=use-implicit-booleaness-not-comparison-to-zero

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from scipy.optimize import curve_fit

AVERAGE_GATES_PER_2Q_LAYER = 1.51


@dataclass
class FitResults:
    """Stores the relevant RB fit parameters for a single qubit pair."""

    alpha: float
    fidelity: float
    success: bool
    fit_amplitude: float
    fit_offset: float
    epc: float | None = None
    average_gate_fidelity: float | None = None


def log_fitted_results(fit_results: Dict[str, FitResults], log_callable=None):
    """Log fitted RB results for all qubit pairs."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        s_qubit = f"Results for qubit pair {qp_name}: "
        s_alpha = f"\tFitted alpha: {fit_result.alpha:.6f} a.u."
        s_fidelity = f"\tFitted fidelity: {100 * fit_result.fidelity:.6f} %"

        if fit_result.success:
            s_qubit += "SUCCESS!\n"
        else:
            s_qubit += "FAIL!\n"

        log_callable(s_qubit + s_alpha + s_fidelity)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode | None = None) -> xr.Dataset:
    return ds

def rb_decay_curve(x, A, alpha, B):
    """Exponential decay model for RB survival probability."""
    return A * alpha**x + B


def clifford_fidelity_from_alpha(alpha: float, n_qubits: int = 2) -> float:
    """Average Clifford fidelity from the fitted RB decay constant."""
    d = 2**n_qubits
    r = 1 - alpha - (1 - alpha) / d
    return 1 - r


def interleaved_gate_fidelity_from_alpha(
    alpha: float, standard_rb_alpha: float, n_qubits: int = 2
) -> float:
    """Interleaved gate fidelity using https://arxiv.org/pdf/1210.7011."""
    return 1 - ((2**n_qubits - 1) * (1 - alpha / standard_rb_alpha) / 2**n_qubits)


def _survival_probability(ds_qp: xr.Dataset) -> xr.DataArray:
    """P(|00>) vs circuit depth, averaged over sequence and shots."""
    survival = (ds_qp.state == 0).mean(dim=["sequence", "shots"])
    if "qubit_pair" in survival.dims:
        survival = survival.squeeze("qubit_pair", drop=True)
    return survival


def _survival_stderr(ds_qp: xr.Dataset) -> xr.DataArray:
    """Standard error of the mean survival probability at each circuit depth."""
    n_samples = ds_qp.sizes["sequence"] * ds_qp.sizes["shots"]
    stderr = (
        (ds_qp.state == 0)
        .stack(combined=("shots", "sequence"))
        .std(dim="combined")
        / np.sqrt(n_samples)
    )
    if "qubit_pair" in stderr.dims:
        stderr = stderr.squeeze("qubit_pair", drop=True)
    return stderr


def _fit_survival(circuit_depths: np.ndarray, survival: np.ndarray) -> tuple[float, float, float]:
    popt, _ = curve_fit(
        rb_decay_curve,
        circuit_depths,
        survival,
        p0=[0.75, 0.9, 0.25],
        maxfev=10000,
    )
    return float(popt[0]), float(popt[1]), float(popt[2])


def _validate_fit(
    circuit_depths: np.ndarray,
    survival: np.ndarray,
    stderr: np.ndarray,
    fit_amplitude: float,
    alpha: float,
    fit_offset: float,
    log_callable=None,
) -> bool:
    """Return True when the fitted curve lies within 4 sigma of the data."""
    if log_callable is None:
        log_callable = print

    fitted_values = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    normalized_residuals = np.abs(fitted_values - survival) / stderr
    max_deviation = float(np.max(normalized_residuals))

    if max_deviation > 4.0:
        log_callable(
            f"Warning: Fitted curve deviates up to {max_deviation:.2f} sigma "
            "from experimental data. Consider reviewing fit quality."
        )
        return False

    log_callable(f"Fit validation passed: Maximum deviation is {max_deviation:.2f} sigma.")
    return True


def _average_gate_fidelity(
    fidelity: float,
    average_layers_per_clifford: float | None,
    average_gates_per_2q_layer: float = AVERAGE_GATES_PER_2Q_LAYER,
) -> float | None:
    if average_layers_per_clifford is None:
        return None
    error_per_2q_layer = (1 - fidelity) / average_layers_per_clifford
    error_per_gate = error_per_2q_layer / average_gates_per_2q_layer
    return 1 - error_per_gate


def _try_load_standard_rb_overlay(node: QualibrationNode, qp_name: str) -> dict | None:
    """Load and fit a reference Standard RB dataset for interleaved overlay plots."""
    standard_rb_load_id = (
        node.machine.qubit_pairs[qp_name]
        .macros[node.parameters.operation]
        .fidelity.get("StandardRB_load_id")
    )
    if standard_rb_load_id is None:
        return None

    try:
        from qualibrate.core.utils.node.content import read_node_data
        from qualibrate.core.utils.node.path_solver import get_node_dir_path
        from qualibrate_config.resolvers import get_qualibrate_config, get_qualibrate_config_path

        base_path = get_qualibrate_config(get_qualibrate_config_path()).storage.location
        node_dir = get_node_dir_path(int(standard_rb_load_id), base_path)
        std_rb_data = read_node_data(node_dir, int(standard_rb_load_id), base_path)
        std_rb_ds = process_raw_dataset(std_rb_data["ds_raw"].sel(qubit_pair=qp_name), node)

        survival = _survival_probability(std_rb_ds)
        circuit_depths = survival.circuit_depth.values
        survival_vals = survival.values
        fit_amplitude, alpha, fit_offset = _fit_survival(circuit_depths, survival_vals)
        fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)

        return {
            "circuit_depth": circuit_depths,
            "survival": survival_vals,
            "fitted_curve": fitted_curve,
            "alpha": alpha,
        }
    except Exception as exc:
        node.log(f"Could not load StandardRB overlay for {qp_name}: {exc}")
        return None


def _assign_standard_rb_overlay(da: xr.Dataset, overlay: dict) -> xr.Dataset:
    """Align a Standard RB overlay onto the interleaved dataset circuit depths."""
    circuit_depths = da.circuit_depth.values
    survival_on_depths = np.full(len(circuit_depths), np.nan)
    fitted_on_depths = np.full(len(circuit_depths), np.nan)
    overlay_depths = overlay["circuit_depth"]

    for idx, depth in enumerate(circuit_depths):
        match = np.where(overlay_depths == depth)[0]
        if match.size:
            survival_on_depths[idx] = overlay["survival"][match[0]]
            fitted_on_depths[idx] = overlay["fitted_curve"][match[0]]

    return da.assign(
        standard_rb_overlay_survival=("circuit_depth", survival_on_depths),
        standard_rb_overlay_fitted=("circuit_depth", fitted_on_depths),
        standard_rb_fit_alpha=float(overlay["alpha"]),
    )


def fit_rb_routine(da: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Fit RB survival probability vs circuit depth for one qubit pair."""
    interleaved = "interleaved" in node.name.lower()
    average_layers_per_clifford = node.namespace.get("average_layers_per_clifford")
    qp_name = str(np.asarray(da.qubit_pair.values).item())

    survival = _survival_probability(da)
    stderr = _survival_stderr(da)
    circuit_depths = survival.circuit_depth.values
    survival_vals = survival.values

    fit_amplitude, alpha, fit_offset = _fit_survival(circuit_depths, survival_vals)
    success = _validate_fit(
        circuit_depths,
        survival_vals,
        stderr.values,
        fit_amplitude,
        alpha,
        fit_offset,
        log_callable=node.log,
    )

    if interleaved:
        fidelity_dict = node.machine.qubit_pairs[qp_name].macros[node.parameters.operation].fidelity
        if "StandardRB_alpha" not in fidelity_dict:
            raise KeyError(
                f"Qubit pair {qp_name}: missing StandardRB_alpha in "
                f"macros[{node.parameters.operation!r}].fidelity. "
                "Run 37a_two_qubit_standard_rb first for this operation."
            )
        standard_rb_alpha = float(fidelity_dict["StandardRB_alpha"])
        fidelity = interleaved_gate_fidelity_from_alpha(alpha, standard_rb_alpha)
        average_gate_fidelity = None
    else:
        fidelity = clifford_fidelity_from_alpha(alpha)
        average_gate_fidelity = _average_gate_fidelity(fidelity, average_layers_per_clifford)

    fitted_curve = rb_decay_curve(circuit_depths, fit_amplitude, alpha, fit_offset)
    fitted_curve_da = xr.DataArray(
        fitted_curve,
        dims=["circuit_depth"],
        coords={"circuit_depth": circuit_depths},
    )

    assign_kwargs = {
        "survival_probability": survival,
        "survival_stderr": stderr,
        "fitted_curve": fitted_curve_da,
        "fit_amplitude": fit_amplitude,
        "fit_alpha": alpha,
        "fit_offset": fit_offset,
        "fidelity": fidelity,
        "epc": 1 - fidelity,
        "success": success,
    }
    if average_gate_fidelity is not None:
        assign_kwargs["average_gate_fidelity"] = average_gate_fidelity

    da = da.assign(**assign_kwargs)

    if interleaved:
        overlay = _try_load_standard_rb_overlay(node, qp_name)
        if overlay is not None:
            da = _assign_standard_rb_overlay(da, overlay)
        else:
            nan_overlay = np.full(len(circuit_depths), np.nan)
            da = da.assign(
                standard_rb_overlay_survival=("circuit_depth", nan_overlay),
                standard_rb_overlay_fitted=("circuit_depth", nan_overlay.copy()),
                standard_rb_fit_alpha=np.nan,
            )

    return da


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Fit RB survival curves for each qubit pair and return an augmented dataset."""
    ds_fit = ds.groupby("qubit_pair").apply(lambda da: fit_rb_routine(da, node))
    ds_fit, fit_results = _extract_relevant_parameters(ds_fit, node)
    return ds_fit, fit_results


def _extract_relevant_parameters(
    ds_fit: xr.Dataset, node: QualibrationNode
) -> Tuple[xr.Dataset, Dict[str, FitResults]]:
    """Extract RB fit parameters and create FitResults for each qubit pair."""
    qubit_pairs = node.namespace["qubit_pairs"]

    if "survival_probability" in ds_fit.data_vars:
        ds_fit.survival_probability.attrs = {"long_name": "P(|00>)", "units": "a.u."}
    if "fitted_curve" in ds_fit.data_vars:
        ds_fit.fitted_curve.attrs = {"long_name": "exponential RB fit", "units": "a.u."}
    if "fidelity" in ds_fit.data_vars:
        ds_fit.fidelity.attrs = {"long_name": "RB fidelity", "units": "a.u."}
    if "fit_alpha" in ds_fit.data_vars:
        ds_fit.fit_alpha.attrs = {"long_name": "RB decay constant alpha", "units": "a.u."}

    fit_results: Dict[str, FitResults] = {}
    for qp in qubit_pairs:
        qp_name = qp.name
        qp_data = ds_fit.sel(qubit_pair=qp_name)

        fit_results[qp_name] = FitResults(
            alpha=float(qp_data.fit_alpha.values),
            fidelity=float(qp_data.fidelity.values),
            success=bool(qp_data.success.values),
            fit_amplitude=float(qp_data.fit_amplitude.values),
            fit_offset=float(qp_data.fit_offset.values),
            epc=float(qp_data.epc.values),
            average_gate_fidelity=(
                float(qp_data.average_gate_fidelity.values)
                if "average_gate_fidelity" in qp_data
                else None
            ),
        )

    return ds_fit, fit_results
