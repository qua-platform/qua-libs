import logging
from typing import Dict, List, Tuple

import numpy as np
import xarray as xr

from qualibrate import QualibrationNode
from qualibration_libs.data import convert_IQ_to_V

from .fit_utils import (
    calculate_crosstalk,
    fit_linear,
    fit_lorentzian_for_each_detuning_fixed,
)
from .program import get_flux_detuning_in_v


def _format_crosstalk_cell(results: dict) -> str:
    if not results.get("success"):
        return "FAIL"
    crosstalk_pct = 100 * results.get("crosstalk_coefficient", np.nan)
    crosstalk_err_pct = 100 * results.get("crosstalk_coefficient_uncertainty", np.nan)
    if np.isfinite(crosstalk_pct) and np.isfinite(crosstalk_err_pct):
        return f"{crosstalk_pct:.2f}±{crosstalk_err_pct:.2f}"
    if np.isfinite(crosstalk_pct):
        return f"{crosstalk_pct:.2f}"
    return "N/A"


def _aggressor_names_from_fit_results(fit_results: Dict, aggressor_qubits=None) -> List[str]:
    if aggressor_qubits:
        return list(aggressor_qubits)

    aggressor_names = []
    for target_results in fit_results.values():
        for name, results in target_results.items():
            if name.startswith("_") or results.get("pair_type") != "cross":
                continue
            if name not in aggressor_names:
                aggressor_names.append(name)
    return aggressor_names


def build_crosstalk_matrix(
    fit_results: Dict,
    aggressor_qubits=None,
) -> Tuple[List[str], List[str], List[List[str]], np.ndarray, np.ndarray]:
    """Build crosstalk matrix rows/columns and numeric values for logging and plotting."""
    target_names = list(fit_results.keys())
    aggressor_names = _aggressor_names_from_fit_results(fit_results, aggressor_qubits)

    cells: List[List[str]] = []
    values_pct = np.full((len(target_names), len(aggressor_names)), np.nan)
    errors_pct = np.full((len(target_names), len(aggressor_names)), np.nan)

    for i, target_name in enumerate(target_names):
        row: List[str] = []
        for j, aggressor_name in enumerate(aggressor_names):
            results = fit_results.get(target_name, {}).get(aggressor_name)
            if not results or results.get("pair_type") != "cross":
                row.append("-")
                continue

            row.append(_format_crosstalk_cell(results))
            if results.get("success"):
                coefficient = results.get("crosstalk_coefficient", np.nan)
                uncertainty = results.get("crosstalk_coefficient_uncertainty", np.nan)
                if np.isfinite(coefficient):
                    values_pct[i, j] = 100 * coefficient
                if np.isfinite(uncertainty):
                    errors_pct[i, j] = 100 * uncertainty
        cells.append(row)

    return target_names, aggressor_names, cells, values_pct, errors_pct


def _format_crosstalk_matrix_table(
    target_names: List[str],
    aggressor_names: List[str],
    cells: List[List[str]],
) -> str:
    """Format crosstalk results as an aligned text matrix."""
    label_width = max(len("target"), max((len(name) for name in target_names), default=0))
    col_widths = [
        max(len(aggressor_names[j]), max((len(cells[i][j]) for i in range(len(cells))), default=0)) + 2
        for j in range(len(aggressor_names))
    ]

    lines = [
        " " * label_width + "".join(f"{name:>{w}}" for name, w in zip(aggressor_names, col_widths)),
    ]
    for target_name, row in zip(target_names, cells):
        lines.append(f"{target_name:<{label_width}}" + "".join(f"{cell:>{w}}" for cell, w in zip(row, col_widths)))
    return "\n".join(lines)


def log_fitted_results(
    fit_results: Dict,
    log_callable=None,
    aggressor_qubits=None,
):
    """Log cross-talk coefficients as an aggressor -> target matrix."""
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    target_names, aggressor_names, cells, _, _ = build_crosstalk_matrix(fit_results, aggressor_qubits)

    if not target_names or not aggressor_names:
        log_callable("(no cross-talk pairs)")
        return

    log_callable(
        "Crosstalk coefficient matrix (%), columns = aggressor, rows = target\n"
        f"{_format_crosstalk_matrix_table(target_names, aggressor_names, cells)}"
    )


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Convert IQ data to volts and attach spectroscopy coordinates for plotting."""
    target_qubits = node.namespace["target_qubits"]
    pairs_by_target = node.namespace["pairs_by_target"]
    self_sweep_grids = node.namespace["self_sweep_grids"]
    expected_frequency_offsets = node.namespace["expected_frequency_offsets"]
    cross_detuning = ds.detuning.values
    cross_flux_bias = ds.flux_bias.values

    ds = convert_IQ_to_V(ds, target_qubits)
    ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})

    n_pairs = max(len(pairs_by_target[target_qubit.name]) for target_qubit in target_qubits)
    aggressor_names = np.full((len(target_qubits), n_pairs), "", dtype=object)
    for i, target_qubit in enumerate(target_qubits):
        pair_names = pairs_by_target[target_qubit.name]
        aggressor_names[i, : len(pair_names)] = pair_names

    full_freq = []
    flux_bias_sweep = []
    detuning_sweep = []
    for target_qubit in target_qubits:
        freq_row = []
        flux_row = []
        det_row = []
        for aggressor_name in pairs_by_target[target_qubit.name]:
            if node.parameters.measure_self and target_qubit.name == aggressor_name:
                detunings = self_sweep_grids["detuning"]
                fluxes = self_sweep_grids["flux_bias"]
            else:
                detunings = cross_detuning
                fluxes = cross_flux_bias
            freq_row.append(detunings + target_qubit.xy.RF_frequency + expected_frequency_offsets[target_qubit.name])
            flux_row.append(fluxes)
            det_row.append(detunings)
        full_freq.append(freq_row)
        flux_bias_sweep.append(flux_row)
        detuning_sweep.append(det_row)

    ds = ds.assign_coords(
        pair=np.arange(n_pairs),
        aggressor=(["qubit", "pair"], aggressor_names),
        full_freq=(["qubit", "pair", "detuning"], np.array(full_freq)),
        flux_bias_sweep=(["qubit", "pair", "flux_bias"], np.array(flux_bias_sweep)),
        detuning_sweep=(["qubit", "pair", "detuning"], np.array(detuning_sweep)),
    )
    ds.full_freq.attrs = {"long_name": "Frequency", "units": "Hz"}
    return ds


def fit_lorentzian_peaks(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Fit Lorentzian peaks for each flux bias point to extract peak frequencies."""
    target_qubits = node.namespace["target_qubits"]
    pairs_by_target = node.namespace["pairs_by_target"]
    peak_data = []

    for target_qubit in target_qubits:
        for pair_idx, aggressor_name in enumerate(pairs_by_target[target_qubit.name]):
            if not aggressor_name:
                continue
            panel = ds.sel(qubit=target_qubit.name, pair=pair_idx)
            da = panel.I.assign_coords(
                flux_bias=("flux_bias", panel.flux_bias_sweep.values),
                detuning=("detuning", panel.detuning_sweep.values),
            )
            peak_freq, peak_freq_err, flux_bias = fit_lorentzian_for_each_detuning_fixed(da)

            if isinstance(peak_freq, xr.DataArray):
                peak_freq = peak_freq.data
            if isinstance(peak_freq_err, xr.DataArray):
                peak_freq_err = peak_freq_err.data
            if isinstance(flux_bias, xr.DataArray):
                flux_bias = flux_bias.data

            peak_data.append(
                xr.Dataset(
                    {
                        "peak_frequencies": (("flux_bias",), peak_freq),
                        "peak_frequency_errors": (("flux_bias",), peak_freq_err),
                    },
                    coords={
                        "flux_bias": flux_bias,
                        "qubit": target_qubit.name,
                        "aggressor": aggressor_name,
                    },
                )
            )

    tmp = xr.concat(peak_data, dim="pair", join="outer")
    tmp = tmp.assign_coords(
        qubit=("pair", [pair_ds.qubit.item() for pair_ds in peak_data]),
        aggressor=("pair", [pair_ds.aggressor.item() for pair_ds in peak_data]),
    )
    return tmp.set_index(pair=["qubit", "aggressor"])


def _get_self_slope_model(target_qubit, node: QualibrationNode) -> float:
    flux_detuning = get_flux_detuning_in_v(node.parameters, target_qubit)
    return get_target_qubit_slope_at_flux_detuning(target_qubit, flux_detuning)


def _log_self_slope_check(
    target_qubit_name: str,
    self_slope_measured: float,
    self_slope_model: float,
    node: QualibrationNode,
):
    if not np.isfinite(self_slope_measured) or not np.isfinite(self_slope_model) or self_slope_model == 0:
        return
    ratio = self_slope_measured / self_slope_model
    if abs(ratio - 1) > node.parameters.self_slope_tolerance:
        node.log(
            f"Self slope check for {target_qubit_name}: measured/model = {ratio:.2f} "
            f"(tolerance {node.parameters.self_slope_tolerance}). "
            "Check flux detuning or freq_vs_flux_01_quad_term (09a)."
        )


def fit_linear_crosstalk(peak_results: xr.Dataset, node: QualibrationNode) -> Dict:
    """Fit linear slopes and compute cross-talk coefficients using measured or model self slopes."""
    fit_results: Dict[str, Dict] = {
        str(target_qubit_name.data): {} for target_qubit_name in peak_results.qubit
    }
    measured_slopes: Dict[Tuple[str, str], Dict] = {}

    for pair in peak_results.pair:
        target_qubit_name = str(pair.qubit.data)
        aggressor_qubit_name = str(pair.aggressor.data)
        peak_freq = peak_results.sel(pair=pair).peak_frequencies
        peak_freq_err = peak_results.sel(pair=pair).peak_frequency_errors
        pair_type = "self" if target_qubit_name == aggressor_qubit_name else "cross"

        try:
            slope, intercept, inlier_mask, slope_err = fit_linear(
                peak_freq.flux_bias, peak_freq, peak_freq_err
            )
            measured_slopes[(target_qubit_name, aggressor_qubit_name)] = dict(
                linear_fit_slope=float(slope),
                linear_fit_slope_err=float(slope_err),
                linear_fit_intercept=float(intercept),
                linear_fit_inlier_mask=inlier_mask.tolist(),
                success=True,
                pair_type=pair_type,
            )
        except Exception as exc:
            logging.warning(f"Linear fit failed for {target_qubit_name} vs. {aggressor_qubit_name}: {exc}")
            measured_slopes[(target_qubit_name, aggressor_qubit_name)] = dict(
                linear_fit_slope=float("nan"),
                linear_fit_slope_err=float("nan"),
                linear_fit_intercept=float("nan"),
                linear_fit_inlier_mask=[],
                success=False,
                pair_type=pair_type,
            )

    for target_qubit_name in fit_results:
        target_qubit = node.machine.qubits[target_qubit_name]
        self_slope_model = _get_self_slope_model(target_qubit, node)
        denominator_err = 0.0

        if node.parameters.measure_self:
            self_result = measured_slopes.get((target_qubit_name, target_qubit_name))
            if self_result is None or not self_result["success"]:
                self_slope = np.nan
                self_slope_ratio = np.nan
            else:
                self_slope = self_result["linear_fit_slope"]
                denominator_err = self_result["linear_fit_slope_err"]
                self_slope_ratio = (
                    self_slope / self_slope_model if self_slope_model not in (0, np.nan) else np.nan
                )
                _log_self_slope_check(target_qubit_name, self_slope, self_slope_model, node)
            denominator = self_slope
        else:
            denominator = self_slope_model
            self_slope = np.nan
            self_slope_ratio = np.nan

        flux_detuning_in_v = float(get_flux_detuning_in_v(node.parameters, target_qubit))
        fit_results[target_qubit_name]["_self_calibration"] = dict(
            self_slope_measured=float(self_slope) if np.isfinite(self_slope) else float("nan"),
            self_slope_model=float(self_slope_model) if np.isfinite(self_slope_model) else float("nan"),
            self_slope_ratio=float(self_slope_ratio) if np.isfinite(self_slope_ratio) else float("nan"),
            success=bool(np.isfinite(denominator) and denominator != 0),
            flux_detuning_mode=node.parameters.flux_detuning_mode,
            flux_detuning_in_v=flux_detuning_in_v,
        )

        for (target_key, aggressor_key), result in measured_slopes.items():
            if target_key != target_qubit_name:
                continue

            entry = {k: v for k, v in result.items()}
            if result["pair_type"] == "self":
                entry["crosstalk_coefficient"] = float("nan")
                entry["crosstalk_coefficient_uncertainty"] = float("nan")
                fit_results[target_qubit_name][aggressor_key] = entry
                continue

            if not result["success"] or not np.isfinite(denominator) or denominator == 0:
                entry["crosstalk_coefficient"] = float("nan")
                entry["crosstalk_coefficient_uncertainty"] = float("nan")
                entry["success"] = False
            else:
                coefficient, uncertainty = calculate_crosstalk(
                    result["linear_fit_slope"],
                    result["linear_fit_slope_err"],
                    denominator,
                    denominator_err,
                )
                entry["crosstalk_coefficient"] = float(coefficient)
                entry["crosstalk_coefficient_uncertainty"] = (
                    float(uncertainty) if np.isfinite(uncertainty) else float("nan")
                )
            fit_results[target_qubit_name][aggressor_key] = entry

    return fit_results


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict]:
    """Fit Lorentzian peaks and linear crosstalk relationships."""
    node.results["peak_results"] = peak_results = fit_lorentzian_peaks(ds, node)
    fit_results = fit_linear_crosstalk(peak_results, node)
    return ds, fit_results


def get_target_qubit_slope_at_flux_detuning(target_qubit, flux_detuning_in_v: float):
    """Return the target qubit df/dphi slope in Hz/V at a given flux detuning."""
    return -2 * target_qubit.freq_vs_flux_01_quad_term * flux_detuning_in_v
