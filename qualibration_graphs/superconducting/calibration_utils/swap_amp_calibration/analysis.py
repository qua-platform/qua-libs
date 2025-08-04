import numpy as np
import xarray as xr
from typing import Tuple, Dict
import logging
from dataclasses import dataclass, asdict
from qualibrate import QualibrationNode
from qualibration_libs.analysis.fitting import fit_oscillation, oscillation


@dataclass
class FitParameters:
    success: bool = False
    J: float = 0.0
    detuning: float = 0.0
    optimal_amplitude: float = 0.0
    optimal_length: int = 0
    zero_padding: int = 0


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    ds = ds.assign({"res_sum": ds.state_control - ds.state_target})

    amp_full = np.array(
        [
            node.namespace["control_amplitudes_scale"] * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude
            for qp in node.namespace["qubit_pairs"]
        ]
    )
    ds = ds.assign_coords({"amp_full": (["qubit", "amplitude"], amp_full)})

    detunings = np.array(
        [
            -(amp_full[i] ** 2) * qp.qubit_control.freq_vs_flux_01_quad_term
            for i, qp in enumerate(node.namespace["qubit_pairs"])
        ]
    )
    ds = ds.assign_coords({"detuning": (["qubit", "amplitude"], detunings)})

    if node.parameters.use_state_discrimination:
        ds = ds.assign({"data_var_control": ds.state_control, "data_var_target": ds.state_target})
    else:
        ds = ds.assign({"data_var_control": ds.I_control, "data_var_target": ds.I_target})

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:
    ds_fit = {}
    fit_results = {}

    for qp in node.namespace["qubit_pairs"]:
        name = qp.name
        ds_qp = ds.sel(qubit=name)

        try:
            if node.parameters.max_number_pulses_per_sweep == 1:
                fit = fit_oscillation(ds_qp.data_var_control, "amplitude")
                fit_curve = oscillation(
                    ds_qp.amplitude,
                    fit.sel(fit_vals="a"),
                    fit.sel(fit_vals="f"),
                    fit.sel(fit_vals="phi"),
                    fit.sel(fit_vals="offset"),
                )
                ds_qp = ds_qp.assign({"fit_amp_control": fit_curve})

                f_fit = fit.sel(fit_vals="f").item()
                phi_fit = fit.sel(fit_vals="phi").item()
                phi_fit -= np.pi * (phi_fit > np.pi / 2)

                factor = float((np.pi - phi_fit) / (2 * np.pi * f_fit))
                new_pi_amp = qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude * factor
                new_pi_amp = min(new_pi_amp, 0.3)  # Clamp to max 0.3

                fit_results[name] = FitParameters(
                    success=True,
                    optimal_amplitude=new_pi_amp,
                )
            else:
                idx = (ds_qp.data_var_target - ds_qp.data_var_control).mean(dim="N_pi_vec").argmax(dim="amplitude")
                amp_opt = (
                    node.namespace["control_amplitudes_scale"]
                    * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude
                    * float(ds_qp.amplitude[idx].values)
                )

                fit_results[name] = FitParameters(
                    success=True,
                    optimal_amplitude=amp_opt,
                )

        except Exception as e:
            fit_results[name] = FitParameters(success=False)
            print(f"Fit failed for {name}: {e}")

        ds_fit[name] = ds_qp

    return xr.concat(ds_fit.values(), dim="qubit"), {k: asdict(v) for k, v in fit_results.items()}


def log_fitted_results(fit_results: Dict[str, Dict], log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for name, result in fit_results.items():
        log_callable(f"{name}: SUCCESS={result['success']}, " f"amp={result['optimal_amplitude']:.3e}")
