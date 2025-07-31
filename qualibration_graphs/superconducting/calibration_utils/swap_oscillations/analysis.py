import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from scipy.optimize import curve_fit
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp, oscillation_decay_exp, fit_oscillation


@dataclass
class FitParameters:
    success: bool = False
    J: float = 0.0
    detuning: float = 0.0
    optimal_amplitude: float = 0.0
    optimal_length: int = 0
    zero_padding: int = 0


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    ds = ds.assign_coords(idle_time = ds.idle_time * 4)
    ds = ds.assign({"res_sum": ds.state_control - ds.state_target})

    amp_full = np.array([
        node.namespace["control_amplitudes_scale"] * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude
        for qp in node.namespace["qubit_pairs"]
    ])
    ds = ds.assign_coords({"amp_full": (["qubit", "amp"], amp_full)})

    detunings = np.array([
        -(amp_full[i] ** 2) * qp.qubit_control.freq_vs_flux_01_quad_term
        for i, qp in enumerate(node.namespace["qubit_pairs"])
    ])
    ds = ds.assign_coords({"detuning": (["qubit", "amp"], detunings)})
    return ds


def rabi_chevron_model(ft, J, f0, a, offset,tau):
    f,t = ft
    J = J
    w = f
    w0 = f0
    g = offset+a * np.sin(2*np.pi*np.sqrt(J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0)))
    return g.ravel()


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    fit_results = {}
    ds_fit = {}
    for qp in node.namespace["qubit_pairs"]:
        ds_qp = ds.sel(qubit=qp.name)
        try:
            amp_guess = ds_qp.state_target.max("idle_time") - ds_qp.state_target.min("idle_time")
            flux_amp_idx = int(amp_guess.argmax())
            flux_amp = float(ds_qp.amp_full[flux_amp_idx])
            fit_data = fit_oscillation_decay_exp(ds_qp.state_control.isel(amp=flux_amp_idx), "idle_time")
            flux_time = int(1 / fit_data.sel(fit_vals="f"))


            J, f0, *_ = fit_rabi_chevron(ds_qp, flux_time, -flux_amp ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term)
            amp_from_fit = np.sqrt(-f0 / qp.qubit_control.freq_vs_flux_01_quad_term)
            flux_time = int(1 / (2 * J) * 1e9)
            zero_padding = flux_time - flux_time % 4 + 4 - flux_time

            ds_fit[qp.name] = ds_qp.assign({
                "fitted": oscillation_decay_exp(
                    ds_qp.idle_time,
                    fit_data.sel(fit_vals="a"),
                    fit_data.sel(fit_vals="f"),
                    fit_data.sel(fit_vals="phi"),
                    fit_data.sel(fit_vals="offset"),
                    fit_data.sel(fit_vals="decay")
                )
            })

            fit_results[qp.name] = FitParameters(
                success=True,
                J=J,
                detuning=f0,
                optimal_amplitude=amp_from_fit,
                optimal_length=flux_time,
                zero_padding=zero_padding
            )
        except Exception as e:
            fit_results[qp.name] = FitParameters(success=False)
            print(f"Fit failed for {qp.name}: {e}")

    ds_fit = xr.concat([ds_fit[k] for k in ds_fit], dim="qubit")
    return ds_fit, fit_results


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    if hasattr(ds_qp, "state_target"):
        da_target = ds_qp.state_target
    else:
        da_target = ds_qp.I_target
    try:
        exp_data = da_target.values
        detuning = da_target.detuning
        time = da_target.idle_time *1e-9
        t,f  = np.meshgrid(time,detuning)
        initial_guess = (1e9/init_length/2,
                init_detuning,
                -1,
                1.0,
                100e-9)
        fdata = np.vstack((f.ravel(),t.ravel()))
        tdata = exp_data.ravel()
        popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
        J = popt[0]
        f0 = popt[1]
        a = popt[2]
        offset = popt[3]
        tau = popt[4]

        return J, f0, a, offset, tau
    except Exception as e:
        # Return NaN values if fitting fails
        return float("nan"), float("nan"), float("nan"), float("nan")


def determine_flux_time(flux_amp_idx, ds_qp):
    signal = ds_qp.state_control.isel(amp=flux_amp_idx)
    fit_data = fit_oscillation_decay_exp(signal, "idle_time")
    fit_time = 1 / fit_data.sel(fit_vals='f')  # Oscillation frequency (example only)
    return int(fit_time)


def abs_amp(qp, amp, pulse_amplitudes):
    return amp * pulse_amplitudes[qp.name]


def detuning(qp, amp, pulse_amplitudes):
    return -(amp * pulse_amplitudes[qp.name]) ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term


def log_fitted_results(fit_results: Dict[str, FitParameters], log_callable=None):
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for qubit, results in fit_results.items():
        log_callable(
            f"{qubit}: SUCCESS={results.success}, J={results.J:.3e}, "
            f"f0={results.detuning:.3e}, amp={results.optimal_amplitude:.3e}, "
            f"len={results.optimal_length}, pad={results.zero_padding}"
        )


