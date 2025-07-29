import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from typing import Tuple, Dict, Optional
import logging
from dataclasses import dataclass
from scipy.optimize import curve_fit
from quam_libs.lib.fit import fit_oscillation, oscillation_decay_exp


@dataclass
class FitParameters:
    success: bool = False
    J: float = 0.0
    detuning: float = 0.0
    optimal_amplitude: float = 0.0
    optimal_length: int = 0
    zero_padding: int = 0


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    def abs_amp(qp, amp):
        return amp * node.namespace["pulse_amplitudes"][qp.name]

    def detuning(qp, amp):
        return -(amp * node.namespace["pulse_amplitudes"][qp.name])**2 * qp.qubit_control.freq_vs_flux_01_quad_term

    ds = ds.assign_coords(
        {"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp.data) for qp in node.namespace["qubit_pairs"]]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qubit", "amp"], np.array([detuning(qp, ds.amp) for qp in node.namespace["qubit_pairs"]]))}
    )
    return ds


def rabi_chevron_model(ft, J, f0, a, offset,tau):
    f,t = ft
    J = J
    w = f
    w0 = f0
    g = offset+a * np.sin(2*np.pi*np.sqrt(J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0)))
    return g.ravel()


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode, pulse_amplitudes, qubit_pairs) -> Tuple[xr.Dataset, Dict[str, FitParameters]]:

    ds = ds.assign_coords(
        {"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp.data) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qubit", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))}
    )

    fit_results = {}
    for qp in node.namespace["qubit_pairs"]:
        ds_qp = ds.sel(qubit=qp.name)
        J, f0, a, offset, tau = fit_rabi_chevron(ds_qp)

        amp_guess = ds_qp.state_target.max("time") - ds_qp.state_target.min("time")
        flux_amp_idx = int(amp_guess.argmax())
        flux_amp = float(ds_qp.amp_full[flux_amp_idx])
        fit_data = fit_oscillation_decay_exp(
            ds_qp.state_control.isel(amp=flux_amp_idx), "time")
        flux_time = int(1/fit_data.sel(fit_vals='f'))
        amplitudes = {}
        lengths = {}
        zero_paddings = {}
        detunings = {}
        amplitudes[qp.name] = flux_amp
        detunings[qp.name] = -flux_amp ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term
        lengths[qp.name] = flux_time - flux_time % 4 + 4
        zero_paddings[qp.name] = lengths[qp.name] - flux_time
        fit_results[qp.name] = ds_qp.assign({'fitted': oscillation_decay_exp(ds_qp.time,
                                                                           fit_data.sel(
                                                                               fit_vals="a"),
                                                                           fit_data.sel(
                                                                               fit_vals="f"),
                                                                           fit_data.sel(
                                                                               fit_vals="phi"),
                                                                           fit_data.sel(
                                                                               fit_vals="offset"),
                                                                           fit_data.sel(fit_vals="decay"))})

        fit_results[qp.name] = FitParameters(
            success=True,
            J=J,
            detuning=f0,
            optimal_amplitude=flux_amp,
            optimal_length=flux_time,
            zero_padding=flux_time - (flux_time % 4) + 4 - flux_time
        )
    return ds, fit_results, amplitudes, lengths, zero_paddings, detunings


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    if hasattr(ds_qp, "state_target"):
        da_target = ds_qp.state_target
    else:
        da_target = ds_qp.I_target
    try:
        exp_data = da_target.values
        detuning = da_target.detuning
        time = da_target.time*1e-9
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
    fit_data = fit_oscillation_decay_exp(signal, "time")
    fit_time = 1 / fit_data.sel(fit_vals='f')  # Oscillation frequency (example only)
    return int(fit_time)


def fit_oscillation_decay_exp(signal, x_key):
    fit_data = fit_oscillation(
        signal,
        x_data=signal[x_key],
        decay_model_func=oscillation_decay_exp
    )
    return fit_data

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


