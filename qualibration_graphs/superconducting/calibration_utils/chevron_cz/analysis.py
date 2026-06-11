import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from qualibration_libs.analysis.fitting import fit_oscillation_decay_exp
from scipy.optimize import curve_fit


def rabi_chevron_model(ft, J, f0, a, offset):
    """Rabi-Chevron population model for the CZ |11⟩↔|02⟩ two-level system.

    Parameters:
    -----------
    ft : tuple
        Tuple ``(f, t)`` where ``f`` is the detuning array (Hz) and ``t`` is the
        pulse duration array (s).
    J : float
        Coupling strength (Hz). The gate time at resonance is ``1/(2J)``.
    f0 : float
        Resonance detuning (Hz) at which the chevron is centred.
    a : float
        Amplitude scaling factor.
    offset : float
        Vertical offset of the population signal.

    Returns:
    --------
    np.ndarray
        Ravelled 1-D array of predicted population values.
    """
    f, t = ft
    det = (f - f0) / 2
    # g = offset+a * np.sin(2*np.pi*np.sqrt(J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0)))
    g = offset + a * (J**2) / (J**2 + det**2) * np.sin(2 * np.pi * np.sqrt(J**2 + det**2) * t) ** 2

    return g.ravel()


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    """Fit the Rabi-Chevron data for one qubit pair using ``rabi_chevron_model``.

    Parameters:
    -----------
    ds_qp : xr.Dataset
        Single-pair dataset containing ``state_stationary`` or ``I_stationary``,
        with ``detuning`` and ``time`` coordinates.
    init_length : float
        Initial guess for the gate length (ns), used to seed ``J = 1e9/init_length``.
    init_detuning : array-like
        Initial guess for the resonance detuning ``f0`` (Hz).

    Returns:
    --------
    tuple[float, float, float, float]
        ``(J, f0, a, offset)`` fit parameters, or ``(nan, nan, nan, nan)`` on failure.
    """
    if hasattr(ds_qp, "state_stationary"):
        data = ds_qp.state_stationary
    else:
        data = ds_qp.I_stationary

    try:
        da_target = data
        exp_data = da_target.values
        detuning = da_target.detuning[0]
        time = da_target.time * 1e-9
        t, f = np.meshgrid(time, detuning)
        initial_guess = (1e9 / init_length, init_detuning[0], -1, 1.0)
        fdata = np.vstack((f.ravel(), t.ravel()))
        tdata = exp_data.ravel()
        popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
        J = popt[0]
        f0 = popt[1]
        a = popt[2]
        offset = popt[3]
        return J, f0, a, offset
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")


@dataclass
class FitParameters:
    """Fit results for a single qubit pair from the CZ chevron calibration.

    Attributes:
    -----------
    success : bool
        True if the fit converged and the extracted parameters are physically valid.
    J : float
        Fitted coupling strength in Hz. The CZ gate time at resonance is ``1/(2J)``.
    f0 : float
        Fitted resonance detuning in Hz (centre of the chevron).
    cz_len : int
        Optimal CZ gate duration in nanoseconds, derived from ``1/(2J) * 1e9``.
    cz_amp : float
        Optimal CZ flux pulse amplitude in volts, derived from ``sqrt(-f0/quad_term)``.
    """

    success: bool
    J: float
    f0: float
    cz_len: int
    cz_amp: float


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Log the CZ calibration fit results for all qubit pairs.

    Parameters:
    -----------
    fit_results : dict
        Dictionary mapping qubit pair names to ``FitParameters`` instances or plain
        dicts (as returned by ``dataclasses.asdict``).
    log_callable : callable, optional
        Logging function. Defaults to the module logger at INFO level.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qp_name, fit_result in fit_results.items():
        # Support both dataclass instances and plain dictionaries (after asdict)
        def _get(field, default=np.nan):
            if hasattr(fit_result, field):
                return getattr(fit_result, field)
            if isinstance(fit_result, dict):
                return fit_result.get(field, default)
            return default

        success = bool(_get("success", False))
        cz_len_val = _get("cz_len", 0)
        cz_amp_val = _get("cz_amp", np.nan)

        s_qubit = f"Results for qubit pair {qp_name}: "
        s_qubit += "SUCCESS!\n" if success else "FAIL!\n"

        if isinstance(cz_len_val, (int, float)) and cz_len_val not in (None, np.nan):
            cz_len_str = f"\tOptimal CZ duration: {int(cz_len_val)} ns"
        else:
            cz_len_str = "\tOptimal CZ duration: N/A"

        if isinstance(cz_amp_val, (int, float)) and not np.isnan(cz_amp_val):
            cz_amp_str = f"\tOptimal CZ amplitude: {cz_amp_val:.6f} V"
        else:
            cz_amp_str = "\tOptimal CZ amplitude: N/A"

        log_callable(s_qubit + cz_len_str + "\n" + cz_amp_str)


def fit_chevron_cz(ds, dim):
    """Fit the Rabi-Chevron pattern for every qubit pair using a groupby-apply loop.

    For each pair the routine:
    1. Finds the amplitude with the largest oscillation contrast as an initial guess.
    2. Fits a decaying oscillation to estimate the initial gate time.
    3. Calls ``fit_rabi_chevron`` with these seeds to obtain ``(J, f0, a, offset)``.

    Parameters:
    -----------
    ds : xr.Dataset
        Processed dataset with ``qubit_pair``, ``amplitude``, ``time``,
        ``detuning``, ``amp_full``, and ``quad_term_moving`` coordinates.
    dim : str
        Dimension name to group by (``"qubit_pair"``).

    Returns:
    --------
    xr.Dataset
        Dataset with a ``fit_vals`` dimension containing ``[J, f0, a, offset]``
        per qubit pair.
    """
    def fit_routine(ds_qp):
        """Fit one qubit pair's chevron and return ``[J, f0, a, offset]`` as a DataArray."""
        if hasattr(ds_qp, "state_stationary"):
            data = ds_qp.state_stationary
        else:
            data = ds_qp.I_stationary
        try:
            # ds_qp is a Dataset for a single qubit_pair
            amp_guess = data.max("time") - data.min("time")
            flux_amp_idx = int(amp_guess.argmax())
            flux_amp = float(ds_qp.amp_full[0][flux_amp_idx])

            # Try the preliminary oscillation fit
            try:
                fit_data = fit_oscillation_decay_exp(data.isel(amplitude=flux_amp_idx), "time")
                flux_time = int(1 / fit_data.sel(fit_vals="f"))
            except Exception:
                # If preliminary fit fails, use a default time
                flux_time = 50  # default 50 ns

            amplitudes = flux_amp
            detunings = -(flux_amp**2) * ds_qp.quad_term_moving
            lengths = flux_time - flux_time % 4 + 4

            t = ds_qp.time * 1e-9
            f = ds_qp.detuning
            t, f = np.meshgrid(t, f)
            J, f0, a, offset = fit_rabi_chevron(ds_qp, lengths * 2, detunings.values)

            # Check if fitting produced valid results
            if np.isnan(J) or np.isnan(f0):
                # Return default/invalid values
                return xr.DataArray([float("nan"), float("nan"), float("nan"), float("nan")], dims=["fit_vals"])

            detunings = f0
            amplitudes = np.sqrt(-detunings / ds_qp.quad_term_moving)
            flux_time = int(1 / (2 * J) * 1e9)
            lengths = flux_time - flux_time % 4 + 4

            # Return as DataArray for stacking
            return xr.DataArray([J, f0, a, offset], dims=["fit_vals"])
        except Exception:
            return xr.DataArray([float("nan"), float("nan"), float("nan"), float("nan")], dims=["fit_vals"])

    # Use groupby-apply pattern
    fit_res = ds.groupby("qubit_pair").apply(fit_routine)
    fit_res = fit_res.assign_coords(fit_vals=("fit_vals", ["J", "f0", "a", "offset"]))
    return fit_res


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Add physical coordinates to the raw dataset.

    Computes and assigns:
    - ``detuning`` — flux-induced frequency shift of the moving qubit (Hz).
    - ``amp_full`` — absolute flux pulse amplitude in volts.
    - ``quad_term_moving`` — ``freq_vs_flux_01_quad_term`` of the moving qubit, stored
      per pair for use in the fitting step.

    Parameters:
    -----------
    ds : xr.Dataset
        Raw dataset as returned by the QUA data fetcher.
    node : QualibrationNode
        Calibration node providing ``pulse_amplitudes`` and qubit pair objects.

    Returns:
    --------
    xr.Dataset
        Dataset with the additional coordinates described above.
    """
    def detuning(qp, amp):
        return -((amp * node.namespace["pulse_amplitudes"][qp.name]) ** 2) * node.namespace["qubit_roles_map"][qp.name].moving.freq_vs_flux_01_quad_term

    def abs_amp(qp, amp):
        return amp * node.namespace["pulse_amplitudes"][qp.name]

    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]

    ds = ds.assign_coords(
        {"detuning": (["qubit_pair", "amplitude"], np.array([detuning(qp, ds.amplitude) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"amp_full": (["qubit_pair", "amplitude"], np.array([abs_amp(qp, ds.amplitude) for qp in qubit_pairs]))},
    )

    ds = ds.assign_coords(
        {
            "quad_term_moving": (
                ["qubit_pair"],
                np.array([node.namespace["qubit_roles_map"][qp.name].moving.freq_vs_flux_01_quad_term for qp in qubit_pairs]),
            )
        }
    )

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the Rabi-Chevron pattern for each qubit pair and extract CZ gate parameters.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The calibration node containing parameters and qubit pairs.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Dataset containing the fit results and dictionary of fit parameters for each qubit pair.
    """

    ds_fit_res = fit_chevron_cz(ds, "qubit_pair")

    ds_fit = xr.merge([ds, ds_fit_res.rename("fit")])

    # Extract the relevant fitted parameters
    fit_data, fit_results = _extract_relevant_fit_parameters(ds_fit, node)
    return fit_data, fit_results


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Derive ``cz_len``, ``cz_amp``, and success flags from raw fit values.

    For each qubit pair, converts the fitted ``J`` and ``f0`` into physically
    meaningful gate parameters and validates them against the swept ranges.
    Assigns ``cz_len`` and ``cz_amp`` as coordinates on the dataset.

    Parameters:
    -----------
    fit : xr.Dataset
        Dataset containing a ``fit`` DataArray with ``fit_vals`` dimension
        (``J``, ``f0``, ``a``, ``offset``) and ``quad_term_moving``, ``amp_full``
        coordinates.
    node : QualibrationNode
        Unused; retained for API consistency with other ``_extract_*`` helpers.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        Updated dataset with ``cz_len`` and ``cz_amp`` coordinates, and a
        dictionary of ``FitParameters`` keyed by qubit pair name.
    """

    # Populate the FitParameters class with fitted values
    fit_results = {}
    for qp in fit.qubit_pair.values:
        try:
            J_val = fit.fit.sel(qubit_pair=qp, fit_vals="J").values.item()
            f0_val = fit.fit.sel(qubit_pair=qp, fit_vals="f0").values.item()

            # Check if values are valid
            if np.isnan(J_val) or np.isnan(f0_val) or J_val <= 0:
                success = False
                cz_len_val = 0
                cz_amp_val = float("nan")
            else:
                try:
                    cz_len_val = int(1 / (2 * J_val) * 1e9)
                    cz_amp_val = np.sqrt(-f0_val / fit.quad_term_moving.sel(qubit_pair=qp).values.item())

                    # Determine success based on reasonable parameter ranges
                    amp_min = fit.amp_full.sel(qubit_pair=qp).min().item()
                    amp_max = fit.amp_full.sel(qubit_pair=qp).max().item()

                    is_length_valid = 10 < cz_len_val < 1000
                    is_amp_valid = amp_min <= cz_amp_val <= amp_max
                    is_not_nan = not np.isnan(cz_amp_val)
                    success = bool(is_length_valid and is_amp_valid and is_not_nan)
                except Exception:
                    # If parameter calculation fails, mark as failed
                    success = False
                    cz_len_val = 0
                    cz_amp_val = float("nan")

            fit_results[qp] = FitParameters(
                success=success,
                J=J_val,
                f0=f0_val,
                cz_len=cz_len_val,
                cz_amp=cz_amp_val,
            )
        except Exception:
            fit_results[qp] = FitParameters(
                success=False,
                J=float("nan"),
                f0=float("nan"),
                cz_len=0,
                cz_amp=float("nan"),
            )

    fit = fit.assign_coords(
        {
            "cz_len": ("qubit_pair", [fit_results[qp].cz_len for qp in fit.qubit_pair.values]),
            "cz_amp": ("qubit_pair", [fit_results[qp].cz_amp for qp in fit.qubit_pair.values]),
        }
    )
    return fit, fit_results
