"""Data analysis for Ramsey versus flux calibration."""

import logging
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qualibrate import QualibrationNode
from quam_config.instrument_limits import instrument_limits
from qualibration_libs.analysis import fit_oscillation_decay_exp, oscillation_decay_exp, peaks_dips
from qualibration_libs.data import add_amplitude_and_phase, convert_IQ_to_V


@dataclass
class FitParameters:
    """Stores the relevant Ramsey vs flux experiment fit parameters for a single qubit"""

    success: bool
    quad_term: float
    flux_offset: float
    freq_offset: float
    flux_offset: float
    t2_star: np.ndarray


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all qubits from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all qubits.
    logger : logging.Logger, optional
        Logger for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    pass


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """Process raw dataset (placeholder)."""
    return ds


def unfold_aliased_frequencies(frequency: xr.DataArray, f_nyquist: float) -> Tuple[xr.DataArray, xr.DataArray]:
    """Unfold aliased frequencies using Nyquist zone tracking (extended zone scheme).

    Analogous to Brillouin zone unfolding in solid-state physics: the fitted
    oscillation frequency folds into [0, f_nyquist] due to discrete time sampling,
    and this function recovers the true frequency by tracking which Nyquist zone
    each flux point belongs to.

    Assumes the true frequency is approximately parabolic, centered near
    flux_bias=0 (first Nyquist zone), generally increasing with |flux_bias|.
    The curve need not be globally monotonic — the algorithm only triggers a
    zone transition when the measured frequency has demonstrably reached the
    zone boundary (f_nyquist for even zones, 0 for odd zones).

    The unfolding formula per zone *n*:
      - n even: f_real = n * f_nyq + f_measured
      - n odd:  f_real = (n+1) * f_nyq - f_measured

    Parameters
    ----------
    frequency : xr.DataArray
        Fitted oscillation frequencies, dims (qubit, flux_bias), in GHz.
    f_nyquist : float
        Nyquist frequency in GHz: 1 / (2 * wait_time_step_in_ns).

    Returns
    -------
    unfolded : xr.DataArray
        Unfolded frequencies (GHz), same shape as input.
    zones : xr.DataArray
        Nyquist zone index per point, same shape as input.
    """
    flux = frequency.flux_bias.values
    center_idx = int(np.argmin(np.abs(flux)))

    unfolded_data = np.full(frequency.shape, np.nan)
    zone_data = np.zeros(frequency.shape)

    for qi, q in enumerate(frequency.qubit.values):
        freq = np.abs(frequency.isel(qubit=qi).values)
        result = np.full(len(flux), np.nan)
        zone_arr = np.zeros(len(flux))

        if not np.isnan(freq[center_idx]):
            result[center_idx] = freq[center_idx]

        for step, rng in [
            (+1, range(center_idx + 1, len(flux))),
            (-1, range(center_idx - 1, -1, -1)),
        ]:
            zone = 0
            # Track the extremum of the measured frequency since the last
            # zone transition.  A real zone crossing requires the measured
            # frequency to have reached the boundary: ≈ f_nyquist for even
            # zones (upper boundary) or ≈ 0 for odd zones (lower boundary).
            extremum = freq[center_idx] if not np.isnan(freq[center_idx]) else 0.0
            for i in rng:
                f_m = freq[i]
                if np.isnan(f_m):
                    result[i] = np.nan
                    zone_arr[i] = zone
                    continue

                if not int(zone) % 2:
                    extremum = max(extremum, f_m)
                else:
                    extremum = min(extremum, f_m)

                f_candidate = _unfold_single(f_m, zone, f_nyquist)

                prev = _find_prev_valid(result, i, -step)
                if prev is not None and f_candidate < prev and _boundary_was_reached(extremum, zone, f_nyquist):
                    zone += 1
                    f_candidate = _unfold_single(f_m, zone, f_nyquist)
                    extremum = f_m

                result[i] = f_candidate
                zone_arr[i] = zone

        unfolded_data[qi] = result
        zone_data[qi] = zone_arr

    unfolded = xr.DataArray(
        unfolded_data,
        dims=frequency.dims,
        coords=frequency.coords,
        attrs={"long_name": "unfolded frequency", "units": "GHz"},
    )
    zones = xr.DataArray(
        zone_data,
        dims=frequency.dims,
        coords=frequency.coords,
        attrs={"long_name": "Nyquist zone"},
    )
    return unfolded, zones


def _unfold_single(f_measured: float, zone: int, f_nyquist: float) -> float:
    """Compute unfolded frequency from measured frequency and zone index."""
    zone = int(zone)
    if not zone % 2:
        return zone * f_nyquist + f_measured
    return (zone + 1) * f_nyquist - f_measured


def _boundary_was_reached(
    extremum_since_zone_change: float,
    zone: int,
    f_nyquist: float,
    threshold: float = 0.9,
) -> bool:
    """Check whether the measured frequency reached the zone boundary since the
    last zone transition.

    At a genuine aliasing fold the measured frequency must have touched the
    boundary — f_nyquist for even zones (upper) or 0 for odd zones (lower).
    If the extremum since the last zone change never got close to the boundary,
    the monotonicity violation is caused by the real frequency curve turning
    over, not by aliasing.

    Parameters
    ----------
    extremum_since_zone_change : float
        Running max (even zone) or running min (odd zone) of the measured
        frequency since the last zone transition.
    zone : int
        Current Nyquist zone index.
    f_nyquist : float
        Nyquist frequency (GHz).
    threshold : float
        Fraction of f_nyquist the extremum must exceed (even zone) or stay
        below (odd zone).  Default 0.9.
    """
    zone = int(zone)
    if not zone % 2:
        return extremum_since_zone_change > threshold * f_nyquist
    return extremum_since_zone_change < (1 - threshold) * f_nyquist


def _find_prev_valid(arr: np.ndarray, idx: int, direction: int):
    """Find the nearest non-NaN value from *idx* in the given direction."""
    i = idx + direction
    while 0 <= i < len(arr):
        if not np.isnan(arr[i]):
            return arr[i]
        i += direction
    return None


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Fit the qubit frequency and FWHM for each qubit in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.

    Returns:
    --------
    xr.Dataset
        Dataset containing the fit results.
    """
    # # TODO: explain the data analysis
    fit_data = fit_oscillation_decay_exp(ds.state, "idle_times")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    fitted = oscillation_decay_exp(
        ds.state.idle_times,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )

    frequency = fit_data.sel(fit_vals="f")

    # Unfold aliased frequencies (Nyquist zone unfolding / extended zone scheme)
    f_nyquist = 1.0 / (2.0 * node.parameters.wait_time_step_in_ns)  # GHz
    frequency_unfolded, nyquist_zones = unfold_aliased_frequencies(frequency, f_nyquist)

    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    frequency = frequency_unfolded.where(frequency_unfolded > 0, drop=True)

    fitvals = frequency.polyfit(dim="flux_bias", deg=2)
    flux = frequency.flux_bias

    a = {}
    flux_offset = {}
    freq_offset = {}
    t2_star = {}

    qubits = ds.qubit.values

    for q in qubits:
        a[q] = float(-1e6 * fitvals.sel(qubit=q, degree=2).polyfit_coefficients.values)
        flux_offset[q] = float(
            (
                -0.5
                * fitvals.sel(qubit=q, degree=1).polyfit_coefficients
                / fitvals.sel(qubit=q, degree=2).polyfit_coefficients
            ).values
        )
        freq_offset[q] = 1e6 * (
            flux_offset[q] ** 2 * float(fitvals.sel(qubit=q, degree=2).polyfit_coefficients.values)
            + flux_offset[q] * float(fitvals.sel(qubit=q, degree=1).polyfit_coefficients.values)
            + float(fitvals.sel(qubit=q, degree=0).polyfit_coefficients.values)
        )
        t2_star[q] = tau.sel(qubit=q).values

    ds_fit = ds.merge(fit_data.rename("fit_results"))

    # Add a, flux_offset, and freq_offset as data variables in the dataset
    ds_fit["quad_term"] = xr.DataArray([a[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["flux_offset"] = xr.DataArray([flux_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["freq_offset"] = xr.DataArray([freq_offset[q] for q in qubits], dims=["qubit"], coords={"qubit": qubits})
    ds_fit["artifitial_detuning"] = xr.DataArray(
        node.parameters.frequency_detuning_in_mhz, dims=["qubit"], coords={"qubit": qubits}
    )
    ds_fit["unfolded_frequency"] = frequency_unfolded
    ds_fit["nyquist_zones"] = nyquist_zones
    ds_fit["f_nyquist"] = f_nyquist

    fit_results = {
        q: FitParameters(
            success=True,
            quad_term=a[q],
            flux_offset=flux_offset[q],
            freq_offset=freq_offset[q],
            t2_star=t2_star[q],
        )
        for q in ds_fit.qubit.values
    }

    return ds_fit, fit_results


def add_qubit_freq_vs_flux(
    ds_fit,
    rf_frequencies_hz: dict,
    artificial_detuning_mhz: float,
):
    """Compute absolute qubit RF frequency vs flux and add to ds_fit.

    f_qubit(Phi) [GHz] = RF_frequency/1e9 + artificial_detuning*1e-3 - unfolded_frequency

    Qubits absent from rf_frequencies_hz (fitting failed) receive NaN rows.
    """
    import numpy as np
    import xarray as xr

    artificial_ghz = artificial_detuning_mhz * 1e-3
    qubits = ds_fit.qubit.values
    n_flux = len(ds_fit.flux_bias)
    f_qubit_data = np.full((len(qubits), n_flux), np.nan)
    for qi, q in enumerate(qubits):
        if q not in rf_frequencies_hz:
            continue
        f_qubit_data[qi] = rf_frequencies_hz[q] / 1e9 + artificial_ghz - ds_fit.unfolded_frequency.sel(qubit=q).values
    ds_fit["f_qubit_vs_flux"] = xr.DataArray(
        f_qubit_data,
        dims=["qubit", "flux_bias"],
        coords={"qubit": qubits, "flux_bias": ds_fit.flux_bias},
        attrs={"long_name": "Qubit RF frequency", "units": "GHz"},
    )
    return ds_fit


def _extract_relevant_fit_parameters(fit: xr.Dataset, node: QualibrationNode):
    """Add metadata to the dataset and fit results."""
    pass
