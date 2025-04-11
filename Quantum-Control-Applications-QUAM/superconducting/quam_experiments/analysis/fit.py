from matplotlib import pyplot as plt
import qiskit_experiments.curve_analysis as ca
from scipy.optimize import curve_fit
import numpy as np
import xarray as xr
from lmfit import Model, Parameter
from scipy.signal import find_peaks, peak_widths
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
from scipy.fft import fft


def lorentzian_peak(x, amplitude, center, width, offset):
    """
    Computes the Lorentzian peak function.

    Parameters
    ----------
    x : array-like
        The input values at which to evaluate the Lorentzian function.
    amplitude : float
        The amplitude of the Lorentzian peak.
    center : float
        The center position of the Lorentzian peak.
    width : float
        The full width at half maximum (FWHM) of the Lorentzian peak.
    offset : float
        The offset value added to the Lorentzian function.

    Returns
    -------
    array-like
        The evaluated Lorentzian function at the input values `x`.

    Notes
    -----
    - The Lorentzian function is defined as:
      L(x) = offset + amplitude * (1 / (1 + ((x - center) / width)^2))
    - This function is commonly used to model resonance peaks in qubit spectroscopy.
    """
    return offset + amplitude * (1 / (1 + ((x - center) / width) ** 2))


def lorentzian_deep(x, amplitude, center, width, offset):
    """
    Computes the Lorentzian function.

    Parameters
    ----------
    x : array-like
        The input values at which to evaluate the Lorentzian function.
    amplitude : float
        The amplitude of the Lorentzian peak.
    center : float
        The center position of the Lorentzian peak.
    width : float
        The full width at half maximum (FWHM) of the Lorentzian peak.
    offset : float
        The offset value added to the Lorentzian function.

    Returns
    -------
    array-like
        The evaluated Lorentzian function at the input values `x`.

    Notes
    -----
    - The Lorentzian function is defined as:
      L(x) = offset - (amplitude * width^2) / (width^2 + (x - center)^2)
    - This function is commonly used to model resonance peaks in spectroscopy and other fields.
    """
    return offset - amplitude * width**2 / (width**2 + (x - center) ** 2)


def peaks_dips(da, dim, prominence_factor=5, number=1, remove_baseline=True) -> xr.Dataset:
    """
    Searches in a data array along the specified dimension for the most prominent peak or dip, and returns a xarray dataset with its location, width, and amplitude, along with a smooth baseline from which the peak emerges.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray.
    dim : str
        The dimension along which to perform the fit.
    prominence_factor : float
        How prominent the peak must be compared with noise, as defined by the standard deviation.
    number : int
        Determines which peak the function returns. 1 is the most prominent peak, 2 is the second most prominent, etc.
    remove_baseline : bool, optional
        If True, the function will remove the baseline from the data before finding the peak (default is False).

    Returns
    -------
    xr.Dataset
        A dataset with the following values:
        - 'amp': Peak amplitude above the base.
        - 'position': Peak location along 'dim'.
        - 'width': Peak full width at half maximum (FWHM).
        - 'baseline': A vector whose dimension is the same as 'dim'. It is the baseline from which the peak is found. This is important for fitting resonator spectroscopy measurements.

    Notes
    -----
    - The function identifies the most prominent peak or dip in the data array along the specified dimension.
    - The baseline is smoothed and subtracted if `remove_baseline` is True.
    """

    def _baseline_als(y, lam, p, niter=10):
        L = len(y)
        D = sparse.csc_matrix(np.diff(np.eye(L), 2))
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

    def _index_of_largest_peak(arr, prominence):
        peaks = find_peaks(arr.copy(), prominence=prominence)
        if len(peaks[0]) > 0:
            # finding the largest peak and it's width
            prom_peak_index = 1.0 * peaks[0][np.argsort(peaks[1]["prominences"])][-number]
        else:
            prom_peak_index = np.nan
        return prom_peak_index

    def _position_from_index(x_axis_vals, position):
        res = []
        if not (np.isnan(position)):
            res.append(x_axis_vals[int(position)])
        else:
            res.append(np.nan)
        return np.array(res)

    def _width_from_index(da, position):
        res = []
        if not (np.isnan(position)):
            res.append(peak_widths(da.copy(), peaks=[int(position)])[0][0])
        else:
            res.append(np.nan)
        return np.array(res)

    peaks_inversion = 2.0 * (da.mean(dim=dim) - da.min(dim=dim) < da.max(dim=dim) - da.mean(dim=dim)) - 1
    da = da * peaks_inversion

    base_line = xr.apply_ufunc(
        _baseline_als,
        da,
        1e8,
        0.001,
        input_core_dims=[[dim], [], []],
        output_core_dims=[[dim]],
        vectorize=True,
    )

    if remove_baseline:
        da = da - base_line

    base_line = base_line * peaks_inversion

    dim_step = da.coords[dim].diff(dim=dim).values[0]

    # Taking a rolling mean and subtracting it to estimate the noise of the signal
    rolling = da.rolling({dim: 10}, center=True).mean(dim=dim)
    std = float((da - rolling).std())

    prom_peak_index = xr.apply_ufunc(
        _index_of_largest_peak,
        da,
        prominence_factor * std,
        input_core_dims=[[dim], []],
        vectorize=True,
    )
    peak_position = xr.apply_ufunc(
        _position_from_index,
        1.0 * da.coords[dim],
        prom_peak_index,
        input_core_dims=[[dim], []],
        output_core_dims=[[]],
        vectorize=True,
    )
    peak_width = (
        xr.apply_ufunc(
            _width_from_index,
            da,
            prom_peak_index,
            input_core_dims=[[dim], []],
            output_core_dims=[[]],
            vectorize=True,
        )
        * dim_step
    )
    peak_amp = da.max(dim=dim) - da.min(dim=dim) - std

    return xr.merge(
        [
            peak_position.rename("position"),
            peak_width.rename("width"),
            peak_amp.rename("amplitude"),
            base_line.rename("base_line"),
        ]
    )


def fix_initial_value(x, da):
    if len(da.dims) == 1:
        return float(x)
    else:
        return x


def decay_exp(t, a, offset, decay, **kwargs):
    return a * np.exp(t * decay) + offset


def fit_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return ca.guess.exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    def get_min(dat):
        return np.min(dat, axis=-1)

    decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename("decay guess")
    amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")
    min_guess = xr.apply_ufunc(get_min, da, input_core_dims=[[dim]]).rename("min guess")

    def apply_fit(x, y, a, offset, decay):
        try:
            # fit = curve_fit(decay_exp, x, y, p0=[a, offset, decay], bounds=(0, [1, 1., -1]))[0]
            fit, residuals = curve_fit(decay_exp, x, y, p0=[a, offset, decay])
            return np.array(fit.tolist() + np.array(residuals).flatten().tolist())
            # return np.array([fit.values[k] for k in ["a", "offset", "decay"]])
        except RuntimeError as e:
            print("Fit failed:")
            print(f"{a=}, {offset=}, {decay=}")
            plt.plot(x, decay_exp(x, a, offset, decay))
            plt.plot(x, y)
            plt.show()
            # raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        min_guess,
        decay_guess,
        input_core_dims=[[dim], [dim], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(
        fit_vals=(
            "fit_vals",
            [
                "a",
                "offset",
                "decay",
                "a_a",
                "a_offset",
                "a_decay",
                "offset_a",
                "offset_offset",
                "offset_decay",
                "decay_a",
                "decay_offset",
                "decay_decay",
            ],
        )
    )


def oscillation_decay_exp(t, a, f, phi, offset, decay):
    return a * np.exp(-t * decay) * np.cos(2 * np.pi * f * t + phi) + offset


def fit_oscillation_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return ca.guess.oscillation_exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_freq(dat):
        def f(d):
            return ca.guess.frequency(da[dim], d)

        return np.apply_along_axis(f, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename("decay guess")
    freq_guess = xr.apply_ufunc(get_freq, da, input_core_dims=[[dim]]).rename("freq guess")
    amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")

    def apply_fit(x, y, a, f, phi, offset, decay):
        try:
            fit, residuals = curve_fit(oscillation_decay_exp, x, y, p0=[a, f, phi, offset, decay])
            return np.array(fit.tolist() + np.array(residuals).flatten().tolist())
        except RuntimeError as e:
            print(f"{a=}, {f=}, {phi=}, {offset=}, {decay=}")
            plt.plot(x, oscillation_decay_exp(x, a, f, phi, offset, decay))
            plt.plot(x, y)
            plt.show()
            # raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess,
        0,
        0.5,
        decay_guess,
        input_core_dims=[[dim], [dim], [], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(
        fit_vals=(
            "fit_vals",
            [
                "a",
                "f",
                "phi",
                "offset",
                "decay",
                "a_a",
                "a_f",
                "a_phi",
                "a_offset",
                "a_decay",
                "f_a",
                "f_f",
                "f_phi",
                "f_offset",
                "f_decay",
                "phi_a",
                "phi_f",
                "phi_phi",
                "phi_offset",
                "phi_decay",
                "offset_a",
                "offset_f",
                "offset_phi",
                "offset_offset",
                "offset_decay",
                "decay_a",
                "decay_f",
                "decay_phi",
                "decay_offset",
                "decay_decay",
            ],
        )
    )


def echo_decay_exp(t, a, offset, decay, decay_echo):
    return a * np.exp(-t * decay - (t * decay_echo) ** 2) + offset
    # return a * np.exp(-t * decay) + offset


def fit_echo_decay_exp(da, dim):
    def get_decay(dat):
        def oed(d):
            return ca.guess.oscillation_exp_decay(da[dim], d)

        return np.apply_along_axis(oed, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    decay_guess = xr.apply_ufunc(get_decay, da, input_core_dims=[[dim]]).rename("decay guess")
    amp_guess = xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess")

    def apply_fit(x, y, a, offset, decay, decay_echo):
        try:
            fit = curve_fit(echo_decay_exp, x, y, p0=[a, offset, decay, decay_echo])[0]
            return fit
        except RuntimeError as e:
            print(f"{a=}, {offset=}, {decay=}, {decay_echo=}")
            plt.plot(x, echo_decay_exp(x, a, offset, decay, decay_echo))
            plt.plot(x, y)
            plt.show()
            # raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        -0.0005,
        decay_guess,
        decay_guess,
        input_core_dims=[[dim], [dim], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(fit_vals=("fit_vals", ["a", "offset", "decay", "decay_echo"]))


def oscillation(t, a, f, phi, offset):
    return a * np.cos(2 * np.pi * f * t + phi) + offset


def fit_oscillation(da, dim):
    def get_freq(dat):
        def f(d):
            return ca.guess.frequency(da[dim], d)

        return np.apply_along_axis(f, -1, dat)

    def get_amp(dat):
        max_ = np.max(dat, axis=-1)
        min_ = np.min(dat, axis=-1)
        return (max_ - min_) / 2

    da_c = da - da.mean(dim=dim)
    freq_guess = fix_initial_value(
        xr.apply_ufunc(get_freq, da_c, input_core_dims=[[dim]]).rename("freq guess"),
        da_c,
    )
    amp_guess = fix_initial_value(xr.apply_ufunc(get_amp, da, input_core_dims=[[dim]]).rename("amp guess"), da)
    # phase_guess = np.pi * (da.loc[{dim : da.coords[dim].values[0]}] < da.mean(dim=dim) )
    phase_guess = np.pi * (da.loc[{dim: np.abs(da.coords[dim]).min()}] < da.mean(dim=dim))
    offset_guess = da.mean(dim=dim)

    def apply_fit(x, y, a, f, phi, offset):
        try:
            model = Model(oscillation, independent_vars=["t"])
            fit = model.fit(
                y,
                t=x,
                a=Parameter("a", value=a, min=0),
                f=Parameter("f", value=f, min=np.abs(0.5 * f), max=np.abs(f * 3 + 1e-3)),
                phi=Parameter("phi", value=phi),
                offset=offset,
            )
            if fit.rsquared < 0.9:
                fit = model.fit(
                    y,
                    t=x,
                    a=Parameter("a", value=a, min=0),
                    f=Parameter(
                        "f",
                        value=1.0 / (np.max(x) - np.min(x)),
                        min=0,
                        max=np.abs(f * 3 + 1e-3),
                    ),
                    phi=Parameter("phi", value=phi),
                    offset=offset,
                )
            return np.array([fit.values[k] for k in ["a", "f", "phi", "offset"]])
        except RuntimeError as e:
            print(f"{a=}, {f=}, {phi=}, {offset=}")
            plt.plot(x, oscillation(x, a, f, phi, offset))
            plt.plot(x, y)
            plt.show()
            raise e

    fit_res = xr.apply_ufunc(
        apply_fit,
        da[dim],
        da,
        amp_guess,
        freq_guess,
        phase_guess,
        offset_guess,
        input_core_dims=[[dim], [dim], [], [], [], []],
        output_core_dims=[["fit_vals"]],
        vectorize=True,
    )
    return fit_res.assign_coords(fit_vals=("fit_vals", ["a", "f", "phi", "offset"]))


def extract_dominant_frequencies(da, dim="idle_time"):
    def extract_dominant_frequency(signal, sample_rate):
        fft_result = fft(signal)
        frequencies = np.fft.fftfreq(len(signal), 1 / sample_rate)
        positive_freq_idx = np.where(frequencies > 0)
        dominant_idx = np.argmax(np.abs(fft_result[positive_freq_idx]))
        return frequencies[positive_freq_idx][dominant_idx]

    def extract_dominant_frequency_wrapper(signal):
        sample_rate = 1 / (da.coords[dim][1].values - da.coords[dim][0].values)  # Assuming uniform sampling
        return extract_dominant_frequency(signal, sample_rate)

    dominant_frequencies = xr.apply_ufunc(
        extract_dominant_frequency_wrapper,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[]],
        vectorize=True,
    )

    return dominant_frequencies
