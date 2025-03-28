import numpy as np
import xarray as xr
from lmfit import Model, Parameter, Parameters
from scipy.signal import find_peaks
from quam_experiments.analysis.fit import peaks_dips


def _S21_abs(w, A, k, phi, kappa_p, omega_p, omega_r, J):

    # real transmission function for two resonator in a shunt geometry, assuming
    # a wide-band resonator (purcell filter) coupled with a coupling strength kappa_p
    # to a feedline, which in turn is coupled with strength J to another resonator
    # based on arxiv:2307.07765

    Delta_p = omega_p - w
    Delta_r = omega_r - w

    return (A + k * w) * np.abs(
        np.cos(phi)
        - np.exp(1j * phi)
        * kappa_p
        * (-2 * 1j * Delta_r)
        / (4 * J**2 + (kappa_p - 2 * 1j * Delta_p) * (-2 * 1j * Delta_r))
    )


def _S21_single(w, A, k, omega_0, omega_r, Q, Qe_real, Qe_imag):

    # complex transmision function for a single resonator based on Khalil et al. (arxiv:1108.3117):
    #  but slightly modified for convenince, by taking into account a non-normalized transmission
    # and a linear slope of the transmission far from resonance

    Qe = Qe_real + 1j * Qe_imag
    return (A + k * w) * (1 - ((Q / Qe) / (1 + 2 * 1j * Q * (w - omega_r) / (omega_0 + omega_r))))


def _truncate_data(transmission, window):
    ds_diff = np.abs(transmission.diff(dim="freq"))
    peak_freq = ds_diff.IQ_abs.idxmax(dim="freq")
    truncated_transmission = transmission.sel(freq=slice(peak_freq - window, peak_freq + window))
    return truncated_transmission


def _guess_2_resonators(transmission):

    # Function to find the intial guess that consists of the frequencies of two resonators, the coupling
    # of one of the (the Purcell filter) to a feedline, and the coupling between them. The procudre is to
    # first find the largest derivative, and assume this is where the dressed resonator is. Then, we
    # truncate the data in a fixed window around that range, and look for the two frequencies with the
    # largest dertivatves that are separated from each other by at least 'min_distnace'. This two dressed
    # frequencies are converted to the bare frequencies using the coupling strength 'J'. Furthermore,
    # a linear fit is done to the upper envelope to account for the possible linear transmission profile.

    first = peaks_dips(transmission.IQ_abs, dim="freq", number=1)
    second = peaks_dips(transmission.IQ_abs, dim="freq", number=2)
    if first.width > second.width:
        omega_p = first.position.values
        kappa_p = 2 * first.width.values
        omega_r = second.position.values
        J = 2 * second.width.values
    else:
        omega_r = first.position.values
        J = 2 * first.width.values
        omega_p = second.position.values
        kappa_p = 2 * second.width.values
    k, A = (-first.base_line).polyfit(dim="freq", deg=1).polyfit_coefficients.values
    init_params = Parameters()
    init_params.add("J", value=J[0], min=0)
    init_params.add("omega_r", value=omega_r[0])
    init_params.add("omega_p", value=omega_p[0])
    init_params.add("k", value=k[0])
    init_params.add("A", value=A[0])
    init_params.add("kappa_p", value=kappa_p[0], min=0)
    init_params.add("phi", value=0)

    return init_params


class _two_resonator_model(Model):
    # A class to fit the S21 model to a data. Accepts an xarray data
    # that contains I and Q measured as a function of freq.

    def __init__(self, J=0, kappa_p=0, *args, **kwargs):
        super().__init__(_S21_abs, *args, **kwargs)

        # params used in the initial guess generator:
        # the window one which the fitting is done, around the initial guess for the resonator peak
        self.window = 100e6

    def make_fit(self, transmission, init_guess=None):

        transmission_trunc = _truncate_data(transmission, self.window)

        if init_guess == None:
            init_guess = _guess_2_resonators(transmission_trunc)

        data = transmission_trunc.IQ_abs.values
        f = transmission_trunc.freq.values

        result = self.fit(data, w=f, params=init_guess)

        return result


def fit_resonator_purcell(
    s21_data: xr.Dataset,
    init_J: float = 15e6,
    init_kappa_p: float = 10e6,
    print_report: bool = False,
):
    """Fits the measured complex transmission as a function of frequency
    and fits it to a model consisting of two resonators coupled to each
    othewr, as described in arxiv:2307.07765.
    IMPORTANT: the fit assumes a that within the measureument window there
    are clear two dips.
    The transmssion function is:
    S_{21} = (k * w + A) * [
             cos(phi) - exp(i phi)  kappa_p  (- 2 * i * Delta_r) /
             ( 4  J^2 + (kappa_p - 2 i Delta_p) (-2 i Delta_r))  ]

    See the output for a descirption of the model parameters.

    Args:
        transmission (xarray.DataSet ): DataSet which golds the measured data,
                                        assumes that it has a DataAray labels
                                        'IQ_abs' and 'phase' containing the
                                        absolute value and the phase of the
                                        signal. The only coordinate is 'freq',
                                        the frequency for which the signal is
                                        measured.
        print_report (bool, optional): If set to True prints the lmfit report.
                                        Defaults to False.

    Returns:
        fit [lmfit.ModelResult] : The resulting fit to the data using the two resonator model.
                            The fitted parameters (accessd through the 'params' object'
                            are:
                            params['J'] - coupling betweem the resonator and the Purcell
                            filter [Hz]
                            params['omega_r'] - bare frequency of the resonator
                            params['omega_p'] - bare frequency of the Purcell filter
                            params['kappa_p'] - coupling strength of the Purcell filter
                            to the resonator
                            params['A'] - empirical amplitude of the transmission
                            params['k']  - an empirical linear slope of the transmission
                            params['phi'] - a possible phase acquired by the signal due
                            to unintended capacitance.
                            Note that the dressed resonator frequency, the location of the
                            resonator  dip in the signal, is not a fit parameter. It can
                            be calculated from the model or taken from the initial guess
                            which looks for that from 'result.init_params['omega_r']'
        fit_eval [np.array] : A complex numpy array of the fit function evaluated on in the
                            relevent range
    """

    resonator_abs = _two_resonator_model(J=init_J, kappa_p=init_kappa_p)

    fit = resonator_abs.make_fit(s21_data)
    fit_eval = resonator_abs.eval(params=fit.params, w=s21_data.freq.values)

    if print_report:
        print(fit.fit_report() + "\n")
        fit.params.pretty_print()

    return fit, fit_eval


def _guess_single(transmission, frequency_LO_IF, rolling_window, window):
    def find_upper_envelope(transmission, rolling_window):
        rolling_transmission = transmission.IQ_abs.rolling(freq=rolling_window, center=True).mean()
        peak_indices, _ = find_peaks(rolling_transmission)
        # include the edges of the range in the envelope fit in case there aren't many inside peaks to use
        peak_indices = np.append([rolling_window, -1], peak_indices)
        envelope = rolling_transmission.isel(freq=peak_indices)
        k, A = envelope.polyfit(dim="freq", deg=1).polyfit_coefficients.values
        return k, A

    k, A = find_upper_envelope(transmission, rolling_window=rolling_window)

    # plt.figure()
    # transmission.IQ_abs.plot()
    # plt.plot(transmission.freq,transmission.freq*k + A)
    # plt.show()

    omega_r = transmission.IQ_abs.idxmin(dim="freq")
    Q = frequency_LO_IF / np.abs((transmission.IQ_abs.diff(dim="freq").idxmin(dim="freq") - omega_r)).values
    Q = Q if Q < 1e4 else 1e4
    Q = 1e4
    Qe = Q / (1 - transmission.IQ_abs.min(dim="freq") / transmission.IQ_abs.max(dim="freq"))

    Qe = Qe if Qe > Q else Q

    init_params = Parameters()
    init_params.add("omega_0", value=frequency_LO_IF, vary=False)
    init_params.add("omega_r", value=omega_r.values + 0.1e6)
    init_params.add("k", value=k)
    init_params.add("A", value=A)
    init_params.add("Q", value=Q, min=0)
    init_params.add("Qe_real", value=Qe.values, min=0)
    init_params.add("Qe_imag", value=0, min=0)

    return init_params


class _single_resonator(Model):
    # A class to fit the S21 model to a data. Accepts an xarray data
    # that contains I and Q measured as a function of freq.

    def __init__(self, *args, **kwargs):
        super().__init__(_S21_single, *args, **kwargs)

        # params used in the initial guess generator:
        # used to smooth data to improve peak detection
        self.rolling_window = 1
        # the window one which the fitting is done, around the initial guess for the resonator peak
        self.window = 15e6

    def make_fit(self, transmission, frequency_LO_IF, init_guess=None):

        # transmission_trunc = _truncate_data(transmission,self.window)
        transmission_trunc = transmission

        if init_guess == None:
            init_guess = _guess_single(
                transmission_trunc,
                frequency_LO_IF=frequency_LO_IF,
                rolling_window=self.rolling_window,
                window=self.window,
            )

        data = (transmission_trunc.IQ_abs * np.exp(1j * transmission_trunc.phase)).values
        f = transmission_trunc.freq.values

        result = self.fit(data, w=f, params=init_guess)

        return result


def fit_resonator(s21_data: xr.Dataset, frequency_LO_IF: float, print_report: bool = False):
    """Fits the measured complex transmission as a function of frequency
    and fits it to a model consisting of a single resonator, as described in
    arxiv:1108.3117.

    The transmission function is:
    S_{21} =(A + k  w)  (
        1 - ((Q/Qe) / (1 + 2 i Q  (w - omega_r)/(omega_0 + omega_r))))

    See the output for a description of the model parameters.

    Args:
        transmission (xarray.DataSet ): DataSet which golds the measured data,
                                        assumes that it has a DataAray labels
                                        'IQ_abs' and 'phase' containing the
                                        absolute value and the phase of the
                                        signal. The only coordinate is 'freq',
                                        the frequency for which the signal is
                                        measured.
        frequency_LO_IF (int): The frequency relative to which the data was taken.
                                Should be the sum of the LO and IF.
        print_report (bool, optional): If set to True prints the lmfit report.
                                        Defaults to False.

    Returns:
        fit [lmfit.ModelResult] : The resulting fit to the data using the two resonator model.
                            The fitted parameters (accessed through the 'params' object'
                            are:
                            params['omega_r'] - resonator frequency
                            params['Qe_imag'] - imaginary part of the external quality
                            factor
                            params['Qe_real'] - real part of the external quality
                            factor
                            params['Q'] - the total quality factor of the resonator
                            params['A'] - empirical amplitude of the transmission
                            params['k']  - an empirical linear slope of the transmission
        fit_eval [np.array] : A complex numpy array of the fit function evaluated on in the
                            relevant range
    """

    resonator = _single_resonator()

    fit = resonator.make_fit(s21_data, frequency_LO_IF=frequency_LO_IF)
    fit_eval = resonator.eval(params=fit.params, w=s21_data.freq.values)

    if print_report:
        print(fit.fit_report() + "\n")
        fit.params.pretty_print()

    return fit, fit_eval
