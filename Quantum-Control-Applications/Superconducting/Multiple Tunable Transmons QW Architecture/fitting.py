from scipy import optimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from typing import List, Union
import itertools
import json
import numpy as np


class Fit:
    """
    This class takes care of the fitting to the measured data.
    It includes:
        - Fitting to: linear line
                      T1 experiment
                      Ramsey experiment
                      transmission resonator spectroscopy
                      reflection resonator spectroscopy
        - Printing the initial guess and fitting results
        - Plotting the data and the fitting function
        - Saving the data
    """

    @staticmethod
    def linear(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ) -> dict:
        """
        Create a linear fit of the form

        .. math::
        f(x) = a * x + b

        for unknown parameters :
             a - The slope of the function
             b - The free parameter of the function

         :param x_data: The data on the x-axis
         :param y_data: The data on the y-axis
         :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(a=20))
         :param verbose: if True prints the initial guess and fitting results
         :param plot: if True plots the data and the fitting function
         :param save: if not False saves the data into a json file
                      The id of the file is save='id'. The name of the json file is `id.json`
         :return: A dictionary of (fit_func, a, b)

        """

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Finding an initial guess to the slope
        a0 = (y[-1] - y[0]) / (x[-1] - x[0])

        # Finding an initial guess to the free parameter
        b0 = y[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "a":
                    a0 = float(guess[key]) * x_normal / y_normal
                elif key == "b":
                    b0 = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )
        # Print the initial guess if verbose=True
        if verbose:
            print(f"Initial guess:\n" f" a = {a0 * y_normal / x_normal:.3f}, \n" f" b = {b0 * y_normal:.3f}")

        # Fitting function
        def func(x_var, c0, c1):
            return a0 * c0 * x_var + b0 * c1

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1])

        popt, pcov = optimize.curve_fit(func, x, y, p0=[1, 1])
        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "a": [
                popt[0] * a0 * y_normal / x_normal,
                perr[0] * a0 * y_normal / x_normal,
            ],
            "b": [popt[1] * b0 * y_normal, perr[1] * b0 * y_normal],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" a = {out['a'][0]:.3f} +/- {out['a'][1]:.3f}, \n"
                f" b = {out['b'][0]:.3f} +/- {out['b'][1]:.3f}"
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"a  = {out['a'][0]:.1f} +/- {out['a'][1]:.1f} \n b  = {out['b'][0]:.1f} +/- {out['b'][1]:.1f}",
            )
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (func(x, popt[0], popt[1]) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)

        return out

    @staticmethod
    def T1(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to T1 experiment of the form

        .. math::
        f(x) = amp * np.exp(-x * (1/T1)) + final_offset

        for unknown parameters :
            T1 - The decay constant [ns]
            amp - The amplitude [a.u.]
            final_offset -  The offset visible for long dephasing times [a.u.]

        :param x_data: The dephasing time [ns]
        :param y_data: Data containing the Ramsey signal
        :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(T1=20))
        :param verbose: if True prints the initial guess and fitting results
        :param plot: if True plots the data and the fitting function
        :param save: if not False saves the data into a json file
                     The id of the file is save='id'. The name of the json file is `id.json`
        :return: A dictionary of (fit_func, T1, amp, final_offset)

        """
        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Finding a guess for the decay (slope of log(peaks))
        derivative = np.abs(np.diff(y))
        if np.std(np.log(derivative)) < np.abs(np.mean(np.log(derivative)[-10:]) - np.mean(np.log(derivative)[:10])):
            guess_T1 = (
                -1
                / ((np.mean(np.log(derivative)[-10:]) - np.mean(np.log(derivative)[:10])) / (len(y) - 1))
                * (x[1] - x[0])
            )
        # Initial guess if the data is too noisy
        else:
            guess_T1 = 100 / x_normal
        # Finding a guess for the offsets
        final_offset = np.mean(y[int(len(y) * 0.9) :])

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "T1":
                    guess_T1 = float(guess[key]) / x_normal
                elif key == "amp":
                    pass
                elif key == "final_offset":
                    final_offset = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )
        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n "
                f"T1 = {guess_T1 * x_normal:.3f}, \n "
                f"amp = {y[0] * y_normal:.3f}, \n "
                f"final offset = {final_offset * y_normal:.3f}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2):
            return a1 * y[0] * np.exp(-x_var / (guess_T1 * a0)) + final_offset * a2

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2])

        popt, pcov = optimize.curve_fit(
            func,
            x,
            y,
            p0=[1, 1, 1],
        )

        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "T1": [(guess_T1 * popt[0]) * x_normal, perr[0] * guess_T1 * x_normal],
            "amp": [popt[1] * y[0] * y_normal, perr[1] * y[0] * y_normal],
            "final_offset": [
                popt[2] * final_offset * y_normal,
                perr[2] * final_offset * y_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" T1 = {out['T1'][0]:.2f} +/- {out['T1'][1]:.3f} ns, \n"
                f" amp = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f} a.u., \n"
                f" final offset = {out['final_offset'][0]:.2f} +/- {out['final_offset'][1]:.3f} a.u."
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"T1  = {out['T1'][0]:.1f} +/- {out['T1'][1]:.1f}ns",
            )
            plt.xlabel("Waiting time [ns]")
            plt.ylabel(r"$\sqrt{I^2+Q^2}$ [a.u.]")
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (fit_type(x, popt) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)

        return out

    @staticmethod
    def ramsey(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to Ramsey experiment of the form

        .. math::
        f(x) = final_offset * (1 - np.exp(-x * (1/T2))) + amp / 2 * (
            np.exp(-x * (1/T2))
            * (initial_offset * 2 + np.cos(2 * np.pi * f * x + phase))
            )

        for unknown parameters :
            f - The detuning frequency [GHz]
            phase - The phase [rad]
            T2 - The decay constant [ns]
            amp - The amplitude [a.u.]
            final_offset -  The offset visible for long dephasing times [a.u.]
            initial_offset - The offset visible for short dephasing times

        :param x_data: The dephasing time [ns]
        :param y_data: Data containing the Ramsey signal
        :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(T2=20))
        :param verbose: if True prints the initial guess and fitting results
        :param plot: if True plots the data and the fitting function
        :param save: if not False saves the data into a json file
                     The id of the file is save='id'. The name of the json file is `id.json`
          :return: A dictionary of (fit_func, f, phase, tau, amp, uncertainty_population, initial_offset)

        """

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Compute the FFT for guessing the frequency
        fft = np.fft.fft(y)
        f = np.fft.fftfreq(len(x))
        # Take the positive part only
        fft = fft[1 : len(f) // 2]
        f = f[1 : len(f) // 2]
        # Remove the DC peak if there is one
        if (np.abs(fft)[1:] - np.abs(fft)[:-1] > 0).any():
            first_read_data_ind = np.where(np.abs(fft)[1:] - np.abs(fft)[:-1] > 0)[0][0]  # away from the DC peak
            fft = fft[first_read_data_ind:]
            f = f[first_read_data_ind:]

        # Finding a guess for the frequency
        out_freq = f[np.argmax(np.abs(fft))]
        guess_freq = out_freq / (x[1] - x[0])

        # The period is 1 / guess_freq --> number of oscillations --> peaks decay to get guess_T2
        period = int(np.ceil(1 / out_freq))
        peaks = (
            np.array([np.std(y[i * period : (i + 1) * period]) for i in range(round(len(y) / period))]) * np.sqrt(2) * 2
        )

        # Finding a guess for the decay (slope of log(peaks))
        if len(peaks) > 1:
            guess_T2 = -1 / ((np.log(peaks)[-1] - np.log(peaks)[0]) / (period * (len(peaks) - 1))) * (x[1] - x[0])
        else:
            guess_T2 = 100 / x_normal
            print(
                Warning(
                    "WARNING: The initial guess for the decay failed, increasing the number of oscillations should solve the issue."
                )
            )

        # Finding a guess for the offsets
        initial_offset = np.mean(y[:period])
        final_offset = np.mean(y[-period:])

        # Finding a guess for the phase
        guess_phase = np.angle(fft[np.argmax(np.abs(fft))]) - guess_freq * 2 * np.pi * x[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    guess_freq = float(guess[key]) * x_normal
                elif key == "phase":
                    guess_phase = float(guess[key])
                elif key == "T2":
                    guess_T2 = float(guess[key]) * x_normal
                elif key == "amp":
                    peaks[0] = float(guess[key]) / y_normal
                elif key == "initial_offset":
                    initial_offset = float(guess[key]) / y_normal
                elif key == "final_offset":
                    final_offset = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )

        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n"
                f" f = {guess_freq / x_normal:.3f}, \n"
                f" phase = {guess_phase:.3f}, \n"
                f" T2 = {guess_T2 * x_normal:.3f}, \n"
                f" amp = {peaks[0] * y_normal:.3f}, \n"
                f" initial offset = {initial_offset * y_normal:.3f}, \n"
                f" final_offset = {final_offset * y_normal:.3f}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2, a3, a4, a5):
            return final_offset * a4 * (1 - np.exp(-x_var / (guess_T2 * a1))) + peaks[0] / 2 * a2 * (
                np.exp(-x_var / (guess_T2 * a1))
                * (a5 * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a0 * guess_freq * x + a3))
            )

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3], a[4], a[5])

        popt, pcov = optimize.curve_fit(
            func,
            x,
            y,
            p0=[1, 1, 1, guess_phase, 1, 1],
        )

        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [popt[0] * guess_freq / x_normal, perr[0] * guess_freq / x_normal],
            "phase": [popt[3] % (2 * np.pi), perr[3] % (2 * np.pi)],
            "T2": [(guess_T2 * popt[1]) * x_normal, perr[1] * guess_T2 * x_normal],
            "amp": [peaks[0] * popt[2] * y_normal, perr[2] * peaks[0] * y_normal],
            "initial_offset": [
                popt[5] * initial_offset * y_normal,
                perr[5] * initial_offset * y_normal,
            ],
            "final_offset": [
                final_offset * popt[4] * y_normal,
                perr[4] * final_offset * y_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz, \n"
                f" phase = {out['phase'][0]:.3f} +/- {out['phase'][1]:.3f} rad, \n"
                f" T2 = {out['T2'][0]:.2f} +/- {out['T2'][1]:.3f} ns, \n"
                f" amp = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f} a.u., \n"
                f" initial offset = {out['initial_offset'][0]:.2f} +/- {out['initial_offset'][1]:.3f}, \n"
                f" final_offset = {out['final_offset'][0]:.2f} +/- {out['final_offset'][1]:.3f} a.u."
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"T2  = {out['T2'][0]:.1f} +/- {out['T2'][1]:.1f}ns \n f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz",
            )
            plt.xlabel("Waiting time [ns]")
            plt.ylabel(r"$\sqrt{I^2+Q^2}$ [a.u.]")
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (fit_type(x, popt) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)
        return out

    @staticmethod
    def transmission_resonator_spectroscopy(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to transmission resonator spectroscopy of the form

        .. math::
        ((kc/k) / (
            1 + (4 * ((x - f) ** 2) / (k ** 2)))) + offset

        for unknown parameters:
            f - The frequency at the peak
            kc - The strength with which the field of the resonator couples to the transmission line
            ki - A parameter that indicates the internal coherence properties of the resonator
            k - The FWHM of the fitted function.  k = ki + kc
            offset - The offset

        :param x_data:  The frequency in Hz
        :param y_data: The transition probability (I^2+Q^2)
        :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(f=20e6))
        :param verbose: if True prints the initial guess and fitting results
        :param plot: if True plots the data and the fitting function
        :param save: if not False saves the data into a json file
                     The id of the file is save='id'. The name of the json file is `id.json`
             :return: A dictionary of (fit_func, f, kc, k, ki, offset)

        """

        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Finding a guess for the max
        peak = max(y)
        arg_max = y.argmax()

        # Finding an initial guess for the FWHM
        if arg_max > len(y) / 2:
            y_FWHM = (peak + np.mean(y[0:10])) / 2
        else:
            y_FWHM = (peak + np.mean(y[-10:-1])) / 2

        # Finding a guess for the width
        width0_arg_right = (np.abs(y_FWHM - y[arg_max + 1 : len(y)])).argmin() + arg_max
        width0_arg_left = (np.abs(y_FWHM - y[0:arg_max])).argmin()
        width0 = x[width0_arg_right] - x[width0_arg_left]

        # Finding the frequency at the min
        f0 = x[arg_max]

        # Finding a guess to offset
        v0 = np.mean(y[0 : int(width0_arg_left - width0 / 2)])

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    f0 = float(guess[key]) / x_normal
                elif key == "k":
                    width0 = float(guess[key]) / x_normal
                elif key == "offset":
                    v0 = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )
        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n "
                f" f = {f0 * x_normal}, \n "
                f" kc = {(peak - v0) * (width0 * x_normal) * y_normal}, \n "
                f" k = {width0 * x_normal}, \n "
                f" offset = {v0 * y_normal}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2, a3):
            return (((peak - v0) * a0) / (1 + (4 * ((x_var - (f0 * a2)) ** 2) / ((width0 * a1) ** 2)))) + (v0 * a3)

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3])

        popt, pcov = optimize.curve_fit(func, x, y, p0=[1, 1, 1, 1])
        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [f0 * popt[2] * x_normal, f0 * perr[2] * x_normal],
            "kc": [
                (peak - v0) * popt[0] * (width0 * popt[1] * x_normal) * y_normal,
                (peak - v0) * perr[0] * (width0 * perr[1] * x_normal) * y_normal,
            ],
            "ki": [
                (popt[1] * width0 * x_normal) - ((peak - v0) * popt[0] * (width0 * popt[1] * x_normal) * y_normal),
                (perr[1] * width0 * x_normal) - ((peak - v0) * perr[0] * (width0 * perr[1] * x_normal) * y_normal),
            ],
            "k": [popt[1] * width0 * x_normal, perr[1] * width0 * x_normal],
            "offset": [v0 * popt[3] * y_normal, v0 * perr[3] * y_normal],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fit results:\n"
                f"f = {out['f'][0]:.3f} +/- {out['f'][1]:.3f} Hz, \n"
                f"kc = {out['kc'][0]:.3f} +/- {out['kc'][1]:.3f} Hz, \n"
                f"ki = {out['ki'][0]:.3f} +/- {out['ki'][1]:.3f} Hz, \n"
                f"k = {out['k'][0]:.3f} +/- {out['k'][1]:.3f} Hz, \n"
                f"offset = {out['offset'][0]:.3f} +/- {out['offset'][1]:.3f} Hz \n"
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"k  = {out['k'][0]:.1f} +/- {out['k'][1]:.1f}Hz",
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(r"$\sqrt{I^2+Q^2}$ [a.u.]")
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (fit_type(x, popt) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)

        return out

    @staticmethod
    def reflection_resonator_spectroscopy(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to reflection resonator spectroscopy of the form

        .. math::
        (offset)-((kc/k) / (
            1 + (4 * ((x - f) ** 2) / (k ** 2)))) + slope * x
        for unknown parameters:
            f - The frequency at the peak
            kc - The strength with which the field of the resonator couples to the transmission line
            ki - A parameter that indicates the internal coherence properties of the resonator
            k - The FWHM of the fitted function.  k = ki + kc
            offset - The offset
            slope - The slope of the function. This is added after experimental considerations.

        :param x_data:  The frequency in Hz
        :param y_data: The transition probability (I^2+Q^2)
        :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(f=20e6))
        :param verbose: if True prints the initial guess and fitting results
        :param plot: if True plots the data and the fitting function
        :param save: if not False saves the data into a json file
                     The id of the file is save='id'. The name of the json file is `id.json`
          :return: A dictionary of (fit_func, f, kc, k, ki, offset)

        """

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Finding a guess for the min
        peak = min(y)
        arg_min = y.argmin()

        # Finding the frequency at the min
        f0 = x[arg_min]

        # Finding an initial guess for the FWHM
        if arg_min > len(y) / 2:
            y_FWHM = (peak + np.mean(y[0:10])) / 2
        else:
            y_FWHM = (peak + np.mean(y[-10:-1])) / 2

        # Finding a guess to the width
        width0_arg_right = (np.abs(y_FWHM - y[arg_min + 1 : len(y)])).argmin() + arg_min
        width0_arg_left = (np.abs(y_FWHM - y[0:arg_min])).argmin()
        width0 = x[width0_arg_right] - x[width0_arg_left]
        width0 = width0
        # Finding guess to offset
        v0 = (np.mean(y[-10:-1]) + np.mean(y[0:10])) / 2

        # Finding a guess to the slope
        m = (np.mean(y[int(width0_arg_right + width0) : -1]) - np.mean(y[0 : int(width0_arg_left - width0)])) / (
            np.mean(x[int(width0_arg_right + width0) : -1]) - np.mean(x[0 : int(width0_arg_left - width0)])
        )

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    f0 = float(guess[key]) / x_normal
                elif key == "k":
                    width0 = float(guess[key]) / x_normal
                elif key == "offset":
                    v0 = float(guess[key]) / y_normal
                elif key == "slope":
                    m = float(guess[key]) / y_normal * x_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )

        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n "
                f" f = {f0 * x_normal}, \n "
                f" kc = {(v0 - peak) * (width0 * x_normal) * y_normal}, \n "
                f" k = {width0 * x_normal}, \n "
                f" offset = {v0 * y_normal}, \n "
                f" slope = {m * y_normal / x_normal} "
            )

        def func(x_var, a0, a1, a2, a3, a4):
            return (
                ((v0 - peak) * a3)
                - (((v0 - peak) * a0) / (1 + (4 * ((x_var - (f0 * a2)) ** 2) / ((width0 * a1) ** 2))))
                + m * a4 * x_var
            )

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3], a[4])

        popt, pcov = optimize.curve_fit(func, x, y, p0=[1, 1, 1, 1, 1])
        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [f0 * popt[2] * x_normal, f0 * perr[2] * x_normal],
            "kc": [
                (v0 - peak) * popt[0] * (width0 * popt[1] * x_normal) * y_normal,
                (v0 - peak) * perr[0] * (width0 * perr[1] * x_normal) * y_normal,
            ],
            "ki": [
                (popt[1] * width0 * x_normal) - ((v0 - peak) * popt[0] * (width0 * popt[1] * x_normal) * y_normal),
                (perr[1] * width0 * x_normal) - ((v0 - peak) * perr[0] * (width0 * perr[1] * x_normal) * y_normal),
            ],
            "k": [popt[1] * width0 * x_normal, perr[1] * width0 * x_normal],
            "offset": [
                (v0 - peak) * popt[3] * y_normal,
                (v0 - peak) * perr[3] * y_normal,
            ],
            "slope": [
                m * popt[4] * y_normal / x_normal,
                m * perr[4] * y_normal / x_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fit results:\n"
                f"f = {out['f'][0]:.3f} +/- {out['f'][1]:.3f} Hz, \n"
                f"kc = {out['kc'][0]:.3f} +/- {out['kc'][1]:.3f} Hz, \n"
                f"ki = {out['ki'][0]:.3f} +/- {out['ki'][1]:.3f} Hz, \n"
                f"k = {out['k'][0]:.3f} +/- {out['k'][1]:.3f} Hz, \n"
                f"offset = {out['offset'][0]:.3f} +/- {out['offset'][1]:.3f} Hz, \n"
                f"slope = {out['slope'][0]:.3f} +/- {out['slope'][1]:.3f} Hz\n"
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"k  = {out['k'][0]:.1f} +/- {out['k'][1]:.1f}Hz",
            )
            plt.xlabel("Frequency [Hz]")
            plt.ylabel(r"$\sqrt{I^2+Q^2}$ [a.u.]")
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (fit_type(x, popt) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)

        return out

    @staticmethod
    def rabi(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to Rabi experiment of the form

        .. math::
        f(x) = amp * (np.sin(0.5 * (2 * np.pi * f) * x_var + phase))**2 * np.exp(-x_var / T) + offset

        for unknown parameters :
            f - The detuning frequency [GHz]
            phase - The phase [rad]
            T - The decay constant [ns]
            amp - The amplitude [a.u.]
            offset -  The offset visible for long dephasing times [a.u.]

        :param x_data: The dephasing time [ns]
        :param y_data: Data containing the Ramsey signal
        :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(T2=20))
        :param verbose: if True prints the initial guess and fitting results
        :param plot: if True plots the data and the fitting function
        :param save: if not False saves the data into a json file
                     The id of the file is save='id'. The name of the json file is `data_fit_id.json`
          :return: A dictionary of (fit_func, f, phase, tau, amp, uncertainty_population, initial_offset)

        """

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Compute the FFT for guessing the frequency
        fft = np.fft.fft(y)
        f = np.fft.fftfreq(len(x))
        # Take the positive part only
        fft = fft[1 : len(f) // 2]
        f = f[1 : len(f) // 2]
        # Remove the DC peak if there is one
        if (np.abs(fft)[1:] - np.abs(fft)[:-1] > 0).any():
            first_read_data_ind = np.where(np.abs(fft)[1:] - np.abs(fft)[:-1] > 0)[0][0]  # away from the DC peak
            fft = fft[first_read_data_ind:]
            f = f[first_read_data_ind:]

        # Finding a guess for the frequency
        out_freq = f[np.argmax(np.abs(fft))]
        guess_freq = out_freq / (x[1] - x[0])

        # The period is 1 / guess_freq --> number of oscillations --> peaks decay to get guess_T
        period = int(np.ceil(1 / out_freq))
        peaks = (
            np.array([np.std(y[i * period : (i + 1) * period]) for i in range(round(len(y) / period))]) * np.sqrt(2) * 2
        )

        # Finding a guess for the decay (slope of log(peaks))
        if len(peaks) > 1:
            guess_T = -1 / ((np.log(peaks)[-1] - np.log(peaks)[0]) / (period * (len(peaks) - 1))) * (x[1] - x[0])
            print(peaks)
        else:
            guess_T = 100 / x_normal
            print(
                Warning(
                    "WARNING: The initial guess for the decay failed, increasing the number of oscillations should solve the issue."
                )
            )

        # Finding a guess for the offset
        offset = np.mean(y[-period:])

        # Finding a guess for the phase
        guess_phase = np.angle(fft[np.argmax(np.abs(fft))]) - guess_freq * 2 * np.pi * x[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    guess_freq = float(guess[key]) * x_normal
                elif key == "phase":
                    guess_phase = float(guess[key])
                elif key == "T":
                    guess_T = float(guess[key]) * x_normal
                elif key == "amp":
                    peaks[0] = float(guess[key]) / y_normal
                elif key == "offset":
                    offset = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )

        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n"
                f" f = {guess_freq / x_normal:.3f}, \n"
                f" phase = {guess_phase:.3f}, \n"
                f" T = {guess_T * x_normal:.3f}, \n"
                f" amplitude = {peaks[0] * y_normal:.3f}, \n"
                f" offset = {offset * y_normal:.3f}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2, a3, a4):
            return (peaks[0] * a0) * (np.sin(0.5 * (2 * np.pi * guess_freq * a1) * x_var + a3)) ** 2 * np.exp(
                -x_var / np.abs(guess_T * a2)
            ) + offset * a4

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3], a[4])

        popt, pcov = optimize.curve_fit(
            func,
            x,
            y,
            p0=[1, 1, 1, guess_phase, 1],
        )
        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [popt[1] * guess_freq / x_normal, perr[1] * guess_freq / x_normal],
            "phase": [popt[3] % (2 * np.pi), perr[3] % (2 * np.pi)],
            "T": [(guess_T * abs(popt[2])) * x_normal, perr[2] * guess_T * x_normal],
            "amp": [popt[0] * peaks[0] * y_normal, perr[0] * peaks[0] * y_normal],
            "offset": [
                offset * popt[4] * y_normal,
                perr[4] * offset * y_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz, \n"
                f" phase = {out['phase'][0]:.3f} +/- {out['phase'][1]:.3f} rad, \n"
                f" T = {out['T'][0]:.2f} +/- {out['T'][1]:.3f} ns, \n"
                f" amplitude = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f}, \n"
                f" offset = {out['offset'][0]:.2f} +/- {out['offset'][1]:.3f} a.u."
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz \n T  = {out['T'][0]:.1f} +/- {out['T'][1]:.1f}ns",
            )
            plt.xlabel("Time [ns]")
            plt.ylabel(r"$\sqrt{I^2+Q^2}$ [a.u.]")
            plt.legend(loc="upper right")
        # Save the data in a json file named 'data_fit_id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (fit_type(x, popt) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            with open(f"data_fit_{save}.json", "w") as outfile:
                outfile.write(json_object)
        return out


class Read:
    """
    This class takes care of reading the saved data.
    """

    @staticmethod
    def read_saved_params(file_id: str, verbose=False) -> dict:
        """
        Read the saved json file and print the saved params if print_params=True
        :param file_id: is the name of the json file as been given in the Fit class
        :param verbose: The parameters that were saved
        :return: Dictionary containing the saved data
        """
        if file_id[-5:] == ".json":
            file_id = file_id[:-5]
        f = open(f"{file_id}.json")
        data = json.load(f)
        if verbose:
            for key, value in data.items():
                print("{} = {}".format(key, value))
        return data
