from .FittingBaseClass import FittingBaseClass
from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt


class Cosine(FittingBaseClass):
    def __init__(
        self,
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ):
        """
        Create a fit to cosine

        .. math::
        f(x) = amp * cos(2 * pi * f + phase) + offset

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

        super().__init__(x_data, y_data, guess, verbose, plot, save)

        self.offset = None
        self.guess_amp = None
        self.guess_freq = None
        self.guess_phase = None

        self.generate_initial_params()

        if self.guess is not None:
            self.load_guesses(self.guess)

        if verbose:
            self.print_initial_guesses()

        self.fit_data(p0=[1, 1, 1, self.guess_phase])

        self.generate_out_dictionary()

        if verbose:
            self.print_fit_results()

        if plot:
            self.plot_fn()

        if save:
            self.save()

    def generate_initial_params(self):
        # Compute the FFT for guessing the frequency
        fft = np.fft.fft(self.y)
        f = np.fft.fftfreq(len(self.x))
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
        self.guess_freq = out_freq / (self.x[1] - self.x[0])

        # Finding a guess for the offsets
        self.offset = np.mean(self.y)

        # Finding a guess for the phase
        self.guess_phase = np.angle(fft[np.argmax(np.abs(fft))]) - self.guess_freq * 2 * np.pi * self.x[0]

        self.guess_amp = (np.max(self.y) - np.min(self.y)) / 2

    def load_guesses(self, guess_dict):

        for key, guess in guess_dict.items():

            if key == "f":
                self.guess_freq = float(guess) * self.x_normal
            elif key == "phase":
                self.guess_phase = float(guess)
            elif key == "amp":
                self.guess_amp = float(guess) / self.y_normal
            elif key == "offset":
                self.offset = float(guess) / self.y_normal
            else:
                raise Exception(
                    f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                )

    def func(self, x_var, a0, a1, a2, a3):

        return (a0 * self.offset) + a1 * self.guess_amp * np.cos(2 * np.pi * a2 * self.guess_freq * self.x + a3)

    def generate_out_dictionary(self):
        # Output the fitting function and its parameters
        self.out = {
            "fit_func": lambda x_var: self.fit_type(x_var / self.x_normal, self.popt) * self.y_normal,
            "f": [self.popt[2] * self.guess_freq / self.x_normal, self.perr[2] * self.guess_freq / self.x_normal],
            "phase": [self.popt[3] % (2 * np.pi), self.perr[3] % (2 * np.pi)],
            "amp": [self.popt[1] * self.guess_amp * self.y_normal, self.guess_amp * self.perr[1] * self.y_normal],
            "offset": [
                self.popt[0] * self.offset * self.y_normal,
                self.perr[0] * self.offset * self.y_normal,
            ],
        }

    def print_initial_guesses(self):
        print(
            f"Initial guess:\n"
            f" f = {self.guess_freq / self.x_normal:.3f}, \n"
            f" phase = {self.guess_phase:.3f}, \n"
            f" amp = {self.guess_amp * self.y_normal:.3f}, \n"
            f" offset = {self.offset * self.y_normal:.3f}, \n"
        )

    def print_fit_results(self):

        out = self.out

        print(
            f"Fitting results:\n"
            f" f = {out['f'][0]:.3f} +/- {out['f'][1]:.3f}, \n"
            f" phase = {out['phase'][0]:.3f} +/- {out['phase'][1]:.3f} rad, \n"
            f" amp = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f} a.u., \n"
            f" offset = {out['offset'][0]:.2f} +/- {out['offset'][1]:.3f}, \n"
        )

    def plot_fn(self):
        plt.plot(self.x_data, self.fit_type(self.x, self.popt) * self.y_normal)
        plt.plot(
            self.x_data,
            self.y_data,
            ".",
        )
        plt.legend(loc="upper right")
