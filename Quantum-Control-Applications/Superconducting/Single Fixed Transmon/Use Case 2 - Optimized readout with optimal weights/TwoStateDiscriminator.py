from StateDiscriminator import StateDiscriminator
import numpy as np
from qm.qua import *

from pandas import DataFrame
from sklearn import mixture
import matplotlib.pyplot as plt


class TwoStateDiscriminator(StateDiscriminator):
    def __init__(self, qmm, config, update_tof, rr_qe, path, meas_len, smearing, lsb):
        super().__init__(qmm, config, update_tof, rr_qe, path, meas_len, smearing, lsb)
        self.num_of_states = 2

    def _update_config(self):
        weights = self.saved_data["weights"]
        smearing = self.saved_data["smearing"]
        meas_len = self.saved_data["meas_len"]
        if self.finish_train == 0:
            self.mu = self.saved_data["mu"].tolist()
            self.sigma = self.saved_data["sigma"].tolist()
        b_vec = weights[0, :] - weights[1, :]
        # create empty list where the opt_weights will be put in the current
        # format of int_weights
        w_plus_cos = []
        w_minus_sin = []
        w_plus_sin = []
        w_minus_cos = []

        # assigning integration weights to list of tuples
        for i in range(smearing // 4, (meas_len + smearing) // 4):
            w_plus_cos.append((np.real(b_vec)[i], 4))
            w_minus_sin.append((np.imag(-b_vec)[i], 4))
            w_plus_sin.append((np.imag(b_vec)[i], 4))
            w_minus_cos.append((np.real(-b_vec)[i], 4))

        self.config["integration_weights"][f"opt_cos_{self.rr_qe}"] = {"cosine": w_plus_cos, "sine": w_minus_sin}
        self._add_iw_to_all_pulses(f"opt_cos_{self.rr_qe}")
        self.config["integration_weights"][f"opt_sin_{self.rr_qe}"] = {"cosine": w_plus_sin, "sine": w_plus_cos}
        self._add_iw_to_all_pulses(f"opt_sin_{self.rr_qe}")
        self.config["integration_weights"][f"opt_minus_sin_{self.rr_qe}"] = {"cosine": w_minus_sin, "sine": w_minus_cos}
        self._add_iw_to_all_pulses(f"opt_minus_sin_{self.rr_qe}")
        if self.update_tof or self.finish_train == 1:
            self.config["elements"][self.rr_qe]["time_of_flight"] = (
                self.config["elements"][self.rr_qe]["time_of_flight"] - self.config["elements"][self.rr_qe]["smearing"]
            )
            self.config["elements"][self.rr_qe]["smearing"] = 0

        if self.finish_train == 1:
            self._IQ_mu_sigma(b_vec)

    def _IQ_mu_sigma(self, b_vec):
        out1 = np.real(self.x) * 2**-12
        if not self.lsb:
            out2 = np.imag(self.x) * 2**-12
            sign = 1
        else:
            out2 = -np.imag(self.x) * 2**-12
            sign = -1
        rr_freq = self._get_qe_freq(self.rr_qe)
        cos = np.cos(2 * np.pi * rr_freq * 1e-9 * (self.ts - self.time_diff))
        sin = np.sin(2 * np.pi * rr_freq * 1e-9 * (self.ts - self.time_diff))
        b_vec = np.repeat(b_vec, 4)
        I_res = np.sum(out1 * (cos * np.real(b_vec) + sin * np.imag(-b_vec)), axis=1) + np.sum(
            out2 * (cos * np.imag(b_vec) + sin * np.real(b_vec)) * sign, axis=1
        )
        Q_res = np.sum(out2 * (cos * np.real(b_vec) + sin * np.imag(-b_vec)), axis=1) - np.sum(
            out1 * (cos * np.imag(b_vec) + sin * np.real(b_vec)) * sign, axis=1
        )
        I_res *= 2**-12
        Q_res *= 2**-12
        import matplotlib.pyplot as plt

        plt.figure()
        for i in range(self.num_of_states):
            I_ = I_res[self.seq0 == i]
            Q_ = Q_res[self.seq0 == i]
            data = {"x": I_, "y": Q_}
            x = DataFrame(data, columns=["x", "y"])
            gmm = mixture.GaussianMixture(n_components=1, covariance_type="spherical", tol=1e-12, reg_covar=1e-12).fit(
                x
            )
            self.mu[i] = gmm.means_[0]
            self.sigma[i] = np.sqrt(gmm.covariances_[0])
            theta = np.linspace(0, 2 * np.pi, 100)
            a = self.sigma[i] * np.cos(theta) + self.mu[i][0]
            b = self.sigma[i] * np.sin(theta) + self.mu[i][1]
            plt.plot(I_, Q_, ".", label=f"state {i}")
            plt.plot([self.mu[i][0]], [self.mu[i][1]], "o")
            plt.plot(a, b)
            plt.axis("equal")
        plt.xlabel("I")
        plt.ylabel("Q")
        plt.legend()
        plt.show()
        data = dict(np.load(self.path))
        data["mu"] = self.mu
        data["sigma"] = self.sigma
        np.savez(self.path, **data)

    def plot_sigma_mu(self):
        theta = np.linspace(0, 2 * np.pi, 100)
        for i in range(self.num_of_states):
            a = self.sigma[i] * np.cos(theta) + self.mu[i][0]
            b = self.sigma[i] * np.sin(theta) + self.mu[i][1]
            plt.plot([self.mu[i][0]], [self.mu[i][1]], "o")
            plt.plot(a, b)

    def get_threshold(self):
        bias = self.saved_data["bias"]
        return bias[0] - bias[1]

    def measure_state(self, pulse, out1, out2, res, adc=None, I=None, Q=None):
        """
        This procedure generates a macro of QUA commands for measuring the readout resonator and discriminating between
        the states of the qubit its states.
        :param pulse: A string with the readout pulse name.
        :param out1: A string with the name first output of the readout resonator (corresponding to the real part of the
         complex IN(t_int) signal).
        :param out2: A string with the name second output of the readout resonator (corresponding to the imaginary part
        of the complex IN(t_int) signal).
        :param res: A boolean QUA variable that will receive the discrimination result (0 or 1)
        :param adc: (optional) the stream variable which the raw ADC data will be saved and will appear in result
        analysis scope.
        """
        II = declare(fixed)
        QQ = declare(fixed)

        if not self.lsb:
            Q1_weight, Q2_weight = f"opt_minus_sin_{self.rr_qe}", f"opt_sin_{self.rr_qe}"
        else:
            Q1_weight, Q2_weight = f"opt_sin_{self.rr_qe}", f"opt_minus_sin_{self.rr_qe}"

        if Q is not None:
            measure(
                pulse,
                self.rr_qe,
                adc,
                dual_demod.full(f"opt_cos_{self.rr_qe}", out1, Q2_weight, out2, II),
                dual_demod.full(Q1_weight, out1, f"opt_cos_{self.rr_qe}", out2, QQ),
            )

        else:
            measure(pulse, self.rr_qe, adc, dual_demod.full(f"opt_cos_{self.rr_qe}", out1, Q2_weight, out2, II))

        assign(res, II < self.get_threshold())
        if I is not None:
            assign(I, II)
        if Q is not None:
            assign(Q, QQ)

        return res, I, Q
