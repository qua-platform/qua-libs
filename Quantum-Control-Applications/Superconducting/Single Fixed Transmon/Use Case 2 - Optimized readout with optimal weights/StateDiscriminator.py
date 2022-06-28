import numpy as np
from pandas import DataFrame
from qm.qua import *
import os
import matplotlib.pyplot as plt
from sklearn import mixture
from scipy import signal

from TimeDiffCalibrator import TimeDiffCalibrator


class StateDiscriminator:
    """
    The state discriminator is a class that generates optimized measure procedure for state discrimination
    of a multi-level qubit.
    .. note:
        Currently only 3-states discrimination is supported. The setup assumed here includes IQ mixer both in the up-
        and down-conversion of the readout pulse.
    """

    def __init__(self, qmm, config, update_tof, rr_qe, path, meas_len, smearing, lsb=False):
        """
        Constructor for the state discriminator class.
        :param qmm: QuantumMachinesManager object
        :param config: A quantum machine configuration dictionary with the readout resonator element (must be mixInputs
        and have 2 outputs).
        :param rr_qe: A string with the name of the readout resonator element (as specified in the config)
        :param path: A path to save optimized parameters, namely, integration weights and bias for each state. This file
        is generated during training, and it is used during the subsequent measure_state procedure.
        :param lsb: defines if the downconversion mixers does a conversion to LO - IF, i.e., lower side band
        """

        self.qmm = qmm
        self.config = config
        self.rr_qe = rr_qe
        self.num_of_states = 3
        self.path = path
        self.saved_data = None
        self.time_diff = None
        self.update_tof = update_tof
        self.finish_train = 0
        self.mu = dict()
        self.sigma = dict()
        self._load_file(path)
        self.lsb = lsb
        self.meas_len = meas_len
        self.smearing = smearing

    def _load_file(self, path):
        if os.path.isfile(path):
            self.saved_data = np.load(path, allow_pickle=True)
            self._update_config()

    def _get_qe_freq(self, qe):
        return self.config["elements"][qe]["intermediate_frequency"]

    def _downconvert(self, qe, x, ts):
        if self.time_diff is None:
            self.time_diff = TimeDiffCalibrator.calibrate(
                self.qmm, list(self.config["controllers"].keys())[0], self._get_qe_freq(qe)
            )
        rr_freq = self._get_qe_freq(qe)
        sig = x * np.exp(-1j * 2 * np.pi * rr_freq * 1e-9 * (ts - self.time_diff))
        return sig

    def _get_traces(self, qe, correction_method, I_res, Q_res, seq0, sig, use_hann_filter):
        if correction_method == "gmm":
            data = {"x": I_res, "y": Q_res}
            x = DataFrame(data, columns=["x", "y"])
            gmm = mixture.GaussianMixture(
                n_components=self.num_of_states, covariance_type="spherical", tol=1e-12, reg_covar=1e-12
            ).fit(x)

            pr_state = gmm.predict(x)
            mapping = [np.argmax(np.bincount(pr_state[seq0 == i])) for i in range(self.num_of_states)]
            traces = np.array([np.mean(sig[pr_state == mapping[i], :], axis=0) for i in range(self.num_of_states)])

        elif correction_method == "none" or correction_method == "robust":
            if correction_method == "none":
                traces = np.array([np.mean(sig[seq0 == i, :], axis=0) for i in range(self.num_of_states)])
            else:
                traces = np.array(
                    [
                        np.median(np.real(sig[seq0 == i, :]), axis=0)
                        + 1j * np.median(np.imag(sig[seq0 == i, :]), axis=0)
                        for i in range(self.num_of_states)
                    ]
                )
        else:
            raise Exception("unknown correction_method")

        if use_hann_filter:
            rr_freq = self._get_qe_freq(qe)
            period_ns = int(1 / rr_freq * 1e9)
            hann = signal.hann(period_ns * 2, sym=True)
            hann = hann / np.sum(hann)
            traces = np.array([np.convolve(traces[i, :], hann, "same") for i in range(self.num_of_states)])
        return traces

    @staticmethod
    def _quantize_traces(traces):
        weights = []
        for i in range(traces.shape[0]):
            weights.append(np.average(np.reshape(traces[i, :], (-1, 4)), axis=1))
        return np.array(weights)

    def _execute_and_fetch(self, program, **execute_args):
        qm = self.qmm.open_qm(self.config)
        job = qm.execute(program, duration_limit=0, data_limit=0, **execute_args)
        res_handles = job.result_handles
        res_handles.wait_for_all_values()
        I_res = res_handles.get("I").fetch_all()["value"]
        I_res = np.concatenate([I_res[0::2], I_res[1::2]])
        Q_res = res_handles.get("Q").fetch_all()["value"]
        Q_res = np.concatenate([Q_res[0::2], Q_res[1::2]])

        if I_res.shape != Q_res.shape:
            raise RuntimeError("")

        ts = res_handles.get("adc1").fetch_all()["value"]["timestamp"]
        ts = np.concatenate([ts[0::2], ts[1::2]])
        in1 = res_handles.get("adc1").fetch_all()["value"]["value"]
        in1 = np.concatenate([in1[0::2], in1[1::2]])
        in2 = res_handles.get("adc2").fetch_all()["value"]
        in2 = np.concatenate([in2[0::2], in2[1::2]])
        if not self.lsb:
            x = in1 + 1j * in2
        else:
            x = in1 - 1j * in2
        return I_res, Q_res, ts, x

    def train(self, program, use_hann_filter=False, plot=False, correction_method="robust", **execute_args):
        """
        The train procedure is used to calibrate the optimal weights and bias for each state. A file with the optimal
        parameters is generated during training, and it is used during the subsequent measure_state procedure.
        A training program must be provided in the constructor.
        :param program: a training program. A program that generates training sets. The program should generate equal
        number of training sets for each one of the states. Collection of training sets is achieved by first preparing
        the qubit in one of the states, and then measure the readout resonator element. The measure command must include
        streaming of the raw data (the tag must be called "adc") and the final complex demodulation results (which is
        constructed from 2 dual demodulations) must be saved under the tags "I" and "Q". E.g:

            measure("readout", "rr", "adc", dual_demod.full('cos', 'out1', 'sin', 'out2', I),
                                            dual_demod.full('minus_sin', 'out1', 'cos', 'out2', Q))

        :param use_hann_filter: Whether or not to use a LPF on the averaged sampled baseband waveforms.
        :type bool.
        :param plot: Whether or not to plot some figures for debug purposes.
        :type bool
        :param correction_method: it is possible to use 'gmm', 'robust', or 'none'
        """

        I_res, Q_res, self.ts, self.x = self._execute_and_fetch(program, **execute_args)

        measures_per_state = len(I_res) // self.num_of_states
        self.seq0 = np.array([[i] * measures_per_state for i in range(self.num_of_states)]).flatten()

        sig = self._downconvert(self.rr_qe, self.x, self.ts)
        traces = self._get_traces(self.rr_qe, correction_method, I_res, Q_res, self.seq0, sig, use_hann_filter)
        weights = self._quantize_traces(traces)

        norm = np.max(np.abs(weights))
        weights = weights / norm
        bias = (np.linalg.norm(weights * norm, axis=1) ** 2) / norm / 2 * (2**-24) * 4

        np.savez(
            self.path,
            weights=weights,
            bias=bias,
            smearing=self.smearing,
            meas_len=self.meas_len,
        )
        self.saved_data = {
            "weights": weights,
            "bias": bias,
            "smearing": self.smearing,
            "meas_len": self.meas_len,
        }
        self.finish_train = 1
        self._update_config()

        if plot:
            plt.figure()
            for i in range(self.num_of_states):
                I_ = I_res[self.seq0 == i]
                Q_ = Q_res[self.seq0 == i]
                plt.plot(I_, Q_, ".", label=f"state {i}")
                plt.axis("equal")
            plt.xlabel("I")
            plt.ylabel("Q")
            plt.legend()

            plt.figure()
            for i in range(self.num_of_states):
                plt.subplot(self.num_of_states, 1, i + 1)
                plt.plot(np.real(weights[i, :]))
                plt.plot(np.imag(weights[i, :]))

            plt.figure()
            for i in range(self.num_of_states):
                plt.plot(np.real(weights[i, :]), np.imag(weights[i, :]))
                plt.axis("equal")

    def _add_iw_to_all_pulses(self, iw):
        for pulse in self.config["pulses"].values():
            if "integration_weights" not in pulse:
                pulse["integration_weights"] = {}
            pulse["integration_weights"][iw] = iw

    def _update_config(self):
        weights = self.saved_data["weights"]
        for i in range(self.num_of_states):
            self.config["integration_weights"][f"state_{i}_in1_{self.rr_qe}"] = {
                "cosine": np.real(weights[i, :]).tolist(),
                "sine": (-np.imag(weights[i, :])).tolist(),
            }
            self._add_iw_to_all_pulses(f"state_{i}_in1_{self.rr_qe}")
            self.config["integration_weights"][f"state_{i}_in2_{self.rr_qe}"] = {
                "cosine": np.imag(weights[i, :]).tolist(),
                "sine": np.real(weights[i, :]).tolist(),
            }
            self._add_iw_to_all_pulses(f"state_{i}_in2_{self.rr_qe}")
            if self.update_tof or self.finish_train == 1:
                self.config["elements"][self.rr_qe]["time_of_flight"] = (
                    self.config["elements"][self.rr_qe]["time_of_flight"]
                    - self.config["elements"][self.rr_qe]["smearing"]
                )

    def measure_state(self, pulse, out1, out2, res, adc=None):
        """
        This procedure generates a macro of QUA commands for measuring the readout resonator and discriminating between
        the states of the qubit its states.
        :param pulse: A string with the readout pulse name.
        :param out1: A string with the name first output of the readout resonator (corresponding to the real part of the
         complex IN(t_int) signal).
        :param out2: A string with the name second output of the readout resonator (corresponding to the imaginary part
        of the complex IN(t_int) signal).
        :param res: An integer QUA variable that will receive the discrimination result (0,1 or 2)
        :param adc: (optional) the stream variable which the raw ADC data will be saved and will appear in result
        analysis scope.
        """
        bias = self.saved_data["bias"]
        # currently it allows only 3 states

        d1_st0 = declare(fixed)
        d2_st0 = declare(fixed)
        d1_st1 = declare(fixed)
        d2_st1 = declare(fixed)
        d1_st2 = declare(fixed)
        d2_st2 = declare(fixed)

        st0 = declare(fixed)
        st1 = declare(fixed)
        st2 = declare(fixed)

        measure(
            pulse,
            self.rr_qe,
            adc,
            demod.full(f"state_0_in1_{self.rr_qe}", d1_st0, out1),
            demod.full(f"state_0_in2_{self.rr_qe}", d2_st0, out2),
            demod.full(f"state_1_in1_{self.rr_qe}", d1_st1, out1),
            demod.full(f"state_1_in2_{self.rr_qe}", d2_st1, out2),
            demod.full(f"state_2_in1_{self.rr_qe}", d1_st2, out1),
            demod.full(f"state_2_in2_{self.rr_qe}", d2_st2, out2),
        )

        assign(st0, d1_st0 + d2_st0 - bias[0])
        assign(st1, d1_st1 + d2_st1 - bias[1])
        assign(st2, d1_st2 + d2_st2 - bias[2])

        with if_((st0 >= st1) & (st0 >= st2)):
            assign(res, 0)
        with if_((st1 >= st0) & (st1 >= st2)):
            assign(res, 1)
        with if_((st2 >= st0) & (st2 >= st1)):
            assign(res, 2)
