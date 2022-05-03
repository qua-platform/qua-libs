import numpy as np
from qm.qua import *
from qm.qua.math import argmax
import os
import matplotlib.pyplot as plt
from scipy import signal

from TimeDiffCalibrator import TimeDiffCalibrator


class StateDiscriminator:
    """
    The state discriminator is a class that generates optimized measure procedure for state discrimination
    of a multi-level qubit.
    .. note:
        The setup assumed here includes IQ mixer both in the up-
        and down-conversion of the readout pulse.
    """

    def __init__(self, qmm, config, rr_qe, num_of_states, path):
        """
        Constructor for the state discriminator class.
        :param qmm: A QuantumMachineManager object
        :param config: A quantum machine configuration dictionary with the readout resonator element (must be mixInputs
        and has 2 outputs).
        :param rr_qe: A string with the name of the readout resonator element (as specified in the config)
        :param path: A path to save optimized parameters, namely, integration weights and bias for each state. This file
        is generated during training, and it is used during the subsequent measure_state procedure.
        """

        self.qmm = qmm
        self.config = config
        self.rr_qe = rr_qe
        self.num_of_states = num_of_states
        self.path = path
        self.saved_data = None
        self.time_diff = None
        self._load_file(path)

    def _load_file(self, path):
        if os.path.isfile(path):
            self.saved_data = np.load(path)
            self._update_config()

    def _get_qe_freq(self, qe):
        return self.config["elements"][qe]["intermediate_frequency"]

    def _downconvert(self, qe, x, ts):
        """
        Down-convert the input signal from qe
        :param qe: quantum element, origin of signal
        :param x: input analog signal
        :param ts: timestamps
        :return:
        """
        if self.time_diff is None:
            """
            There's a time difference between the reception of the analog input signal,
            and the moment that the signal is written to memory (when the timestamps are created).
            This time difference needs to be accounted for in order to digitally down-convert correctly
            """
            self.time_diff = TimeDiffCalibrator.calibrate(self.qmm, list(self.config["controllers"].keys())[0])
        rr_freq = self._get_qe_freq(qe)
        sig = x * np.exp(-1j * 2 * np.pi * rr_freq * 1e-9 * (ts - self.time_diff))
        return sig

    def _get_traces(self, qe, seq0, sig, use_hann_filter):
        """
        Get the measured waveforms from the resonator in each of the states.
        Need to select the median due to the gaussian noise and apply a LPF
        :param qe: the quantum element being measured
        :param seq0: indexes for the different states
        :param sig: the measured waveform
        :param use_hann_filter:
        :return:
        """
        traces = np.array(
            [
                np.median(np.real(sig[seq0[i] : seq0[i + 1], :]), axis=0)
                + 1j * np.median(np.imag(sig[seq0[i] : seq0[i + 1], :]), axis=0)
                for i in range(self.num_of_states)
            ]
        )

        if use_hann_filter:
            rr_freq = self._get_qe_freq(qe)
            period_ns = int(1 / rr_freq * 1e9)
            hann = signal.hann(period_ns * 2, sym=True)
            hann = hann / np.sum(hann)
            traces = np.array([np.convolve(traces[i, :], hann, "same") for i in range(self.num_of_states)])
        return traces

    @staticmethod
    def _quantize_traces(traces):
        """
        Convert input waveform that comes in on a 1ns scale to an averaged waveform on a 4ns scale,
        as saved digitally by the OPX
        :param traces: the down-converted waveforms
        :return:
        """
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
        Q_res = res_handles.get("Q").fetch_all()["value"]

        if I_res.shape != Q_res.shape:
            raise RuntimeError("")

        ts = res_handles.adc_input1.fetch_all()["timestamp"].reshape((len(I_res), -1))
        in1 = res_handles.adc_input1.fetch_all()["value"].reshape((len(I_res), -1))
        in2 = res_handles.adc_input2.fetch_all()["value"].reshape((len(I_res), -1))
        return I_res, Q_res, ts, in1 + 1j * in2

    def train(self, program, use_hann_filter=True, plot=False, **execute_args):
        """
        The train procedure is used to calibrate the optimal weights and bias for each state. A file with the optimal
        parameters is generated during training, and it is used during the subsequent measure_state procedure.
        :param program: The program should generate equal number of training sets for each one of the qubit states.
        One first prepares the qubit in one of the states, and then measures the readout resonator element. The measure
        command must include saving the raw data (the tag must be called "adc") and the final complex demodulation
        results (which is constructed from 4 real demodulations) must be saved under the tags "I" and "Q". E.g:

            measure("readout", "rr", "adc", demod.full("integW_cos", I1, "out1"),
                                            demod.full("integW_sin", Q1, "out1"),
                                            demod.full("integW_cos", I2, "out2"),
                                            demod.full("integW_sin", Q2, "out2"))
            assign(I, I1 + Q2)
            assign(Q, -Q1 + I2)
            save(I, 'I')
            save(Q, 'Q')

        :param use_hann_filter: Whether or not to use a LPF on the averaged sampled baseband waveforms.
        :type use_hann_filter: bool
        :param plot: Whether or not to plot some figures for debug purposes.
        :type plot: bool
        """

        I_res, Q_res, ts, x = self._execute_and_fetch(program, **execute_args)

        measures_per_state = len(I_res) // self.num_of_states
        seq0 = [i * measures_per_state for i in range(self.num_of_states + 1)]

        sig = self._downconvert(self.rr_qe, x, ts)
        traces = self._get_traces(self.rr_qe, seq0, sig, use_hann_filter)
        weights = self._quantize_traces(traces)
        norm = np.max(np.abs(weights))

        """
        The weights and biases are calculated in order to optimally perform the Maximum Likelihood estimation of
        the qubit state
        """
        weights = weights / norm
        bias = (np.linalg.norm(weights * norm, axis=1) ** 2) / norm / 2 * (2**-24) * 4

        np.savez(self.path, weights=weights, bias=bias)
        self.saved_data = {"weights": weights, "bias": bias}
        self._update_config()

        if plot:
            plt.figure()
            for i in range(self.num_of_states):
                I_ = I_res[seq0[i] : seq0[i + 1]]
                Q_ = Q_res[seq0[i] : seq0[i + 1]]
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

    def _add_iw_to_all_pulses(self, iw):
        for pulse in self.config["pulses"].values():
            if "integration_weights" not in pulse:
                pulse["integration_weights"] = {}
            pulse["integration_weights"][iw] = iw

    def _update_config(self):
        weights = self.saved_data["weights"]
        for i in range(self.num_of_states):
            self.config["integration_weights"][f"state_{i}_in1"] = {
                "cosine": np.real(weights[i, :]).tolist(),
                "sine": (-np.imag(weights[i, :])).tolist(),
            }
            self._add_iw_to_all_pulses(f"state_{i}_in1")
            self.config["integration_weights"][f"state_{i}_in2"] = {
                "cosine": np.imag(weights[i, :]).tolist(),
                "sine": np.real(weights[i, :]).tolist(),
            }
            self._add_iw_to_all_pulses(f"state_{i}_in2")

    def measure_state(self, pulse, out1, out2, res, adc=None):
        """
        This procedure generates a macro of QUA commands for measuring the readout resonator and discriminating between
        the states of the qubit its states.
        :param pulse: readout pulse name.
        :param out1: output 1 name of the readout resonator (corresponding to the real part of the
         complex IN(t) signal).
        :param out2: output 2 name of the readout resonator (corresponding to the imaginary part
        of the complex IN(t) signal).
        :param res: An integer QUA variable that will receive the discrimination result (0,1,... #states)
        :param adc: (optional) the stream variable which the raw ADC data will be saved and will appear in result
        analysis scope.
        """
        bias = self.saved_data["bias"]

        d1_st = declare(fixed, size=self.num_of_states)
        d2_st = declare(fixed, size=self.num_of_states)

        st = declare(fixed, size=self.num_of_states)

        measure(
            pulse,
            self.rr_qe,
            adc,
            *[demod.full(f"state_{str(i)}_in1", d1_st[i], out1) for i in range(self.num_of_states)],
            *[demod.full(f"state_{str(i)}_in2", d2_st[i], out2) for i in range(self.num_of_states)],
        )

        for i in range(self.num_of_states):
            assign(st[i], d1_st[i] + d2_st[i] - bias[i])

        assign(res, argmax(st))
