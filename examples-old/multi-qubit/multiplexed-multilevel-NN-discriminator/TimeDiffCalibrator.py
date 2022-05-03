from qm.qua import *
import numpy as np
from copy import deepcopy


class TimeDiffCalibrator:
    @staticmethod
    def _update_config(freq, config, qe):
        config["elements"][qe]["operations"]["time_diff_long_readout"] = "time_diff_long_readout_pulse"
        config["elements"][qe]["intermediate_frequency"] = freq
        config["mixers"][config["elements"][qe]["mixInputs"]["mixer"]][0]["intermediate_frequency"] = freq
        print(
            f"ATTENTION: Using the mixer at the 0'th position of {config['elements'][qe]['mixInputs']['mixer']} to "
            f"calibrate time difference."
        )
        config["pulses"]["time_diff_long_readout_pulse"] = {
            "operation": "measurement",
            "length": int(1e3),
            "waveforms": {
                "I": "time_diff_const_wf",
                "Q": "time_diff_zero_wf",
            },
            "integration_weights": {
                "time_diff_integW1": "time_diff_integW1",
                "time_diff_integW2": "time_diff_integW2",
            },
            "digital_marker": "time_diff_ON",
        }
        config["digital_waveforms"]["time_diff_ON"] = {"samples": [(1, 0)]}
        config["waveforms"]["time_diff_const_wf"] = {"type": "constant", "sample": 0.4}
        config["waveforms"]["time_diff_zero_wf"] = {"type": "constant", "sample": 0.0}
        config["integration_weights"]["time_diff_integW1"] = {
            "cosine": [1.0] * int(1e3 / 4),
            "sine": [0.0] * int(1e3 / 4),
        }

        config["integration_weights"]["time_diff_integW2"] = {
            "cosine": [0.0] * int(1e3 / 4),
            "sine": [1.0] * int(1e3 / 4),
        }

        return config

    @staticmethod
    def calibrate(qmm, config, qe, **execute_args):
        config = deepcopy(config)
        with program() as cal_phase:
            I1 = declare(fixed)
            Q1 = declare(fixed)
            I2 = declare(fixed)
            Q2 = declare(fixed)
            I = declare(fixed)
            Q = declare(fixed)
            n = declare(int)
            adc = declare_stream(adc_trace=True)
            with for_(n, 1, n < 1e4, n + 1):
                reset_phase(qe)
                measure(
                    "time_diff_long_readout",
                    qe,
                    adc,
                    demod.full("time_diff_integW1", I1, "out1"),
                    demod.full("time_diff_integW2", Q1, "out1"),
                    demod.full("time_diff_integW1", I2, "out2"),
                    demod.full("time_diff_integW2", Q2, "out2"),
                )
                assign(I, I1 + Q2)
                assign(Q, -Q1 + I2)
                save(I, "I")
                save(Q, "Q")
            with stream_processing():
                adc.input1().save_all("adc_input1")
                adc.input2().save_all("adc_input2")
        freq = 8.78e3
        qm = qmm.open_qm(TimeDiffCalibrator._update_config(freq, config, qe))
        job = qm.execute(cal_phase, **execute_args)

        job.result_handles.wait_for_all_values()
        adc1 = np.mean(job.result_handles.adc_input1.fetch_all()["value"], axis=0)
        adc2 = np.mean(job.result_handles.adc_input2.fetch_all()["value"], axis=0)
        adc_ts = np.arange(0, len(adc1))

        I = np.mean(job.result_handles.get("I").fetch_all()["value"])
        Q = np.mean(job.result_handles.get("Q").fetch_all()["value"])

        sig = (adc1 + 1j * adc2) * np.exp(-1j * 2 * np.pi * freq * 1e-9 * adc_ts)
        d = np.sum(sig)

        I_ = np.real(d)
        Q_ = np.imag(d)

        time_diff_ns = np.angle((I + 1j * Q) / (I_ + 1j * Q_)) / 1e-9 / 2 / np.pi / freq
        time_diff = np.round(time_diff_ns / 4) * 4

        return time_diff
