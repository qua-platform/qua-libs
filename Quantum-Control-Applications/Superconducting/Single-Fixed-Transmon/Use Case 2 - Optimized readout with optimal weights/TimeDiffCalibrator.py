import numpy as np
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *


# Todo - Replace by the known delay, maybe also not needed (reset phase)
class TimeDiffCalibrator:
    @staticmethod
    def _default_config(freq, con_name):
        return {
            "version": 1,
            "controllers": {
                con_name: {
                    "type": "opx1",
                    "analog_outputs": {1: {"offset": 0.0}, 2: {"offset": 0.0}},
                    "digital_outputs": {},
                    "analog_inputs": {1: {"offset": 0.0}, 2: {"offset": 0.0}},
                }
            },
            "elements": {
                "rr": {
                    "mixInputs": {"I": (con_name, 1), "Q": (con_name, 2), "lo_frequency": 0, "mixer": "mixer"},
                    "intermediate_frequency": freq,
                    "operations": {
                        "readout": "readout",
                    },
                    "outputs": {"out1": (con_name, 1), "out2": (con_name, 2)},
                    "time_of_flight": 32,
                    "smearing": 0,
                },
            },
            "pulses": {
                "readout": {
                    "operation": "measurement",
                    "length": 1000,
                    "waveforms": {
                        "I": "const_wf",
                        "Q": "zero_wf",
                    },
                    "integration_weights": {
                        "cos": "cos_weights",
                        "sin": "sin_weights",
                        "minus_sin": "minus_sin_weights",
                    },
                    "digital_marker": "ON",
                },
            },
            "waveforms": {
                "const_wf": {"type": "constant", "sample": 0.4},
                "zero_wf": {"type": "constant", "sample": 0.0},
            },
            "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
            "integration_weights": {
                "cos_weights": {
                    "cosine": [(1.0, 1000)],  # Previous format for versions before 1.20: [1.0] * readout_len
                    "sine": [(0.0, 1000)],
                },
                "sin_weights": {
                    "cosine": [(0.0, 1000)],
                    "sine": [(1.0, 1000)],
                },
                "minus_sin_weights": {
                    "cosine": [(0.0, 1000)],
                    "sine": [(-1.0, 1000)],
                },
            },
            "mixers": {"mixer": [{"intermediate_frequency": freq, "lo_frequency": 0, "correction": [1, 0, 0, 1]}]},
        }

    @staticmethod
    def calibrate(qmm, con_name, res_freq):
        with program() as cal_phase:
            I = declare(fixed)
            Q = declare(fixed)

            measure(
                "readout",
                "rr",
                "adc",
                dual_demod.full("cos", "sin", I),
                dual_demod.full("minus_sin", "cos", Q),
            )
            save(I, "I")
            save(Q, "Q")

        freq = res_freq

        qm = qmm.open_qm(TimeDiffCalibrator._default_config(freq, con_name))

        job = qm.execute(cal_phase)

        job.result_handles.wait_for_all_values()
        adc1 = job.result_handles.adc_input1.fetch_all()["value"]
        adc2 = job.result_handles.adc_input2.fetch_all()["value"]
        adc_ts = job.result_handles.adc_input1.fetch_all()["timestamp"]

        I = job.result_handles.get("I").fetch_all()["value"][0]
        Q = job.result_handles.get("Q").fetch_all()["value"][0]

        sig = (adc1 + 1j * adc2) * np.exp(-1j * 2 * np.pi * freq * 1e-9 * adc_ts)
        d = np.sum(sig)

        I_ = np.real(d)
        Q_ = np.imag(d)

        time_diff_ns = np.angle((I + 1j * Q) / (I_ + 1j * Q_)) / 1e-9 / 2 / np.pi / freq
        time_diff = np.round(time_diff_ns / 4) * 4
        return time_diff_ns  # measured at the resonator freq in execute mode not simulation
