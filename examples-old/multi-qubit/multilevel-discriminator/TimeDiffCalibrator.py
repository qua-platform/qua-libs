import numpy as np
from qm import SimulationConfig, LoopbackInterface
from qm.qua import *


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
                    "mixInputs": {
                        "I": (con_name, 1),
                        "Q": (con_name, 2),
                        "lo_frequency": 0,
                        "mixer": "mixer",
                    },
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
                        "integW1": "integW1",
                        "integW2": "integW2",
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
                "integW1": {
                    "cosine": [1.0] * int(1000 / 4),
                    "sine": [0.0] * int(1000 / 4),
                },
                "integW2": {
                    "cosine": [0.0] * int(1000 / 4),
                    "sine": [1.0] * int(1000 / 4),
                },
            },
            "mixers": {
                "mixer": [
                    {
                        "intermediate_frequency": freq,
                        "lo_frequency": 0,
                        "correction": [1, 0, 0, 1],
                    }
                ]
            },
        }

    @staticmethod
    def calibrate(qmm, con_name):
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
                reset_phase("rr")
                measure(
                    "readout",
                    "rr",
                    "adc",
                    demod.full("integW1", I1, "out1"),
                    demod.full("integW2", Q1, "out1"),
                    demod.full("integW1", I2, "out2"),
                    demod.full("integW2", Q2, "out2"),
                )
                assign(I, I1 + Q2)
                assign(Q, -Q1 + I2)
                save(I, "I")
                save(Q, "Q")
            with stream_processing():
                adc.input1().save_all("adc_input1")
                adc.input2().save_all("adc_input2")
        freq = 8.4567e3
        qm = qmm.open_qm(TimeDiffCalibrator._default_config(freq, con_name))
        job = qm.simulate(
            cal_phase,
            SimulationConfig(
                500,
                simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con2", 2, "con2", 2)]),
            ),
        )

        adc1 = np.mean(job.result_handles.adc_input1.fetch_all()["value"], axis=0)
        adc2 = np.mean(job.result_handles.adc_input2.fetch_all()["value"], axis=0)
        # adc_ts = job.result_handles.adc_input1.fetch_all()['timestamp']
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
