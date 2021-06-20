from typing import List

import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import SimulationConfig, QuantumMachinesManager, LoopbackInterface
import numpy as np
import scipy.signal as signal
import cma

qmm = QuantumMachinesManager()

pulse_len = 128
tof = 252

waveform = [0.] * 30 + [0.2] * (pulse_len - 60) + [0.] * 30
A = 1

# We use an  arbitrarily selected filter for distorting the signal

distorted_waveform = signal.lfilter(np.array([1]),
                                    np.array([0.25, -0.15, 0.1]), waveform)
# to keep waveform in DAC range
distorted_waveform = distorted_waveform / np.max(np.abs(distorted_waveform)) / (max(distorted_waveform) / 3)

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "flux1": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 10,
            "operations": {
                "readoutOp": "readoutPulse",
            },
            "time_of_flight": tof,
            "smearing": 0,
        },
    },
    "pulses": {
        "readoutPulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
            "integration_weights": {"x": "xWeights", "y": "yWeights"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "arbitrary", "samples": distorted_waveform},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "xWeights": {
            "cosine": [1.0] * (pulse_len // 4),
            "sine": [0.0] * (pulse_len // 4),
        },
        "yWeights": {
            "cosine": [0.0] * (pulse_len // 4),
            "sine": [1.0] * (pulse_len // 4),
        },
    },
}


def single_freq_response(freq):
    s = declare_stream(adc_trace=True)
    update_frequency('flux1',freq)
    measure("readoutOp", "flux1", s)
    return s


with program() as filter_optimization:
    stream = single_freq_response(1e6)
    with stream_processing():
        stream.input1().fft()
