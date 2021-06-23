from typing import List

import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import SimulationConfig, QuantumMachinesManager, LoopbackInterface
import numpy as np
import scipy.signal as signal
import cma

qmm = QuantumMachinesManager()

with program() as filter_optimization:
    stream = declare_stream(adc_trace=True)
    measure("readoutOp", "flux1", stream)
    with stream_processing():
        stream.input1().save("adc")

pulse_len = 128
tof = 248

waveform = [0.] * 30 + [0.2] * (pulse_len - 60) + [0.] * 30

# We use an  arbitrarily selected filter for distorting the signal
distorted_waveform = signal.lfilter(np.array([1]),
                                    np.array([0.95, -0.15, 0.1]), waveform)

plt.plot(waveform)
plt.plot(distorted_waveform)

plt.figure()


def cost(params: List[float]):
    """
    params:
    """
    M = 0  # number of feedback taps 0, 1, 2.
    feedback_filter = np.array(params[:M])
    feedforward_filter = np.array(params[M:])
    print("feedback:", feedback_filter)
    print("feedforward:", feedforward_filter)
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opx1",
                "analog_outputs": {
                    1: {"offset": +0.0, "filter": {"feedback": feedback_filter, "feedforward": feedforward_filter}},
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

    job = qmm.simulate(config, filter_optimization,
                       SimulationConfig(duration=150,
                                        simulation_interface=LoopbackInterface([('con1', 1, 'con1', 1)], latency=200)
                                        )
                       )
    job.result_handles.wait_for_all_values()
    corrected_signal = -job.result_handles.adc.fetch_all() / 4096

    if True:
        plt.plot(waveform)
        plt.plot(distorted_waveform)
        plt.plot(corrected_signal * np.sum(waveform) / np.sum(corrected_signal), '--')
        plt.legend(['Target waveform', 'Distorted waveform', 'Corrected signal'])

    corr = np.correlate(corrected_signal, waveform, 'full') / \
           (np.sqrt(np.correlate(corrected_signal, corrected_signal) * np.correlate(waveform, waveform)))
    loss = 1 - np.max(corr)

    print("loss:", loss)
    return loss


param_number = 5
iterations = 15

es = cma.CMAEvolutionStrategy(np.random.rand(param_number), 1, {'bounds': [-1, 1]})
es.optimize(cost, iterations=iterations)
plt.figure()
cost(es.result_pretty().xbest)
plt.show()
