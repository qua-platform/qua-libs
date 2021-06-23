import scipy.signal as sig
import numpy as np
from qm.qua import *
import matplotlib.pyplot as plt
from qm.QuantumMachinesManager import (
    SimulationConfig,
    QuantumMachinesManager,
    LoopbackInterface,
)

ntaps = 40
delays = [0, 12, 12.25, 12.35]


def delay_gaussian(delay):
    def get_coefficents(delay):
        return np.sinc(np.linspace(0, ntaps, ntaps + 1)[0:-1] - delay)

    qmm = QuantumMachinesManager()

    with program() as filter_delay:
        play("gaussian", "flux1")

    pulse_len = 60

    feedforward_filter = get_coefficents(delay)
    print("feedforward taps:", feedforward_filter)
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opx1",
                "analog_outputs": {
                    1: {"offset": +0.0, "filter": {"feedforward": feedforward_filter}},
                },
            },
        },
        "elements": {
            "flux1": {
                "singleInput": {"port": ("con1", 1)},
                "intermediate_frequency": 10,
                "operations": {
                    "gaussian": "gaussian_pulse",
                },
            },
        },
        "pulses": {
            "gaussian_pulse": {
                "operation": "control",
                "length": pulse_len,
                "waveforms": {"single": "gaussian_wf"},
            },
        },
        "waveforms": {
            "gaussian_wf": {
                "type": "arbitrary",
                "samples": 0.25 * sig.gaussian(pulse_len, 5),
            },
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

    job = qmm.simulate(
        config,
        filter_delay,
        SimulationConfig(duration=150, include_analog_waveforms=True),
    )
    job.result_handles.wait_for_all_values()
    job.get_simulated_samples().con1.plot()


for delay in delays:
    delay_gaussian(delay)
plt.legend(delays)
plt.axis([270, 330, -0.01, 0.26])
