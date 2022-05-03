import scipy.signal as sig
import numpy as np
from qm.qua import *
import matplotlib.pyplot as plt
import warnings
from qm.QuantumMachinesManager import (
    SimulationConfig,
    QuantumMachinesManager,
    LoopbackInterface,
)

ntaps = 40
delays = [0, 22, 22.25, 22.35]


def delay_gaussian(delay, ntaps):
    def get_coefficents(delay, ntaps):
        n_extra = 5
        full_coeff = np.sinc(np.linspace(0 - n_extra, ntaps + n_extra, ntaps + 1 + 2 * n_extra)[0:-1] - delay)
        extra_coeff = np.abs(np.concatenate((full_coeff[:n_extra], full_coeff[-n_extra:])))
        if np.any(extra_coeff > 0.02):  # Contribution is more than 2%
            warnings.warn("Contribution from missing coefficients is not negligible.")
        coeff = full_coeff[n_extra:-n_extra]
        return coeff

    qmm = QuantumMachinesManager()

    with program() as filter_delay:
        play("gaussian", "flux1")

    pulse_len = 60

    feedforward_filter = get_coefficents(delay, ntaps)
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
    delay_gaussian(delay, ntaps)

plt.legend(delays)
plt.axis([270, 340, -0.01, 0.26])
