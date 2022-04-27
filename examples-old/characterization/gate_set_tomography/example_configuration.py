import numpy as np

pulse_len = 1000


def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma**2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "x_control": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 100e6,
            "operations": {
                "readoutOp": "readoutPulse",
                "x_pi/2": "gaussPulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
        "y_control": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 100e6,
            "operations": {
                "readoutOp": "readoutPulse",
                "y_pi/2": "gaussPulse",
            },
            "time_of_flight": 180,
            "smearing": 0,
        },
        "readout": {
            "singleInput": {"port": ("con1", 3)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 100e6,
            "operations": {
                "readoutOp": "readoutPulse",
                "x_pi/2": "gaussPulse",
            },
            "time_of_flight": 180,
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
        "constPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "gaussPulse": {
            "operation": "control",
            "length": pulse_len,
            "waveforms": {"single": "gauss_wf"},
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.3},
        "gauss_wf": {"type": "arbitrary", "samples": gauss(0.4, 0, 120, 0, pulse_len)},
        "ramp_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0, -0.5, pulse_len).tolist(),
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
        "xWeights2": {
            "cosine": [1.0] * (2 * pulse_len // 4),
            "sine": [0.0] * (2 * pulse_len // 4),
        },
        "yWeights": {
            "cosine": [0.0] * (pulse_len // 4),
            "sine": [1.0] * (pulse_len // 4),
        },
    },
}
