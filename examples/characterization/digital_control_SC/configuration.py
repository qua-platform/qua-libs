import numpy as np

gauss_pulse_len = 20  # ns
const_pulse_len = 100
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg ** 2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)
readout_pulse_len = 20
ω_10 = 4.958e9
ω_d = ω_10/3
I_ref = 110e-6  # 130 µA
R = 1e3  # 1 kΩ
V_ref = R * I_ref  # 130 mV
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
                5: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        },
    },
    "elements": {
        "qubit_control": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": 5.10e9,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 0,  # ω_d,
            "operations": {
                "gauss_pulse": "gauss_pulse_in",  # to a pulse
                "pi_pulse": "pi_pulse_in"
            },
        },
        "SFQ_trigger": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": 5.10e9,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 0,  # ω_d,
            "operations": {
                "const_pulse": "const_pulse_in",  # to a pulse
                "pi_pulse": "pi_pulse_in"
            },
        },
        "qubit_flux_bias": {
            "singleInput": {"port": ("con1", 5)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
        },

        "RR": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": 6.00e9,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 0,  # 6.15e9,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
        "SFQ_bias": {
            "singleInput": {"port": ("con1", 5)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 5e6,
            "operations": {
                "playOp": "constPulse",
            },
        }
    },
    "pulses": {
        "meas_pulse_in": {  # Readout pulse
            "operation": "measurement",
            "length": 200,
            "waveforms": {
                "I": "exc_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "marker1",
        },
        "constPulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "marker1",
        },
        "gauss_pulse_in": {
            "operation": "control",
            "length": gauss_pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "pi_pulse_in": {  # Assumed to be calibrated
            "operation": "control",
            "length": const_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "const_pulse_in": {  # Assumed to be calibrated
            "operation": "control",
            "length": const_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": V_ref},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "exc_wf": {"type": "constant", "sample": 0.479},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {
            "cosine": [4.0] * 28,

            "sine": [0.0] * 28
        },
        "integW2": {
            "cosine": [0.0] * 28,
            "sine": [4.0] * 28
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 0,  # 6.15e9,
                "lo_frequency": 6.00e9,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": 0,  # ω_d,
                "lo_frequency": 5.10e9,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
