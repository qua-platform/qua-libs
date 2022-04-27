import numpy as np

gauss_pulse_len = 20  # ns
const_pulse_len = 500
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg**2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)
readout_pulse_len = 20
omega_10 = 4.958e9
omega_d = omega_10 / 3
SFQ_IF = 50e6
SFQ_LO = 5.10e9
qubit_IF = 200e6
qubit_LO = 5.10e9
pi_pulse_len = 32
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # qubitI
                2: {"offset": +0.0},  # qubitQ
                3: {"offset": +0.0},  # RR I
                4: {"offset": +0.0},  # RR Q
                5: {"offset": +0.0},  # SFQ_bias
                6: {"offset": +0.0},  # SFQ_trig_I
                7: {"offset": +0.0},  # SFQ_trig_Q
                8: {"offset": +0.0},  # qubit_bias
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
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": qubit_IF,  # ω_d,
            "operations": {
                "gauss_pulse": "gauss_pulse_in",  # to a pulse
                "pi_pulse": "pi_pulse_in",
            },
        },
        "SFQ_trigger": {
            "mixInputs": {
                "I": ("con1", 6),
                "Q": ("con1", 7),
                "lo_frequency": SFQ_LO,
                "mixer": "mixer_SFQ",
            },
            "intermediate_frequency": SFQ_IF,  # ω_d,
            "operations": {
                "const_pulse": "const_pulse_in",  # to a pulse
                "pi_pulse": "pi_pulse_in",
                "pi2_pulse": "pi2_pulse_in",
            },
        },
        "qubit_flux_bias": {
            "singleInput": {"port": ("con1", 8)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 0,
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
            "intermediate_frequency": 50e6,  # 6.15e9,
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
            "intermediate_frequency": 0,
            "operations": {
                "playOp": "constPulse",
                "pi_pulse": "pi_pulse",
                "pi2_pulse": "pi_pulse",
            },
        },
    },
    "pulses": {
        "meas_pulse_in": {  # Readout pulse
            "operation": "measurement",
            "length": readout_pulse_len,
            "waveforms": {
                "I": "exc_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
        },
        "constPulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "const_wf"},
        },
        "gauss_pulse_in": {
            "operation": "control",
            "length": gauss_pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": pi_pulse_len,  # in ns
            "waveforms": {"single": "const_wf"},
        },
        "pi_pulse_in": {  # Assumed to be calibrated
            "operation": "control",
            "length": pi_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "pi2_pulse_in": {  # Assumed to be calibrated
            "operation": "control",
            "length": pi_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "const_pulse_in": {  # Assumed to be calibrated
            "operation": "control",
            "length": const_pulse_len,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "exc_wf": {"type": "constant", "sample": 0.479},
    },
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {
            "cosine": [4.0] * readout_pulse_len,
            "sine": [0.0] * readout_pulse_len,
        },
        "integW2": {
            "cosine": [0.0] * readout_pulse_len,
            "sine": [4.0] * readout_pulse_len,
        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 50e6,  # 6.15e9,
                "lo_frequency": 6.00e9,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,  # ω_d,
                "lo_frequency": qubit_LO,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_SFQ": [
            {
                "intermediate_frequency": SFQ_IF,  # ω_d,
                "lo_frequency": SFQ_LO,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
