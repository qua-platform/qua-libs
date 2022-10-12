readout_len = 10e6
if_freq = 100e6
lo_freq=6e9
calibration_amp = 0.125
calibration_pulse_length = 10e3
time_of_flight = 192
offset_amp = 2 ** -3  # (0.125)

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "mixInputs": {
                    "I": ("con1", 1),
                    "Q": ("con1", 2),
                    "lo_frequency": lo_freq,
                    "mixer": "octave_octave1_1",
                },
                "intermediate_frequency": if_freq,
                "operations": {
                    "readout": "calibration_pulse",
                    "calibration_long": "long_calibration_pulse",
                },
                "digitalInputs": {},
        },
    },
    "pulses": {
            "calibration_pulse": {
                "operation": "control",
                "length": calibration_pulse_length,
                "waveforms": {
                    "I": "readout_wf",
                    "Q": "zero_wf",
                },
            },
            "long_calibration_pulse": {
                "operation": "control",
                "length": calibration_pulse_length * 100,
                "waveforms": {
                    "I": "readout_wf",
                    "Q": "zero_wf",
                },
                # "digital_marker": "ON",
            },
            "DC_offset_pulse": {
                "operation": "control",
                "length": calibration_pulse_length,
                "waveforms": {"single": "DC_offset_wf"},
            },
            "Analyze_pulse": {
                "operation": "measurement",
                "length": calibration_pulse_length,
                "waveforms": {
                    "I": "zero_wf",
                    "Q": "zero_wf",
                },
                "integration_weights": {
                    "cos": "cosine_weights",
                    "sin": "sine_weights",
                    "minus_sin": "minus_sine_weights",
                },
                "digital_marker": "ON",
            },
        },
        "waveforms": {
            "readout_wf": {
                "type": "constant",
                "sample": calibration_amp,
            },
            "zero_wf": {
                "type": "constant",
                "sample": 0.0,
            },
            "DC_offset_wf": {"type": "constant", "sample": offset_amp},
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
            "OFF": {"samples": [(0, 0)]},
        },
        "integration_weights": {
            "cosine_weights": {
                "cosine": [(1.0, calibration_pulse_length)],
                "sine": [(0.0, calibration_pulse_length)],
            },
            "sine_weights": {
                "cosine": [(0.0, calibration_pulse_length)],
                "sine": [(1.0, calibration_pulse_length)],
            },
            "minus_sine_weights": {
                "cosine": [(0.0, calibration_pulse_length)],
                "sine": [(-1.0, calibration_pulse_length)],
            },
        },
"mixers": {
            "octave_octave1_1": [
{
                    "intermediate_frequency": if_freq,
                    "lo_frequency": lo_freq,
                    "correction": [1,0,0,1],
                },
            ],
        },
}