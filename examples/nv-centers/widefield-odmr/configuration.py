init_time = 10e3
pi_time = 48
NV_LO = 2.87e9
AOM_RF = 80e6

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # I
                2: {"offset": +0.0},  # Q
                3: {"offset": +0.0},  # Fluorescence AOM
            },
            "digital_outputs": {
                1: {},  # Camera Trigger
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # SPCM 1
                2: {"offset": +0.0},  # SPCM 2
            },
        }
    },
    "elements": {
        "NV": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": NV_LO,
                "mixer": "mixer_NV",
            },
            "intermediate_frequency": 1e6,
            "operations": {
                "CW": "CWPulse",
                "pi": "piPulse",
            },
        },
        "AOM": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": AOM_RF,  # AOM Freq
            "operations": {
                "init": "initPulse",
            },
        },
        "camera": {
            "digitalInputs": {
                "TrigIn": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                }
            },
            "operations": {
                "trigger": "triggerPulse",
            },
        },
        "spcm1": {
            "singleInput": {"port": ("con1", 3)},
            "operations": {
                "readout": "readoutPulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": 28,
            "smearing": 0,
            "outputPulseParameters": {
                "signalThreshold": 500,
                "signalPolarity": "Ascending",
                "derivativeThreshold": 100,
                "derivativePolarity": "Ascending",
            },
        },
        "spcm2": {
            "singleInput": {"port": ("con1", 3)},
            "operations": {
                "readout": "readoutPulse",
            },
            "outputs": {"out1": ("con1", 2)},
            "time_of_flight": 28,
            "smearing": 0,
            "outputPulseParameters": {
                "signalThreshold": 500,
                "signalPolarity": "Ascending",
                "derivativeThreshold": 100,
                "derivativePolarity": "Ascending",
            },
        },
    },
    "pulses": {
        "CWPulse": {
            "operation": "control",
            "length": init_time,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "piPulse": {
            "operation": "control",
            "length": pi_time,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "initPulse": {
            "operation": "control",
            "length": init_time,
            "waveforms": {"single": "const_wf"},
        },
        "triggerPulse": {
            "operation": "control",
            "length": init_time,
            "digital_marker": "triggerWF",
        },
        "readoutPulse": {
            "operation": "control",
            "length": init_time,
            "waveforms": {"single": "zero_wf"},
        },
    },
    "digital_waveforms": {
        "triggerWF": {"samples": [(1, 0)]},
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        # 'gauss_wf': {
        #     'type': 'arbitrary',
        #     'samples': gauss_pulse
        # },
        # 'pi_wf': {
        #     'type': 'arbitrary',
        #     'samples': gauss(0.2, 0, 12, pulse_len)
        # },
        # '-pi/2_wf': {
        #     'type': 'arbitrary',
        #     'samples': gauss(-0.1, 0, 12, pulse_len)
        # },
        # 'pi/2_wf': {
        #     'type': 'arbitrary',
        #     'samples': gauss(0.1, 0, 12, pulse_len)
        # },
        "zero_wf": {"type": "constant", "sample": 0},
    },
    "mixers": {
        "mixer_NV": [
            {
                "intermediate_frequency": 1e6,
                "lo_frequency": NV_LO,
                "correction": [1, 0, 0, 1],
            }
        ],
    },
}
