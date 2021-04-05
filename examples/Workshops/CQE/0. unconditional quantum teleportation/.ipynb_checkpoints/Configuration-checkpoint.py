config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0}, #a-reset
                2: {"offset": +0.0}, #a-RO
                3: {"offset": +0.0}, #a-init
                4: {"offset": +0.0}, #a-espin manip I
                5: {"offset": +0.0}, #a-espin manip Q
                6: {"offset": +0.0}, #n-spin manip
                7: {"offset": +0.0}, #b-RO
                8: {"offset": +0.0}, #b-Reset
                
                

            },
            "digital_outputs": {1: {}, 2: {},3: {}, 4: {},5: {}, 6: {}},
        }
    },
    "elements": {
        "a-reset": {
            "singleInput": {"port": ("con1", 1)},
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
        "a-ro": {
        "singleInput": {"port": ("con1", 1)},
        "digitalInputs": {
            "digital_input1": {
                "port": ("con1", 2),
                "delay": 0,
                "buffer": 0,
            },
        },
        "intermediate_frequency": 0,
        "operations": {
            "playOp": "constPulse",
        
        },
        
    },
    },
    "pulses": {
        "zeroPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
        },
        "constPulse": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "constPulse_trig": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "trig",
        },
        "constPulse_stutter": {
            "operation": "control",
            "length": 1000,  # in ns
            "waveforms": {"single": "const_wf"},
            "digital_marker": "stutter",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "trig": {"samples": [(1, 100)]},
        "stutter": {"samples": [(1, 100), (0, 200), (1, 76), (0, 10), (1, 0)]},
    },
}