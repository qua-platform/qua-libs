import numpy as np

gauss_pulse_len = 20  # ns
const_pulse_len = 100
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg**2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)
readout_time = 100
# omega_10 = 4.958e9
# omega_d = omega_10 / 3

pump_len = 100
prep_len = 100
ramp_len = 20
ramp_max = 0.1

v_prep_top = 0.2
v_prep_bot = -0.2
v_readout_top = 0.2
v_readout_bot = -0.2
v_evolve_top = 0.1
v_evolve_bot = -0.1

ramp_down_prep = np.linspace(v_prep_top, v_evolve_bot, ramp_len).tolist()
ramp_up_prep = np.linspace(v_prep_bot, v_evolve_top, ramp_len).tolist()
ramp_up_readout = np.linspace(v_evolve_bot, v_readout_top, ramp_len).tolist()
ramp_down_readout = np.linspace(v_evolve_top, v_readout_bot, ramp_len).tolist()

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # Vr
                2: {"offset": +0.0},  # Vl
                3: {"offset": +0.0},  # RF-QPC-Out
            },
            "analog_inputs": {
                1: {"offset": +0.0},  # RF-QPC-in
            },
        },
    },
    "elements": {
        "DET": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0e6,
            "operations": {
                "evolve": "constPulse",  # to a pulse
            },
        },
        "D1": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0e6,
            # 'hold_offset':{'duration': 100},
            "operations": {
                "const": "constPulse",
                "prep": "prepPosPulse",  # to a pulse
                "pump": "constPulse",
                # "prep_neg": "constNegPulse",  # to a pulse
                "evolve": "evolveD1Pulse",  # to a pulse
                "readout": "readoutPosPulse",  # to a pulse
                "ramp_down_evolve": "rampDownD1Pulse",
                "ramp_up_read": "rampUpD1Pulse",
            },
        },
        "D2": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0e6,
            "operations": {
                # "prep_pos": "constPosPulse",  # to a pulse
                "prep": "prepNegPulse",  # to a pulse
                "evolve": "evolveD2Pulse",  # to a pulse
                "pump": "constPulse",
                "readout": "readoutNegPulse",  # to a pulse
                "ramp_down_read": "rampDownD2Pulse",
                "ramp_up_evolve": "rampUpD2Pulse",
            },
        },
        "D1_RF": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0e6,
            "operations": {
                "const": "constPulse",
                # "prep": "prepPulse",  # to a pulse
                # "readout": "readoutPulse",
                # "s-pump": "pumpPulse",  # to a pulse
                # "pi_pulse": "pi_pulse_in"
            },
        },
        "D2_RF": {
            "singleInput": {"port": ("con1", 2)},
            "intermediate_frequency": 0e6,
            "operations": {
                "const": "constPulse",
                # "prep": "prepPulse",  # to a pulse
                # "readout": "readoutPulse",
                # "s-pump": "pumpPulse",  # to a pulse
                # "pi_pulse": "pi_pulse_in"
            },
        },
        "RF-QPC": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": 200e6,
            "operations": {
                "measure": "measure_pulse",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
    },
    "pulses": {
        "measure_pulse": {  # Readout pulse
            "operation": "measurement",
            "length": readout_time,
            "waveforms": {
                "single": "const_wf",
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
        "rampDownD1Pulse": {
            "operation": "control",
            "length": ramp_len,  # in ns
            "waveforms": {"single": "ramp_down_d1_wf"},
        },
        "rampUpD1Pulse": {
            "operation": "control",
            "length": ramp_len,  # in ns
            "waveforms": {"single": "ramp_up_d1_wf"},
        },
        "rampDownD2Pulse": {
            "operation": "control",
            "length": ramp_len,  # in ns
            "waveforms": {"single": "ramp_down_d2_wf"},
        },
        "rampUpD2Pulse": {
            "operation": "control",
            "length": ramp_len,  # in ns
            "waveforms": {"single": "ramp_up_d2_wf"},
        },
        "readoutPosPulse": {
            "operation": "control",
            "length": readout_time,  # in ns
            "waveforms": {"single": "readout_pos_wf"},
        },
        "readoutNegPulse": {
            "operation": "control",
            "length": readout_time,  # in ns
            "waveforms": {"single": "readout_neg_wf"},
        },
        "evolveD2Pulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "evolve_pos_wf"},
        },
        "evolveD1Pulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "evolve_neg_wf"},
        },
        "prepPosPulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "prep_pos_wf"},
        },
        "prepNegPulse": {
            "operation": "control",
            "length": const_pulse_len,  # in ns
            "waveforms": {"single": "const_neg_wf"},
        },
        "prepPulse": {
            "operation": "control",
            "length": prep_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "readoutPulse": {
            "operation": "control",
            "length": readout_time,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
        },
        "pumpPulse": {
            "operation": "control",
            "length": pump_len,
            "waveforms": {"I": "const_wf", "Q": "const_wf"},
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
        "const_wf": {"type": "constant", "sample": 0.1},
        "prep_pos_wf": {"type": "constant", "sample": v_prep_top},
        "prep_neg_wf": {"type": "constant", "sample": v_prep_bot},
        "evolve_pos_wf": {"type": "constant", "sample": v_evolve_top},
        "evolve_neg_wf": {"type": "constant", "sample": v_evolve_bot},
        "readout_pos_wf": {"type": "constant", "sample": v_readout_top},
        "readout_neg_wf": {"type": "constant", "sample": v_readout_bot},
        "const_neg_wf": {"type": "constant", "sample": -0.2},
        "ev_d1_wf": {"type": "constant", "sample": -0.2},
        "ev_d2_wf": {"type": "constant", "sample": 0.1},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "ramp_down_d1_wf": {"type": "arbitrary", "samples": ramp_down_prep},
        "ramp_up_d1_wf": {"type": "arbitrary", "samples": ramp_up_readout},
        "ramp_down_d2_wf": {"type": "arbitrary", "samples": ramp_down_readout},
        "ramp_up_d2_wf": {"type": "arbitrary", "samples": ramp_up_prep},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
        "exc_wf": {"type": "constant", "sample": 0.479},
        "ramp_wf": {
            "type": "arbitrary",
            "samples": np.linspace(0, ramp_max, pump_len).tolist(),
        },
    },
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {"cosine": [4.0] * readout_time, "sine": [0.0] * readout_time},
        "integW2": {"cosine": [0.0] * readout_time, "sine": [4.0] * readout_time},
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
                "intermediate_frequency": 0,  # ω_d,
                "lo_frequency": 5.10e9,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
