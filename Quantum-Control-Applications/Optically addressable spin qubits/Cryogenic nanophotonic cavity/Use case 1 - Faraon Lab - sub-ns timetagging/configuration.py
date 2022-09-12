import numpy as np
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool

#######################
# AUXILIARY FUNCTIONS #
#######################

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer

    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


def generalized_gaussian(amplitude, length, sigma, exponent):
    """
    Generalized gaussian waveform for pulse definitions
    :param amplitude: maximum amplitude of the waveform
    :param length: length of the waveform
    :param sigma: sigma (width) of the pulse
    :param exponent: exponent of the generalized gaussian (normal gaussian, exponent =2)
    :return: waveform, will start at 0
    """
    t = np.arange(0, length, 1)
    return amplitude * np.exp(-(((t - length / 2) / sigma) ** exponent))


#############
# VARIABLES #
#############

u = unit()
qop_ip = "192.168.1.209"

# Frequencies
Yb_IF_freq = 275.76e6 + 500e3  # in units of Hz
Yb_LO_freq = 2.83e9  # in units of Hz
optical_transition_IF = 268e6  # in units of Hz

# Pulses lengths
initialization_len = 3000  # in ns
excited_state_init = 300  # in ns
meas_len = 100  # in ns
long_meas_len = 160e3  # in ns
readout_len = 1e3  # in ns

# MW parameters
mw_amp_NV = 0.49  # in units of volts - NO MORE THAN 0.15 V TO NOT BURN AOM
mw_len_NV = 128  # in units of ns

B_RF_len = 100  # in units of ns
B_RF_amp = 0.1  # in units of volts

pi_amp_NV = 0.1  # in units of volts
pi_len_NV = 100  # in units of ns

pi_half_amp_NV = pi_amp_NV / 2  # in units of volts
pi_half_len_NV = pi_len_NV  # in units of ns

x90_wf = generalized_gaussian(mw_amp_NV, mw_len_NV / 2, mw_len_NV / 2 * 0.3, 4)
x180_wf = generalized_gaussian(mw_amp_NV, mw_len_NV, mw_len_NV * 0.3, 4)
mx180_wf = generalized_gaussian(-mw_amp_NV, mw_len_NV, mw_len_NV * 0.3, 4)
mx90_wf = generalized_gaussian(-mw_amp_NV, mw_len_NV / 2, mw_len_NV / 2 * 0.3, 4)
y90_wf = generalized_gaussian(mw_amp_NV, mw_len_NV / 2, mw_len_NV / 2 * 0.3, 4)
y180_wf = generalized_gaussian(mw_amp_NV, mw_len_NV, mw_len_NV * 0.3, 4)
my180_wf = generalized_gaussian(-mw_amp_NV, mw_len_NV, mw_len_NV * 0.3, 4)
my90_wf = generalized_gaussian(-mw_amp_NV, mw_len_NV / 2, mw_len_NV / 2 * 0.3, 4)


# Readout parameters
signal_threshold = 700

# Delays
detection_delay = 136
mw_delay = 0
laser_delay = 136

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0, "delay": mw_delay},  # Yb I microwave
                2: {"offset": 0.0, "delay": mw_delay},  # Yb Q microwave
                3: {"offset": 0.0, "delay": mw_delay},  # B^{RF}
                9: {"offset": 0.0, "delay": mw_delay},  # photon_source
            },
            "digital_outputs": {
                1: {},  # A-transition AOM0
                2: {},  # A-transition AOM1
                3: {},  # F-transition AOM0
                4: {},  # F-transition AOM1
                5: {},  # excited state mw switch0
                6: {},  # excited state mw switch1
                7: {},  # SNSPD shutter AOM
                8: {},  # Yb mw switch0
                9: {},  # Yb mw switch1
            },
            "analog_inputs": {
                1: {"offset": 0, "gain_db": 0},  # SPCM
                2: {"offset": 0, "gain_db": 0},  # SPCM
            },
        }
    },
    "elements": {
        "Yb": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2), "lo_frequency": Yb_LO_freq, "mixer": "mixer_Yb"},
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "switch0": {
                    "port": ("con1", 1),
                    "delay": 136,
                    "buffer": 0,
                },
                "switch1": {
                    "port": ("con1", 9),
                    "delay": 136 + 140 - 20 - 50 - 40,  # Calibrated 04/08/2022
                    "buffer": 40,
                },
            },
            "operations": {
                "cw": "const_pulse_IQ",
                "pi": "x180_pulse",
                "pi_half": "x90_pulse",
                "x90": "x90_pulse",
                "x180": "x180_pulse",
                "-x180": "-x180_pulse",
                "-x90": "-x90_pulse",
                "y90": "y90_pulse",
                "y180": "y180_pulse",
                "-y180": "-y180_pulse",
                "-y90": "-y90_pulse",
            },
        },
        "AOM": {
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON",
            },
        },
        "A_transition": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": optical_transition_IF,
            "digitalInputs": {
                "marker0": {
                    "port": ("con1", 1),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "marker1": {
                    "port": ("con1", 2),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON_A",
            },
        },
        "B_RF": {
            "singleInput": {"port": ("con1", 3)},
            "intermediate_frequency": optical_transition_IF,
            "operations": {
                "+cw": "+const_pulse",
                "-cw": "-const_pulse",
            },
        },
        "F_transition": {
            "digitalInputs": {
                "marker0": {
                    "port": ("con1", 3),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "marker1": {
                    "port": ("con1", 4),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON",
            },
        },
        "excited_state_mw": {
            "digitalInputs": {
                "switch0": {
                    "port": ("con1", 5),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "switch1": {
                    "port": ("con1", 6),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "switch_ON": "switch_ON",
            },
        },
        "SNSPD": {
            "singleInput": {"port": ("con1", 1)},  # not used
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 4),
                    # "delay": detection_delay,
                    "delay": 52,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "outputPulseParameters": {
                "signalThreshold": signal_threshold,
                "signalPolarity": "Below",
                "derivativeThreshold": 2000,
                "derivativePolarity": "Below",
            },
            "time_of_flight": detection_delay,
            # "time_of_flight": 36,
            "smearing": 0,
        },
        "event_trigger": {
            "singleInput": {  # Not needed, but without we cannot save the adc signal (bug)
                "port": ("con1", 3),
            },
            "intermediate_frequency": 0,
            "operations": {
                "readout": "readout2_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": 36,
            "smearing": 0,
        },
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"single": "cw_wf"},
        },
        "+const_pulse": {
            "operation": "control",
            "length": B_RF_len,
            "waveforms": {"single": "+cw_wf"},
        },
        "-const_pulse": {
            "operation": "control",
            "length": B_RF_len,
            "waveforms": {"single": "-cw_wf"},
        },
        "const_pulse_IQ": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "cw_wf", "Q": "zero_wf"},
        },
        "x90_pulse": {
            "operation": "control",
            "length": mw_len_NV / 2,
            "waveforms": {"I": "x90_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
        "x180_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "x180_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
        "-x180_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "mx180_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
        "-x90_pulse": {
            "operation": "control",
            "length": mw_len_NV / 2,
            "waveforms": {"I": "mx90_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
        "y90_pulse": {
            "operation": "control",
            "length": mw_len_NV / 2,
            "waveforms": {"I": "zero_wf", "Q": "y90_wf"},
            "digital_marker": "ON",
        },
        "y180_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "y180_wf"},
            "digital_marker": "ON",
        },
        "-y180_pulse": {
            "operation": "control",
            "length": mw_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "my180_wf"},
            "digital_marker": "ON",
        },
        "-y90_pulse": {
            "operation": "control",
            "length": mw_len_NV / 2,
            "waveforms": {"I": "zero_wf", "Q": "my90_wf"},
            "digital_marker": "ON",
        },
        "laser_ON": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "laser_ON_A": {
            "operation": "control",
            "waveforms": {"single": "cw_wf"},
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "switch_ON": {
            "operation": "control",
            "length": excited_state_init,
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": meas_len,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
        "readout2_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            },
            "digital_marker": "ON",
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_meas_len,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},
        },
    },
    "waveforms": {
        "cw_wf": {"type": "constant", "sample": mw_amp_NV},
        "+cw_wf": {"type": "constant", "sample": B_RF_amp},
        "-cw_wf": {"type": "constant", "sample": -B_RF_amp},
        "pi_wf": {"type": "constant", "sample": pi_amp_NV},
        "minus_pi_wf": {"type": "constant", "sample": -pi_amp_NV},
        "pi_half_wf": {"type": "constant", "sample": pi_half_amp_NV},
        "minus_pi_half_wf": {"type": "constant", "sample": -pi_half_amp_NV},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90_wf.tolist()},
        "x180_wf": {"type": "arbitrary", "samples": x180_wf.tolist()},
        "mx180_wf": {"type": "arbitrary", "samples": mx180_wf.tolist()},
        "mx90_wf": {"type": "arbitrary", "samples": mx90_wf.tolist()},
        "y90_wf": {"type": "arbitrary", "samples": y90_wf.tolist()},
        "y180_wf": {"type": "arbitrary", "samples": y180_wf.tolist()},
        "my180_wf": {"type": "arbitrary", "samples": my180_wf.tolist()},
        "my90_wf": {"type": "arbitrary", "samples": my90_wf.tolist()},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1.0, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
    "mixers": {
        "mixer_Yb": [
            {"intermediate_frequency": Yb_IF_freq, "lo_frequency": Yb_LO_freq, "correction": IQ_imbalance(0.0, 0.0)},
        ],
    },
}
