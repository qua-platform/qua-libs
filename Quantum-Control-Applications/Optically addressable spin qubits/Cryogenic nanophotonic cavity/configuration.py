import numpy as np
from single_photon_trace import photon_signal, photon_signal_shifted
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


#############
# VARIABLES #
#############
u = unit()
qop_ip = "172.16.2.103"

# Frequencies
Yb_IF_freq = 40e6  # in units of Hz
Yb_LO_freq = 2.83e9  # in units of Hz
optical_transition_IF = 300e6  # in units of Hz

# Pulses lengths
initialization_len = 3000  # in ns
excited_state_init = 300  # in ns
meas_len = 2e3  # in ns
long_meas_len = 160e3  # in ns

# MW parameters
mw_amp_NV = 0.2  # in units of volts
mw_len_NV = 100  # in units of ns

B_RF_len = 100  # in units of ns
B_RF_amp = 0.1  # in units of volts

pi_amp_NV = 0.1  # in units of volts
pi_len_NV = 100  # in units of ns

pi_half_amp_NV = pi_amp_NV / 2  # in units of volts
pi_half_len_NV = pi_len_NV  # in units of ns

# Readout parameters
signal_threshold = 900

# Delays
detection_delay = 136
mw_delay = 0
laser_delay = 0

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
            },
        }
    },
    "elements": {
        "Yb": {
            "mixInputs": {"I": ("con1", 1), "Q": ("con1", 2), "lo_frequency": Yb_LO_freq, "mixer": "mixer_Yb"},
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "switch0": {
                    "port": ("con1", 8),
                    "delay": 136,
                    "buffer": 0,
                },
                "switch1": {
                    "port": ("con1", 9),
                    "delay": 136,
                    "buffer": 0,
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
        "photon_source": {
            "singleInput": {"port": ("con1", 9)},
            "operations": {
                "photon": "photon_pulse",
                "photon2": "photon_pulse2",
            },
        },
        "SNSPD": {
            # "singleInput": {"port": ("con1", 1)},  # not used
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 2),
                    "delay": detection_delay,
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
                "signalPolarity": "Ascending",
                "derivativeThreshold": 1023,
                "derivativePolarity": "Below",
            },
            "time_of_flight": detection_delay,
            "smearing": 0,
        },
    },
    "pulses": {
        "photon_pulse": {"operation": "control", "length": len(photon_signal), "waveforms": {"single": "photon_wf"}},
        "photon_pulse2": {
            "operation": "control",
            "length": len(photon_signal_shifted),
            "waveforms": {"single": "photon_shifted_wf"},
        },
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
            "length": pi_half_len_NV,
            "waveforms": {"I": "pi_half_wf", "Q": "zero_wf"},
        },
        "x180_pulse": {
            "operation": "control",
            "length": pi_len_NV,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "-x180_pulse": {
            "operation": "control",
            "length": pi_len_NV,
            "waveforms": {"I": "minus_pi_wf", "Q": "zero_wf"},
        },
        "-x90_pulse": {
            "operation": "control",
            "length": pi_half_len_NV,
            "waveforms": {"I": "minus_pi_half_wf", "Q": "zero_wf"},
        },
        "y90_pulse": {
            "operation": "control",
            "length": pi_half_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "pi_half_wf"},
        },
        "y180_pulse": {
            "operation": "control",
            "length": pi_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "pi_wf"},
        },
        "-y180_pulse": {
            "operation": "control",
            "length": pi_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "minus_pi_wf"},
        },
        "-y90_pulse": {
            "operation": "control",
            "length": pi_half_len_NV,
            "waveforms": {"I": "zero_wf", "Q": "minus_pi_half_wf"},
        },
        "laser_ON": {
            "operation": "control",
            "length": initialization_len,
            "digital_marker": "ON",
        },
        "laser_ON_A": {
            "operation": "control",
            "waveforms": {"single": "zero_wf"},
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
            # "waveforms": {"single": "zero_wf"},
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_meas_len,
            "digital_marker": "ON",
            # "waveforms": {"single": "zero_wf"},
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
        "photon_wf": {"type": "arbitrary", "samples": (photon_signal / 5).tolist()},
        "photon_shifted_wf": {"type": "arbitrary", "samples": (photon_signal_shifted / 5).tolist()},
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
