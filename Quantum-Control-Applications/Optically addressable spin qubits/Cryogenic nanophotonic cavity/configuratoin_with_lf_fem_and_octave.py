"""
QUA-Config supporting OPX1000 w/ LF-FEM & Octave
"""

from pathlib import Path
import numpy as np
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
import plotly.io as pio

pio.renderers.default = "browser"


#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


######################
# Network parameters #
######################
qop_ip = "127.0.0.1"
cluster_name = "my_cluster"
qop_port = None

#############
# Save Path #
#############
# Path to save data
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

default_additional_files = {
    Path(__file__).name: Path(__file__).name,
    "optimal_weights.npz": "optimal_weights.npz",
}

#####################
# OPX configuration #
#####################
con = "con1"
fem = 1  # This should be the index of the LF-FEM module, e.g., 1

sampling_rate = int(1e9)  # or, int(2e9)

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
        con: {
            "type": "opx1000",
            "fems": {
                fem: {
                    "type": "LF",
                    "analog_outputs": {
                        # Yb I microwave
                        1: {
                            "offset": 0.0,
                            "delay": mw_delay,
                            # The "output_mode" can be used to tailor the max voltage and frequency bandwidth, i.e.,
                            #   "direct":    1Vpp (-0.5V to 0.5V), 750MHz bandwidth (default)
                            #   "amplified": 5Vpp (-2.5V to 2.5V), 330MHz bandwidth
                            # Note, 'offset' takes absolute values, e.g., if in amplified mode and want to output 2.0 V, then set "offset": 2.0
                            "output_mode": "direct",
                            # The "sampling_rate" can be adjusted by using more FEM cores, i.e.,
                            #   1 GS/s: uses one core per output (default)
                            #   2 GS/s: uses two cores per output
                            # NOTE: duration parameterization of arb. waveforms, sticky elements and chirping
                            #       aren't yet supported in 2 GS/s.
                            "sampling_rate": sampling_rate,
                            # At 1 GS/s, use the "upsampling_mode" to optimize output for
                            #   modulated pulses (optimized for modulated pulses):      "mw"    (default)
                            #   unmodulated pulses (optimized for clean step response): "pulse"
                            "upsampling_mode": "mw",
                        },
                        # Yb Q microwave
                        2: {
                            "offset": 0.0,
                            "delay": mw_delay,
                            "output_mode": "direct",
                            "sampling_rate": sampling_rate,
                            "upsampling_mode": "mw",
                        },
                        # B^{RF}
                        3: {
                            "offset": 0.0,
                            "delay": mw_delay,
                            "output_mode": "direct",
                            "sampling_rate": sampling_rate,
                            "upsampling_mode": "mw",
                        },
                        # Photon Source
                        8: {"delay": mw_delay, "offset": 0.0},
                    },
                    "digital_outputs": {
                        1: {},  # A-transition AOM0
                        2: {},  # A-transition AOM1
                        3: {},  # F-transition AOM0
                        4: {},  # F-transition AOM1
                        5: {},  # excited state mw switch0
                        6: {},  # excited state mw switch1
                        7: {},  # Yb mw switch0
                        8: {},  # Yb mw switch1
                    },
                    "analog_inputs": {
                        1: {"offset": 0, "gain_db": 0, "sampling_rate": sampling_rate},  # SPCM
                    },
                }
            },
        }
    },
    "elements": {
        "Yb": {
            "RF_inputs": {"port": ("oct1", 1)},
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "switch0": {
                    "port": (con, fem, 7),
                    "delay": 136,
                    "buffer": 0,
                },
                "switch1": {
                    "port": (con, fem, 8),
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
                    "port": (con, fem, 1),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON",
            },
        },
        "A_transition": {
            "singleInput": {"port": (con, fem, 1)},
            "intermediate_frequency": optical_transition_IF,
            "digitalInputs": {
                "marker0": {
                    "port": (con, fem, 1),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "marker1": {
                    "port": (con, fem, 2),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "laser_ON": "laser_ON_A",
            },
        },
        "B_RF": {
            "singleInput": {"port": (con, fem, 3)},
            "intermediate_frequency": optical_transition_IF,
            "operations": {
                "+cw": "+const_pulse",
                "-cw": "-const_pulse",
            },
        },
        "F_transition": {
            "digitalInputs": {
                "marker0": {
                    "port": (con, fem, 3),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "marker1": {
                    "port": (con, fem, 4),
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
                    "port": (con, fem, 5),
                    "delay": laser_delay,
                    "buffer": 0,
                },
                "switch1": {
                    "port": (con, fem, 6),
                    "delay": laser_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "switch_ON": "switch_ON",
            },
        },
        "SNSPD": {
            # "singleInput": {"port": (con, fem, 1)},  # not used
            "intermediate_frequency": Yb_IF_freq,
            "digitalInputs": {
                "marker": {
                    "port": (con, fem, 2),
                    "delay": detection_delay,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": (con, fem, 1)},
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
    "octaves": {
        "octave1": {
            "RF_outputs": {
                1: {
                    "LO_frequency": Yb_LO_freq,
                    "LO_source": "internal",  # can be external or internal. internal is the default
                    "output_mode": "always_on",
                    # can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed". "always_off" is the default
                    "gain": 0,  # can be in the range [-20 : 0.5 : 20]dB
                },
            },
            "connectivity": (con, fem),
        }
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
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
}
