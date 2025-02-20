import os

import matplotlib.pyplot as plt
import numpy as np
from octave_sdk import Octave
from qm import QuantumMachine
from qm.octave import *
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from scipy.signal.windows import gaussian
from set_octave import OctaveUnit, octave_declaration

#######################
# AUXILIARY FUNCTIONS #
#######################

u = unit(coerce_to_integer=True)
######################
# Network parameters #
######################
opx_ip = "192.168.88.244"
#qop_ip = "10.209.68.77" #"10.209.68.77"
octave_ip = "192.168.88.253"
cluster_name = "Cluster_1" #"Cluster_1"  # Write your cluster_name if version >= QOP220
qop_port = 9510
octave = "oct1"

############################
# Set octave configuration #
############################
octave_1 = OctaveUnit("oct1", octave_ip, port=80, con="con1")
octaves = [octave_1]
octave_config = octave_declaration(octaves)

#############
# VARIABLES #
#############

# Frequencies
pulsed_laser_AOM_IF = 100 * u.MHz
readout_AOM_IF = 50.0 * u.MHz
control_AOM_IF = 50.0 * u.MHz
control_EOM_IF = 100 * u.MHz
control_EOM_LO = 4.9 * u.GHz

# Pulses lengths
readout_aom_len = 1000 * u.ns
control_aom_len = 1000 * u.ns
control_eom_len = 16 * u.ns #100 * u.ns
pulsed_laser_aom_len = 100 * u.ns
snspd_readout_len = 16 * u.ns
gaussian_len = 100 * u.ns

# Delays
readout_aom_delay = 0 * u.ns
control_aom_delay = 0 * u.ns
control_eom_delay = 0 * u.ns
pulsed_laser_aom_delay = 0 * u.ns

# Amplitudes
readout_amp = 0.1
control_aom_amp = 0.1
control_eom_amp = 0.4
pulsed_laser_amp = 0.1
gaussian_amp = 0.1


# Time of flight
time_of_flight = 164 * u.ns

#################
# CONFIGURATION #
#################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
                3: {"offset": 0.0},
                4: {"offset": 0.0},
                5: {"offset": 0.0},
                6: {"offset": 0.0},
                7: {"offset": 0.0},
                8: {"offset": 0.0},
                9: {"offset": 0.0},
                10: {"offset": 0.0},
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
                4: {},
                5: {},
                6: {},
                7: {},
                8: {},
                9: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0},
                2: {"offset": 0.0},
            },
        },
    },
    "elements": {
        "readout_aom": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "intermediate_frequency": readout_AOM_IF,
            "operations": {
                "readout": "cw_readout_aom",
            },
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 3),
                    "delay": readout_aom_delay,
                    "buffer": 0,
                },
            },
        },
        "control_aom": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "intermediate_frequency": control_AOM_IF,
            "operations": {
                "control": "cw_control_aom",
                "gaussian": "gaussian_pulse",
            },
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 2),
                    "delay": control_aom_delay,
                    "buffer": 0,
                },
            },
        },
        "control_eom": {
            "RF_inputs": {"port": ("oct1", 1)},
            "intermediate_frequency": control_EOM_IF,
            "operations": {
                "control": "cw_control_eom",
                "short": "cw_control_eom_short",
            },
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1),
                    "delay": control_eom_delay,
                    "buffer": 0,
                },
            },
        },
        "control_eom2": {
            "RF_inputs": {"port": ("oct1", 1)},
            "intermediate_frequency": control_EOM_IF,
            "operations": {
                "control": "cw_control_eom",
                "short": "cw_control_eom_short",
            },
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 1),
                    "delay": control_eom_delay,
                    "buffer": 0,
                },
            },
        },
        "pulsed_laser_aom": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "intermediate_frequency": pulsed_laser_AOM_IF,
            "operations": {
                "control": "cw_pulsed_laser_aom",
            },
            "digitalInputs": {
                "marker": {
                    "port": ("con1", 6),
                    "delay": pulsed_laser_aom_delay,
                    "buffer": 0,
                },
            },
        },
        "SNSPD": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "operations": {"readout": "readout_pulse_snspd"},
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "time_tagger": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "operations": {"readout": "readout_pulse_tt"},
            "outputs": {
                "out1": ("con1", 1),
            },
            "outputPulseParameters": {
                "signalThreshold": -20,
                "signalPolarity": "Below",
                "derivativeThreshold": -100,
                "derivativePolarity": "Below",
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "time_tagger2": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "operations": {"readout": "readout_pulse_tt"},
            "outputs": {
                "out1": ("con1", 1),
            },
            "outputPulseParameters": {
                "signalThreshold": -20,
                "signalPolarity": "Below",
                "derivativeThreshold": -100,
                "derivativePolarity": "Below",
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "octaves": {
        "oct1": {
            "RF_outputs": {
                1: {
                    "LO_frequency": control_EOM_LO,
                    "LO_source": "internal",  # can be external or internal. internal is the default
                    "output_mode": "triggered",  # can be: "always_on" / "always_off"/ "triggered" / "triggered_reversed". "always_off" is the default
                    "gain": 20,  # can be in the range [-20 : 0.5 : 20]dB
                },
            },
            "connectivity": "con1",
        },
    },
    "pulses": {
        "cw_readout_aom": {
            "operation": "control",
            "length": readout_aom_len,
            "waveforms": {"single": "cw_r"},
            "digital_marker": "ON",
        },
        "cw_control_aom": {
            "operation": "control",
            "length": control_aom_len,
            "waveforms": {"single": "cw_c_a"},
            "digital_marker": "ON",
        },
        "cw_control_eom": {
            "operation": "control",
            "length": control_eom_len,
            "waveforms": {
                "I": "cw_c_e",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "cw_control_eom_short": {
            "operation": "control",
            "length": 16,
            "waveforms": {
                "I": "eom_short_wf",
                "Q": "zero_wf",
            },
            "digital_marker": "ON",
        },
        "cw_pulsed_laser_aom": {
            "operation": "control",
            "length": pulsed_laser_aom_len,
            "waveforms": {"single": "cw_pl_a"},
            "digital_marker": "ON",
        },
        "readout_pulse_snspd": {
            "operation": "measurement",
            "length": snspd_readout_len,
            "waveforms": {"single": "zero_wf"},
            "integration_weights": {"constant": "constant_weights_snspd"},
            "digital_marker": "ON",
        },
        "readout_pulse_tt": {
            "operation": "measurement",
            "length": snspd_readout_len,
            "waveforms": {"single": "zero_wf"},
            "integration_weights": {"constant": "constant_weights_snspd"},
            "digital_marker": "ON",
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_len,
            "waveforms": {"single": "gaussian_wf"},
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "cw_r": {"type": "constant", "sample": readout_amp},
        "cw_c_a": {"type": "constant", "sample": control_aom_amp},
        "cw_c_e": {"type": "constant", "sample": control_eom_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "cw_pl_a": {"type": "constant", "sample": pulsed_laser_amp},
        "gaussian_wf": {
            "type": "arbitrary",
            "samples": list(gaussian_amp * gaussian(gaussian_len, gaussian_len / 5)),
        },
        "eom_short_wf": {
            "type": "arbitrary",
            "samples": list([0.4,0.4,0.4,0.4,0, 0,0,0,0,0, 0,0,0,0,0, 0]),
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},  # [(on/off, ns)]
        "OFF": {"samples": [(0, 0)]},  # [(on/off, ns)]
    },
    "integration_weights": {
        "constant_weights_snspd": {
            "cosine": [(1, snspd_readout_len)],
            "sine": [(0.0, snspd_readout_len)],
        },
    },
}
