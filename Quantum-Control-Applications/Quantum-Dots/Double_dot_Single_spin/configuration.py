from pathlib import Path
from scipy.signal.windows import gaussian
from qualang_tools.units import unit


#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "172.16.33.100"  # Write the QM router IP address
cluster_name = "Cluster_81"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

# Path to save data
octave_config = None

#############################################
#              OPX PARAMETERS               #
#############################################

######################
#       READOUT      #
######################
# DC readout parameters
readout_len = 10000
readout_amp = 0.4
IV_scale_factor = 0.5e-9  # in A/V

# Reflectometry
resonator_IF = 151 * u.MHz
reflectometry_readout_length = 1 * u.us
reflect_amp = 30 * u.mV

# Time of flight
time_of_flight = 300

######################
#      DC GATES      #
######################
P1_amp = 0.1
P2_amp = 0.4
B_center_amp = 0.2
charge_sensor_amp = 0.2

block_length = 100
bias_length = 200

hold_offset_duration = 200

######################
#      RF GATES      #
######################
qubit_LO_left = 4 * u.GHz
qubit_IF_left = 100 * u.MHz
qubit_LO_right = 4 * u.GHz
qubit_IF_right = 100 * u.MHz

# Pi pulse
pi_amp_left = 0.1
pi_half_amp_left = 0.1
pi_length_left = 40
pi_amp_right = 0.1
pi_half_amp_right = 0.1
pi_length_right = 40
# Square pulse
cw_amp = 0.3
cw_len = 100
# Gaussian pulse
gaussian_length = 20
gaussian_amp = 0.1


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # qubit_left I
                2: {"offset": 0.0},  # qubit_left Q
                3: {"offset": 0.0},  # qubit_right I
                4: {"offset": 0.0},  # qubit_right Q
                5: {"offset": 0.0},  # P1 qubit_left
                6: {"offset": 0.0},  # P2 qubit_right
                7: {"offset": 0.0},  # Barrier center
                8: {"offset": 0.0},  # charge sensor gate
                9: {"offset": 0.0},  # charge sensor DC
                10: {"offset": 0.0},  # charge sensor RF
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},   # DC input
                2: {"offset": 0.0, "gain_db": 0},   # RF input
            },
        },
    },
    "elements": {
        "B_center": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "operations": {
                "bias": "bias_B_center_pulse",
            },
        },
        "B_center_sticky": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_B_center_pulse",
            },
        },
        "P1": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "bias": "bias_P1_pulse",
            },
        },
        "P1_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_P1_pulse",
            },
        },
        "P2": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "bias": "bias_P2_pulse",
            },
        },
        "P2_sticky": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_P2_pulse",
            },
        },
        "qdac_trigger1": {
            "digitalInputs": {
                "trigger": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        "qdac_trigger2": {
            "digitalInputs": {
                "trigger": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        "qubit_left": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": qubit_LO_left,
                "mixer": "mixer_qubit_left",  # a fixed name, do not change.
            },
            "intermediate_frequency": qubit_IF_left,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_left_pulse",
                "gauss": "gaussian_pulse",
                "pi_half": "pi_half_left_pulse",
            },
        },
        "qubit_right": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO_right,
                "mixer": "mixer_qubit_right",  # a fixed name, do not change.
            },
            "intermediate_frequency": qubit_IF_right,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_right_pulse",
                "gauss": "gaussian_pulse",
                "pi_half": "pi_half_right_pulse",
            },
        },
        "charge_sensor_gate": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "operations": {
                "bias": "bias_charge_pulse",
            },
        },
        "charge_sensor_gate_sticky": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "bias": "bias_charge_pulse",
            },
        },
        "charge_sensor_RF": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "charge_sensor_DC": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "bias_P1_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_P1_pulse_wf",
            },
        },
        "bias_P2_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_P2_pulse_wf",
            },
        },
        "bias_B_center_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_B_center_pulse_wf",
            },
        },
        "bias_charge_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_charge_pulse_wf",
            },
        },
        "cw_pulse": {
            "operation": "control",
            "length": cw_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_length,
            "waveforms": {
                "I": "gaussian_wf",
                "Q": "zero_wf",
            },
        },
        "pi_left_pulse": {
            "operation": "control",
            "length": pi_length_left,
            "waveforms": {
                "I": "pi_left_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_left_pulse": {
            "operation": "control",
            "length": pi_length_left,
            "waveforms": {
                "I": "pi_half_left_wf",
                "Q": "zero_wf",
            },
        },
        "pi_right_pulse": {
            "operation": "control",
            "length": pi_length_right,
            "waveforms": {
                "I": "pi_right_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_right_pulse": {
            "operation": "control",
            "length": pi_length_right,
            "waveforms": {
                "I": "pi_half_right_wf",
                "Q": "zero_wf",
            },
        },
        "trigger_pulse": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "ON",
        },
        "reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cw_cosine_weights",
                "sin": "cw_sine_weights",
            },
            "digital_marker": "ON",
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_pulse_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "bias_P1_pulse_wf": {"type": "constant", "sample": P1_amp},
        "bias_P2_pulse_wf": {"type": "constant", "sample": P2_amp},
        "bias_B_center_pulse_wf": {"type": "constant", "sample": B_center_amp},
        "bias_charge_pulse_wf": {"type": "constant", "sample": charge_sensor_amp},
        "readout_pulse_wf": {"type": "constant", "sample": readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": cw_amp},
        "reflect_wf": {"type": "constant", "sample": reflect_amp},
        "gaussian_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in gaussian_amp * gaussian(gaussian_length, gaussian_length / 5)],
        },
        "pi_left_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp_left * gaussian(pi_length_left, pi_length_left / 5)]},
        "pi_half_left_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_half_amp_left * gaussian(pi_length_left, pi_length_left / 5)],
        },
        "pi_right_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp_right * gaussian(pi_length_right, pi_length_right / 5)]},
        "pi_half_right_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in pi_half_amp_right * gaussian(pi_length_right, pi_length_right / 5)],
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine_weights": {
            "cosine": [(1, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(1.0, readout_len)],
        },
        "cw_cosine_weights": {
            "cosine": [(1.0, reflectometry_readout_length)],
            "sine": [(0.0, reflectometry_readout_length)],
        },
        "cw_sine_weights": {
            "cosine": [(0.0, reflectometry_readout_length)],
            "sine": [(1.0, reflectometry_readout_length)],
        },
    },
    "mixers": {
        "mixer_qubit_left": [
            {
                "intermediate_frequency": qubit_IF_left,
                "lo_frequency": qubit_LO_left,
                "correction": (1, 0, 0, 1),
            },
        ],
        "mixer_qubit_right": [
            {
                "intermediate_frequency": qubit_IF_right,
                "lo_frequency": qubit_LO_right,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}
