import numpy as np
from qualang_tools.units import unit
from qualang_tools.voltage_gates import VoltageGateSequence
from qdac2_driver import QDACII, load_voltage_list

# Backward-compatible alias for experiment files that use the old name
OPX_virtual_gate_sequence = VoltageGateSequence

####################
# Helper functions #
####################
def update_readout_length(new_readout_length, config):

    config["pulses"]["lock_in_readout_pulse"]["length"] = new_readout_length
    config["integration_weights"]["cosine_weights"] = {
        "cosine": [(1.0, new_readout_length)],
        "sine": [(0.0, new_readout_length)],
    }
    config["integration_weights"]["sine_weights"] = {
        "cosine": [(0.0, new_readout_length)],
        "sine": [(1.0, new_readout_length)],
    }
    config["integration_weights"]["minus_sine_weights"] = {
        "cosine": [(0.0, new_readout_length)],
        "sine": [(-1.0, new_readout_length)],
    }
    
#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

######################
# Network parameters #
######################
qop_ip = "172.16.33.101"  # Write the QM router IP address
cluster_name = "Cluster_83"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

qdac_ip = "127.0.0.1"
qdac_port = 5025

# Path to save data
octave_config = None

######################
#       READOUT      #
######################
qds_IF = 1 * u.MHz
lock_in_readout_length = 1 * u.us
lock_in_readout_amp = 10 * u.mV
rotation_angle = (0.0 / 180) * np.pi

# Time of flight
time_of_flight = 24

######################
#      DC GATES      #
######################

## Section defining the points from the charge stability map - can be done in the config
level_readout = [0.12, -0.12]
level_dephasing = [-0.2, -0.1]

dephasing_ramp = 100
readout_ramp = 100
init_ramp = 100

# Duration of each step in ns
duration_readout = lock_in_readout_length
duration_compensation_pulse = 5 * u.us
duration_dephasing = 2000  # nanoseconds
duration_init = 10_000
duration_init_jumps = 16

# Step parameters
step_length = 16
P4_step_amp = 0.25
P5_step_amp = 0.25
P6_step_amp = 0.25
X4_step_amp = 0.25
X5_step_amp = 0.25
T6_step_amp = 0.25
charge_sensor_amp = 0.25

# Time to ramp down to zero for sticky elements in ns
ramp_down_duration = 4
bias_tee_cut_off_frequency = 400 * u.Hz

######################
#    QUBIT PULSES    #
######################
# Durations in ns
pi_length = 32
pi_half_length = 16
# Amplitudes in V
pi_amps = [0.27, -0.27]
pi_half_amps = [0.27, -0.27]


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # 
                2: {"offset": 0.0},  # QDS 1 MHz drive
                3: {"offset": 0.0},  # 
                4: {"offset": 0.0},  # 
                5: {"offset": 0.0},  # P4
                6: {"offset": 0.0},  # X4
                7: {"offset": 0.0},  # P5
                8: {"offset": 0.0},  # X5
                9: {"offset": 0.0},  # P6
                10: {"offset": 0.0},  # T6
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                2: {"offset": 0.0, "gain_db": 0},  # Lock-in channel
            },
        },
    },
    "elements": {
        "P4": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "step": "P4_step_pulse",
            },
        },
        "P4_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P4_step_pulse",
            },
        },
        "P5": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "operations": {
                "step": "P5_step_pulse",
            },
        },
        "P5_sticky": {
            "singleInput": {
                "port": ("con1", 7),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P5_step_pulse",
            },
        },
        "P6": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "operations": {
                "step": "P6_step_pulse",
            },
        },
        "P6_sticky": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "P6_step_pulse",
            },
        },
        "X4": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "operations": {
                "step": "X4_step_pulse",
            },
        },
        "X4_sticky": {
            "singleInput": {
                "port": ("con1", 6),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "X4_step_pulse",
            },
        },
        "X5": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "operations": {
                "step": "X5_step_pulse",
            },
        },
        "X5_sticky": {
            "singleInput": {
                "port": ("con1", 8),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "X5_step_pulse",
            },
        },
        "T6": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "operations": {
                "step": "T6_step_pulse",
            },
        },
        "T6_sticky": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "T6_step_pulse",
            },
        },
        "sensor_gate": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "operations": {
                "step": "bias_charge_pulse",
            },
        },
        "sensor_gate_sticky": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "sticky": {"analog": True, "duration": ramp_down_duration},
            "operations": {
                "step": "bias_charge_pulse",
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
        "QDS": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": qds_IF,
            "operations": {
                "readout": "lock_in_readout_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "QDS_twin": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "intermediate_frequency": qds_IF,
            "operations": {
                "readout": "lock_in_readout_pulse",
            },
            "outputs": {
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "P4_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P4_step_wf",
            },
        },
        "P5_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P5_step_wf",
            },
        },
        "P6_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P6_step_wf",
            },
        },
        "X4_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "X4_step_wf",
            },
        },
        "X5_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "X5_step_wf",
            },
        },
        "T6_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "T6_step_wf",
            },
        },
        "bias_charge_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "charge_sensor_step_wf",
            },
        },
        "trigger_pulse": {
            "operation": "control",
            "length": 1000,
            "digital_marker": "ON",
        },
        "lock_in_readout_pulse": {
            "operation": "measurement",
            "length": lock_in_readout_length,
            "waveforms": {
                "single": "lock_in_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "rotated_cos": "rotated_cosine_weights",
                "rotated_sin": "rotated_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "P4_step_wf": {"type": "constant", "sample": P4_step_amp},
        "P5_step_wf": {"type": "constant", "sample": P5_step_amp},
        "P6_step_wf": {"type": "constant", "sample": P6_step_amp},
        "X4_step_wf": {"type": "constant", "sample": X4_step_amp},
        "X5_step_wf": {"type": "constant", "sample": X5_step_amp},
        "T6_step_wf": {"type": "constant", "sample": T6_step_amp},
        "charge_sensor_step_wf": {"type": "constant", "sample": charge_sensor_amp},
        "lock_in_wf": {"type": "constant", "sample": lock_in_readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1, lock_in_readout_length)],
            "sine": [(0.0, lock_in_readout_length)],
        },
        "cosine_weights": {
            "cosine": [(1.0, lock_in_readout_length)],
            "sine": [(0.0, lock_in_readout_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, lock_in_readout_length)],
            "sine": [(1.0, lock_in_readout_length)],
        },
        "rotated_cosine_weights": {
            "cosine": [(np.cos(rotation_angle), lock_in_readout_length)],
            "sine": [(np.sin(rotation_angle), lock_in_readout_length)],
        },
        "rotated_sine_weights": {
            "cosine": [(-np.sin(rotation_angle), lock_in_readout_length)],
            "sine": [(np.cos(rotation_angle), lock_in_readout_length)],
        },
    },
}