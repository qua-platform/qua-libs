from pathlib import Path

import numpy as np
import plotly.io as pio
from qualang_tools.units import unit
from qualang_tools.voltage_gates import VoltageGateSequence
from scipy.signal.windows import gaussian

pio.renderers.default = "browser"

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the 'I' & 'Q' ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the 'I' & 'Q' ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


######################
# Network parameters #
######################
qop_ip = "127.0.0.1"  # Write the QM router IP address
cluster_name = "my_cluster"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

#############
# Save Path #
#############
# Path to save data
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)

default_additional_files = {
    Path(__file__).name: Path(__file__).name,
}

#####################
# OPX configuration #
#####################
# Set octave_config to None if no octave are present
octave_config = None

######################
#       READOUT      #
######################
# DC readout parameters
readout_len = 1 * u.us
readout_amp = 0.0
IV_scale_factor = 0.5e-9  # in A/V

# Reflectometry
resonator_IF = 151 * u.MHz
reflectometry_readout_length = 1 * u.us
reflectometry_readout_amp = 30 * u.mV

# Time of flight
time_of_flight = 28

######################
#      DC GATES      #
######################

## Section defining the points from the charge stability map - can be done in the config
# Relevant points in the charge stability map as ["P1", "P2"] in V
level_init = [0.1, -0.1]
level_manip = [0.2, -0.2]
level_readout = [0.12, -0.12]

# Duration of each step in ns
duration_init = 2500
duration_manip = 1000
duration_readout = readout_len + 100
duration_compensation_pulse = 4 * u.us  # Note, may need to be increased when running long RB sequences

# Step parameters
coulomb_step_length = 60  # in ns
step_length = 16  # in ns
P1_step_amp = 0.25  # in V
P2_step_amp = 0.25  # in V
charge_sensor_amp = 0.25  # in V

# Time to ramp down to zero for sticky elements in ns
hold_offset_duration = 4  # in ns
bias_tee_cut_off_frequency = 10 * u.kHz

######################
#    QUBIT PULSES    #
######################
qubit_LO = 4 * u.GHz
qubit_IF = 100 * u.MHz
qubit_g = 0.0
qubit_phi = 0.0

# Pi pulse
pi_amp = 0.25  # in V
pi_length = 32  # in ns
# Pi half
pi_half_amp = 0.25  # in V
pi_half_length = 16  # in ns
# Gaussian pulse
gaussian_amp = 0.1  # in V
gaussian_length = 20  # in ns
# CW pulse
cw_amp = 0.3  # in V
cw_len = 100  # in ns

# Chirp Pulse
chirp_duration = 1000  # in clock cycles
chirp_rate = 5000
chirp_units = "Hz/nsec"
chirp_amp = 0.05  # 0.3
processing_time = 196  # time in ns for chirp to be calculated

# RB Gate specifics
x180_len = 100
x180_amp = 0.1  # 0.35

x90_len = x180_len
x90_amp = x180_amp / 2

minus_x90_len = x180_len
minus_x90_amp = -x90_amp

y180_len = x180_len
y180_amp = x180_amp

y90_len = x180_len
y90_amp = y180_amp / 2

minus_y90_len = y180_len
minus_y90_amp = -y90_amp

#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # P1
                2: {"offset": 0.0},  # P2
                3: {"offset": 0.0},  # EDSR I quadrature (Octave I2)
                4: {"offset": 0.0},  # EDSR Q quadrature (Octave Q2)
                5: {"offset": 0.0},  # Sensor gate
                9: {"offset": 0.0},  # RF reflectometry
                10: {"offset": 0.0},  # DC readout
            },
            "digital_outputs": {
                1: {},  # TTL for QDAC
                2: {},  # TTL for QDAC
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # RF reflectometry input
                2: {"offset": 0.0, "gain_db": 0},  # DC readout input
            },
        },
    },
    "elements": {
        "P1": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "operations": {
                "step": "P1_step_pulse",
            },
        },
        "P1_sticky": {
            "singleInput": {
                "port": ("con1", 1),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "step": "P1_step_pulse",
            },
        },
        "P2": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "operations": {
                "step": "P2_step_pulse",
            },
        },
        "P2_sticky": {
            "singleInput": {
                "port": ("con1", 2),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
            "operations": {
                "step": "P2_step_pulse",
            },
        },
        "sensor_gate": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "operations": {
                "step": "bias_charge_pulse",
            },
        },
        "sensor_gate_sticky": {
            "singleInput": {
                "port": ("con1", 5),
            },
            "sticky": {"analog": True, "duration": hold_offset_duration},
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
        "qubit": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": qubit_LO,
                "mixer": "mixer_qubit_left",  # a fixed name, do not change.
            },
            "intermediate_frequency": qubit_IF,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
                "gauss": "gaussian_pulse",
                "chirp": "chirp_pulse",
                "x180": "x180_pulse",
                "x90": "x90_pulse",
                "-x90": "-x90_pulse",
                "y180": "y180_pulse",
                "y90": "y90_pulse",
                "-y90": "-y90_pulse",
            },
        },
        "tank_circuit": {
            "singleInput": {
                "port": ("con1", 9),
            },
            "intermediate_frequency": resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "TIA": {
            "singleInput": {
                "port": ("con1", 10),
            },
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
    },
    "pulses": {
        "P1_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P1_step_wf",
            },
        },
        "P2_step_pulse": {
            "operation": "control",
            "length": step_length,
            "waveforms": {
                "single": "P2_step_wf",
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
        "cw_pulse": {
            "operation": "control",
            "length": cw_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "chirp_pulse": {
            "operation": "control",
            "length": chirp_duration,
            "waveforms": {
                "I": "chirp_wf",
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
        "pi_pulse": {
            "operation": "control",
            "length": pi_length,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_half_length,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "reflectometry_readout_pulse": {
            "operation": "measurement",
            "length": reflectometry_readout_length,
            "waveforms": {
                "single": "reflect_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
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
                "constant": "constant_weights",
            },
            "digital_marker": "ON",
        },
        "x180_pulse": {
            "operation": "control",
            "length": x180_len,
            "waveforms": {
                "I": "x180_I_wf",
                "Q": "x180_Q_wf",
            },
        },
        "x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "x90_I_wf",
                "Q": "x90_Q_wf",
            },
        },
        "-x90_pulse": {
            "operation": "control",
            "length": x90_len,
            "waveforms": {
                "I": "-x90_I_wf",
                "Q": "-x90_Q_wf",
            },
        },
        "y180_pulse": {
            "operation": "control",
            "length": y180_len,
            "waveforms": {
                "I": "y180_I_wf",
                "Q": "y180_Q_wf",
            },
        },
        "y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "y90_I_wf",
                "Q": "y90_Q_wf",
            },
        },
        "-y90_pulse": {
            "operation": "control",
            "length": y90_len,
            "waveforms": {
                "I": "-y90_I_wf",
                "Q": "-y90_Q_wf",
            },
        },
    },
    "waveforms": {
        "P1_step_wf": {"type": "constant", "sample": P1_step_amp},
        "P2_step_wf": {"type": "constant", "sample": P2_step_amp},
        "charge_sensor_step_wf": {"type": "constant", "sample": charge_sensor_amp},
        "pi_wf": {"type": "constant", "sample": pi_amp},
        "pi_half_wf": {"type": "constant", "sample": pi_half_amp},
        "gaussian_wf": {
            "type": "arbitrary",
            "samples": list(gaussian_amp * gaussian(gaussian_length, gaussian_length / 5)),
        },
        "readout_pulse_wf": {"type": "constant", "sample": readout_amp},
        "reflect_wf": {"type": "constant", "sample": reflectometry_readout_amp},
        "const_wf": {"type": "constant", "sample": cw_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "chirp_wf": {"type": "constant", "sample": chirp_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x180_I_wf": {"type": "constant", "sample": x180_amp},
        "x180_Q_wf": {"type": "constant", "sample": 0.0},
        "x90_I_wf": {"type": "constant", "sample": x90_amp},
        "x90_Q_wf": {"type": "constant", "sample": 0.0},
        "-x90_I_wf": {"type": "constant", "sample": minus_x90_amp},
        "-x90_Q_wf": {"type": "constant", "sample": 0.0},
        "y180_I_wf": {"type": "constant", "sample": 0.0},
        "y180_Q_wf": {"type": "constant", "sample": y180_amp},
        "y90_I_wf": {"type": "constant", "sample": 0.0},
        "y90_Q_wf": {"type": "constant", "sample": y90_amp},
        "-y90_I_wf": {"type": "constant", "sample": 0.0},
        "-y90_Q_wf": {"type": "constant", "sample": minus_y90_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1, readout_len)],
            "sine": [(0.0, readout_len)],
        },
        "cosine_weights": {
            "cosine": [(1.0, reflectometry_readout_length)],
            "sine": [(0.0, reflectometry_readout_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, reflectometry_readout_length)],
            "sine": [(1.0, reflectometry_readout_length)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": IQ_imbalance(qubit_g, qubit_phi),
            },
        ],
    },
}
