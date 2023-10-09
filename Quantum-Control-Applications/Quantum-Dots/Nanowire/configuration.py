from pathlib import Path
from scipy.signal.windows import gaussian
from set_octave import OctaveUnit, octave_declaration
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
save_dir = Path().absolute() / "QM" / "INSTALLATION" / "data"

############################
# Set octave configuration #
############################
# Custom port mapping example
port_mapping = {
        ("con1", 1): ("octave1", "I1"),
        ("con1", 2): ("octave1", "Q1"),
        ("con1", 3): ("octave1", "I2"),
        ("con1", 4): ("octave1", "Q2"),
        ("con1", 5): ("octave1", "I3"),
        ("con1", 6): ("octave1", "Q3"),
        ("con1", 7): ("octave1", "I4"),
        ("con1", 8): ("octave1", "Q4"),
        ("con1", 9): ("octave1", "I5"),
        ("con1", 10): ("octave1", "Q5"),
    }
# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", qop_ip, port=11050, con="con1", clock="Internal", port_mapping="default")

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#############################################
#              OPX PARAMETERS               #
#############################################

######################
#       READOUT      #
######################
# DC readout parameters
readout_len = 10 * u.us
readout_amp = 0

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
P2_amp = 0.5

block_length = 100
bias_length = 200

hold_offset_duration = 200

######################
#      RF GATES      #
######################
qubit_LO = 4 * u.GHz
qubit_IF = 100 * u.MHz

# Pi pulse
pi_amp = 0.1
pi_half_amp = 0.1
pi_length = 40
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
                1: {"offset": 0.0},  # plunger gate 1
                2: {"offset": 0.0},  # plunger gate 2
                3: {"offset": 0.0},  # qubit I
                4: {"offset": 0.0},  # qubit Q
                9: {"offset": 0.0},  # tank circuit
                10: {"offset": 0.0},  # TIA
            },
            "digital_outputs": {
                1: {},
                2: {},
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},
                2: {"offset": 0.0, "gain_db": 0},
            },
        },
    },
    "elements": {
        "gate_1": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "operations": {
                "bias": "bias_P1_pulse",
            },
        },
        "P1_sticky": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "sticky": {'analog': True, 'duration': hold_offset_duration },
            "operations": {
                "bias": "bias_P1_pulse",
            },
        },
        "gate_2": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "operations": {
                "bias": "bias_P2_pulse",
            },
        },
        "P2_sticky": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "sticky": {'analog': True, 'duration': hold_offset_duration},
            "operations": {
                "bias": "bias_P2_pulse",
            },
        },
        "qdac_trigger1": {
            "singleInput": {
                "port": ("con1", 1),
            },
            'digitalInputs': {
                'trigger': {
                    'port': ('con1', 1),
                    'delay': 0,
                    'buffer': 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        "qdac_trigger2": {
            "singleInput": {
                "port": ("con1", 1),
            },
            'digitalInputs': {
                'trigger': {
                    'port': ('con1', 2),
                    'delay': 0,
                    'buffer': 0,
                }
            },
            "operations": {
                "trigger": "trigger_pulse",
            },
        },
        'qubit': {
            'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                "lo_frequency": qubit_LO,
                "mixer": "octave_octave1_1",  # a fixed name, do not change.
            },
            'intermediate_frequency': qubit_IF,
            'operations': {
                'cw': 'cw_pulse',
                'pi': 'pi_pulse',
                'gauss': 'gaussian_pulse',
                'pi_half': 'pi_half_pulse',
            },
        },
        "tank_circuit": {
            "singleInput": {
                "port": ("con1", 9),
            },
            'intermediate_frequency': resonator_IF,
            "operations": {
                "readout": "reflectometry_readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
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
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "octave": {
             'mixInputs': {
                'I': ('con1', 1),
                'Q': ('con1', 2),
                "lo_frequency": qubit_LO,
                "mixer": "octave_octave1_1",  # a fixed name, do not change.
            },
            'intermediate_frequency': qubit_IF,
            "operations": {
                "readout": "octave_pulse",
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
        "bias_P1_pulse": {
            "operation": "control",
            "length": bias_length,
            "waveforms": {
                "single": "bias_P1_pulse_wf",
            },
        },
        "bias_P2_pulse": {
            "operation": "control",
            "length":  bias_length,
            "waveforms": {
                "single": "bias_P2_pulse_wf",
            },
        },
        "cw_pulse": {
            'operation': 'control',
            'length': cw_len,
            'waveforms': {
                'I': 'const_wf',
                'Q': 'zero_wf',
            },
        },
        "gaussian_pulse": {
            'operation': 'control',
            'length': gaussian_length,
            'waveforms': {
                'I': 'gaussian_wf',
                'Q': 'zero_wf',
            },
        },
        "pi_pulse": {
            'operation': 'control',
            'length': pi_length,
            'waveforms': {
                'I': 'pi_wf',
                'Q': 'zero_wf',
            },
        },
        "pi_half_pulse": {
            'operation': 'control',
            'length': pi_length,
            'waveforms': {
                'I': 'pi_half_wf',
                'Q': 'zero_wf',
            },
        },
        'trigger_pulse': {
            'operation': 'control',
            'length': 1000,
            'waveforms': {
                'single': 'zero_wf',
            },
            'digital_marker': 'ON'
        },
        "reflectometry_readout_pulse": {
            'operation': 'measurement',
            'length': reflectometry_readout_length,
            'waveforms': {
                'single': 'reflect_wf',
            },
            'integration_weights': {
                'cos': 'cw_cosine_weights',
                'sin': 'cw_sine_weights',
            },
            'digital_marker': 'ON',
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
        "octave_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_pulse_wf",
                "Q": "readout_pulse_wf",
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
        "readout_pulse_wf": {"type": "constant", "sample": readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        'const_wf': {'type': 'constant', 'sample': cw_amp},
        'reflect_wf': {'type': 'constant', 'sample': reflect_amp},
        "gaussian_wf": {"type": "arbitrary", "samples": [float(arg) for arg in gaussian_amp * gaussian(gaussian_length, gaussian_length / 5)]},
        "pi_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_amp * gaussian(pi_length, pi_length / 5)]},
        "pi_half_wf": {"type": "arbitrary", "samples": [float(arg) for arg in pi_half_amp * gaussian(pi_length, pi_length / 5)]},
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
        'cw_cosine_weights': {
            'cosine': [(1.0, reflectometry_readout_length)],
            'sine': [(0.0, reflectometry_readout_length)],
        },
        'cw_sine_weights': {
            'cosine': [(0.0, reflectometry_readout_length)],
            'sine': [(1.0, reflectometry_readout_length)],
        },
    },
    "mixers": {
        "octave_octave1_1": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": (1, 0, 0, 1),
            },
        ],
    },
}
