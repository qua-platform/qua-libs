import numpy as np
from qualang_tools.units import unit
from set_octave import OctaveUnit, octave_declaration
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


######################
# Network parameters #
######################
qop_ip = "172.16.33.101"  # Write the QM router IP address
cluster_name = "Cluster_81"  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

############################
# Set octave configuration #
############################

# The Octave port is 11xxx, where xxx are the last three digits of the Octave internal IP that can be accessed from
# the OPX admin panel if you QOP version is >= QOP220. Otherwise, it is 50 for Octave1, then 51, 52 and so on.
octave_1 = OctaveUnit("octave1", qop_ip, port=11234, con="con1")
# octave_2 = OctaveUnit("octave2", qop_ip, port=11051, con="con1")

# If the control PC or local network is connected to the internal network of the QM router (port 2 onwards)
# or directly to the Octave (without QM the router), use the local octave IP and port 80.
# octave_ip = "192.168.88.X"
# octave_1 = OctaveUnit("octave1", octave_ip, port=80, con="con1")

# Add the octaves
octaves = [octave_1]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
#       PROBE       #
#####################

probe_amp = -300 * u.mV
probe_len = 20 * u.ns

##########################
#   SPIN RANDOMIZATION   #
##########################

random_amp = 200 * u.mV
random_len = 20 * u.ns

####################
#       PUMP       #
####################

spin_LO = 16 * u.GHz
spin_IF = 103 * u.MHz
# Octave gain in dB
octave_gain = 0

# CW pulse
cw_amp = 0.3  # in V
cw_len = 100  # in ns

# Pi pulse
square_pi_amp = 0.25  # in V
square_pi_len = 120  # in ns
# Pi half
square_pi_half_amp = 0.25  # in V
square_pi_half_len = 60  # in ns
# Gaussian pulse
gaussian_amp = 0.1  # in V
gaussian_len = 120  # in ns
gaussian_sigma = gaussian_len / 5
drag_coef = 0
anharmonicity = -200 * u.MHz
AC_stark_detuning = 0 * u.MHz
gaussian_wf, gaussian_der_wf = np.array(
    drag_gaussian_pulse_waveforms(gaussian_amp, gaussian_len, gaussian_sigma, drag_coef, anharmonicity, AC_stark_detuning)
)
gaussian_I_wf = gaussian_wf
gaussian_Q_wf = gaussian_der_wf

######################
#       READOUT      #
######################

readout_amp = 0.2
readout_len = 100


#############################################
#                  Config                   #
#############################################
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0},  # Pump I quadrature
                2: {"offset": 0.0},  # Pump Q quadrature
                3: {"offset": 0.0},  # Probe
            },
            "digital_outputs": {
                1: {},  # TTL
            },
            "analog_inputs": {
                1: {"offset": 0.0, "gain_db": 0},  # Potential second trigger input
            },
        },
    },
    "elements": {
        "pump": {
            "RF_inputs": {"port": ("octave1", 1)},
            "intermediate_frequency": spin_IF,
            "operations": {
                "cw": "const_pulse",
                "pi": "square_pi_pulse",
                "pi_half": "square_pi_half_pulse",
                "gaussian": "gaussian_pulse",
            },
        },
        "probe": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "operations": {
                "const": "const_probe_pulse",
                "const_random": "const_randomization_pulse",
                "readout": "readout_pulse"

            },
        },
    },
    "octaves": {
        "octave1": {
            "RF_outputs": {
                1: {
                    "LO_frequency": spin_LO,
                    "LO_source": "internal",
                    "output_mode": "always_on",
                    "gain": 0,
                },
            },
            "RF_inputs": {
                1: {
                    "LO_frequency": spin_LO,
                    "LO_source": "internal",
                },
            },
            "connectivity": "con1",
        }
    },
    "pulses": {
        "const_pulse": {
            "operation": "control",
            "length": cw_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "pi_wf",
                "Q": "zero_wf",
            },
        },
        "square_pi_half_pulse": {
            "operation": "control",
            "length": square_pi_len,
            "waveforms": {
                "I": "pi_half_wf",
                "Q": "zero_wf",
            },
        },
        "gaussian_pulse": {
            "operation": "control",
            "length": gaussian_len,
            "waveforms": {
                "I": "gaussian_I_wf",
                "Q": "gaussian_Q_wf",
            },
        },
        "const_probe_pulse": {
            "operation": "control",
            "length": probe_len,
            "waveforms": {
                "single": "const_probe_wf",
            },
        },
        "const_randomization_pulse": {
            "operation": "control",
            "length": random_len,
            "waveforms": {
                "single": "const_randomization_wf",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "readout_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": cw_amp},
        "pi_wf": {"type": "constant", "sample": square_pi_amp},
        "pi_half_wf": {"type": "constant", "sample": square_pi_half_amp},
        "const_probe_wf": {"type": "constant", "sample": probe_amp},
        "const_randomization_wf": {"type": "constant", "sample": random_amp},
        "readout_wf": {"type": "constant", "sample": readout_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gaussian_I_wf": {"type": "arbitrary", "samples": gaussian_I_wf.tolist()},
        "gaussian_Q_wf": {"type": "arbitrary", "samples": gaussian_Q_wf.tolist()},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
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
    },
}