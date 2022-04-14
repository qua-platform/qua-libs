from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import Helpers as helpers

import numpy as np

#########################################
# Connecting to QM server:
#########################################
# qmm = QuantumMachinesManager()


####################
# The Configuration:
####################

mw_qubit_IF = 50e6
opt_qubit_amp_IF = 80e6
opt_qubit_phase_IF = 110e6

LO_freq = 12.3e9

spcm_pulse = -np.load("spcm_pulse.npy")
spcm_pulse2 = -np.load("spcm_pulse2.npy")


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # I spin
                2: {"offset": 0.0},  # Q spin
                3: {"offset": 0.0},  # photon amp
                4: {"offset": 0.0},  # photon phase
            },
            "digital_outputs": {
                1: {},
                2: {},
                3: {},
            },  # laser
            "analog_inputs": {
                1: {"offset": 0.0},  # readout1
                2: {"offset": 0.0},  # readout2
            },
        },
    },
    "elements": {
        "readout": {
            "digitalInputs": {
                "AOM": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "trig_pulse",
            },
        },
        "spin_qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": mw_qubit_IF,
            "operations": {
                "pi": "pi_pulse",
                "pi2": "pi2_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "locking": {
            "digitalInputs": {
                "AOM": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "on": "trig_pulse",
            },
        },
        "opt_qubit_amp": {
            "singleInput": {
                "port": ("con1", 3),
            },
            # 'intermediate_frequency': opt_qubit_amp_IF,
            "digitalInputs": {
                "AOM": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "photon": "photon_pulse",
            },
        },
        "opt_qubit_phase": {
            "singleInput": {
                "port": ("con1", 4),
            },
            # 'intermediate_frequency': opt_qubit_phase_IF,
            "operations": {
                "phase_shift": "phase_shift_pulse",
            },
        },
        "readout1": {
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": 28,
            "smearing": 0,
            "operations": {
                "readout": "readout",
            },
            #######################
            ####### Ignore ########
            #######################
            "digitalInputs": {
                "laser_in": {
                    "port": ("con1", 7),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "singleInput": {"port": ("con1", 1)},
        },
        "readout2": {
            "outputs": {"out1": ("con1", 2)},
            "time_of_flight": 28,
            "smearing": 0,
            # 'outputPulse': [int(arg) for arg in spcm_pulse2],
            "operations": {
                "readout": "readout",
            },
            #######################
            ####### Ignore ########
            #######################
            "digitalInputs": {
                "laser_in": {
                    "port": ("con1", 8),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "singleInput": {"port": ("con1", 1)},
        },
    },
    "pulses": {
        "trig_pulse": {"operation": "control", "length": 40, "digital_marker": "ON"},
        "photon_pulse": {
            "operation": "control",
            "length": 24,
            "waveforms": {
                "single": "gauss_photon_wf",
            },
            "digital_marker": "ON",
        },
        "phase_shift_pulse": {
            "operation": "control",
            "length": 24,
            "waveforms": {
                "single": "const_wf",
            },
        },
        "pi_pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "I": "gauss_pi_wf",
                "Q": "zero_wf",
            },
        },
        "pi2_pulse": {
            "operation": "control",
            "length": 32,
            "waveforms": {
                "I": "gauss_pi2_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "readout": {
            "operation": "measurement",
            "length": 3000,
            "waveforms": {"single": "zero_wf"},  # fake
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "gauss_pi_wf": {
            "type": "arbitrary",
            "samples": helpers.gauss(0.36, 0.0, 4.0, 32),
        },
        "gauss_pi2_wf": {
            "type": "arbitrary",
            "samples": helpers.gauss(0.18, 0.0, 4.0, 32),
        },
        "gauss_photon_wf": {
            "type": "arbitrary",
            "samples": helpers.gauss(0.36, 0.0, 2.0, 24),
        },
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": mw_qubit_IF,
                "lo_frequency": LO_freq,
                "correction": [1, 0, 0, 1],
            }
        ]
    },
}

# qm = qmm.open_qm(config)
