import numpy as np
from scipy import signal

IF_freq = 50e6 * 0
LO_freq = 2.87e9 - IF_freq

meas_len = 300
long_meas_len = 50000

time_of_flight = 180

pi_len = 32
pi_amp = 0.4

pi_half_len = pi_len
pi_half_amp = 0.2

AOM_freq = 0 * 200e6  # Put zero for AM control, or AOM freq for direct control

fsm_vpp = 0.99  # Need to calibrate to be 10Vpp after the amplifier (input to FSM)
um_over_vpp = 20  # How many ums are in the full range
um_over_v = um_over_vpp / 2  # How many ums are in the amplitude (half range)
um_step = fsm_vpp / um_over_vpp  # 1um

simulate = True


def IQ_imbalance(g, phi):
    c = np.cos(phi)
    s = np.sin(phi)
    n = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(n * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": 0.0},  # I
                2: {"offset": 0.0},  # Q
                3: {
                    "offset": 0.0,
                    "filter": {"feedforward": [2**-9], "feedback": [1 - 2**-9]},
                },  # FSM X
                4: {
                    "offset": 0.0,
                    "filter": {"feedforward": [2**-9], "feedback": [1 - 2**-9]},
                },  # FSM Y
                5: {"offset": 0.0},  # AOM
            },
            "digital_outputs": {
                # 1: {},  # Laser
                2: {},  # Spcm1
                3: {},  # Spcm2
            },
            "analog_inputs": {
                1: {"offset": 0},  # Spcm1
                2: {"offset": 0},  # Spcm2
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_freq,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": IF_freq,
            "operations": {
                "cw": "cw_pulse",
                "pi": "pi_pulse",
                "pi_half": "pi_half_pulse",
            },
        },
        "laser_AOM": {
            "singleInput": {"port": ("con1", 5)},
            "intermediate_frequency": AOM_freq,
            "operations": {
                "init": "laser_init_pulse",
            },
        },
        "spcm1": {
            "singleInput": {"port": ("con1", 1)},  # fake
            "digitalInputs": {
                "gate": {
                    "port": ("con1", 2),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 1)},
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "outputPulseParameters": {  # Analog input limited to 0.5V, need to attenuate SPCM
                "signalThreshold": 300,  # 1 == 1/2^11 ~~ 0.5mV
                "signalPolarity": "Descending",  # ADC input is inverted, so need to be decreasing.
                "derivativeThreshold": 50,
                "derivativePolarity": "Descending",
            },
        },
        "spcm2": {
            "singleInput": {"port": ("con1", 1)},  # fake
            "digitalInputs": {
                "gate": {
                    "port": ("con1", 3),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "operations": {
                "readout": "readout_pulse",
                "long_readout": "long_readout_pulse",
            },
            "outputs": {"out1": ("con1", 2)},
            "time_of_flight": time_of_flight,
            "smearing": 0,
            "outputPulseParameters": {  # Analog input limited to 0.5V, need to attenuate SPCM
                "signalThreshold": 300,  # 1 == 1/2^11 ~~ 0.5mV
                "signalPolarity": "Descending",  # ADC input is inverted, so need to be decreasing.
                "derivativeThreshold": 50,
                "derivativePolarity": "Descending",
            },
        },
        "FSM_X": {
            "singleInput": {
                "port": ("con1", 3),
            },
            "hold_offset": {"duration": 1},  # This makes it a sticky element, means that it "holds" the voltage
            "operations": {
                "um_step": "stepPulse",
                "half_range": "edgePulse",
            },
        },
        "FSM_Y": {
            "singleInput": {
                "port": ("con1", 4),
            },
            "hold_offset": {"duration": 1},  # This makes it a sticky element, means that it "holds" the voltage
            "operations": {
                "um_step": "stepPulse",
                "half_range": "edgePulse",
            },
        },
    },
    "pulses": {
        "cw_pulse": {
            "operation": "control",
            "length": 500,
            "waveforms": {"I": "const_wf", "Q": "zero_wf"},
        },
        "pi_pulse": {
            "operation": "control",
            "length": pi_len,
            "waveforms": {"I": "pi_gauss_wf", "Q": "zero_wf"},
        },
        "pi_half_pulse": {
            "operation": "control",
            "length": pi_half_len,
            "waveforms": {"I": "pi_half_gauss_wf", "Q": "zero_wf"},
        },
        "laser_init_pulse": {
            "operation": "control",
            "length": 2000,
            "waveforms": {"single": "laser_wf"},
            "digital_marker": "ON",  # Needed if there's also a digital switch
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": meas_len,
            "digital_marker": "ON",
            "waveforms": {"single": "zero_wf"},  # fake
        },
        "long_readout_pulse": {
            "operation": "measurement",
            "length": long_meas_len,
            # 'digital_marker': 'ON',
            "waveforms": {"single": "zero_wf"},  # fake
        },
        "stepPulse": {
            "operation": "control",
            "length": 16,  # Minimal time because it is being used by a sticky element
            "waveforms": {"single": "step_wf"},
        },
        "edgePulse": {
            "operation": "control",
            "length": 16,  # Minimal time because it is being used by a sticky element
            "waveforms": {"single": "edge_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.4},
        "pi_gauss_wf": {
            "type": "arbitrary",
            "samples": (pi_amp * signal.windows.gaussian(pi_len, pi_len / 5)).tolist(),
        },
        "pi_half_gauss_wf": {
            "type": "arbitrary",
            "samples": (pi_half_amp * signal.windows.gaussian(pi_half_len, pi_half_len / 5)).tolist(),
        },
        "laser_wf": {"type": "constant", "sample": 0.499},  # Need to calibrate
        "zero_wf": {"type": "constant", "sample": 0.0},
        "step_wf": {"type": "constant", "sample": um_step},
        "edge_wf": {"type": "constant", "sample": fsm_vpp / 2},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},  # (on/off, ns)
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": IF_freq,
                "lo_frequency": LO_freq,
                "correction": IQ_imbalance(0, 0),
            },  # SSB
            {
                "intermediate_frequency": 0,
                "lo_frequency": LO_freq,
                "correction": IQ_imbalance(0, 0),
            },  # SSB
        ]
    },
}
