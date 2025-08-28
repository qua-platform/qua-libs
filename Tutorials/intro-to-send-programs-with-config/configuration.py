"""
QUA-Config supporting OPX1000 w/ LF-FEM + MW-FEM
"""

import numpy as np
import plotly.io as pio
from qualang_tools.units import unit
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)

def find_delay(adc):
    # Filter the data to get the pulse arrival time
    signal = savgol_filter(np.abs(adc), 11, 3)
    # Detect the arrival of the readout signal
    th = (np.mean(signal[:100]) + np.mean(signal[:-100])) / 2
    delay = np.where(signal > th)[0][0]
    delay = np.round(delay / 4) * 4  # Find the closest multiple integer of 4ns
    print(f"delay = {delay}ns")
    return delay

def plot_adc(adc,adc_single_run,delay):
    fig = plt.figure()
    plt.subplot(121)
    plt.title("Single run")
    plt.plot(adc_single_run.real, "b", label="I")
    plt.plot(adc_single_run.imag, "r", label="Q")
    plt.axvline(delay, color="k", linestyle="--", label="TOF")
    plt.fill_between(range(len(adc_single_run)), -0.5, 0.5, color="grey", alpha=0.2, label="ADC Range")
    plt.xlabel("Time [ns]")
    plt.ylabel("Signal amplitude [V]")
    plt.legend()
    plt.subplot(122)
    plt.title("Averaged run")
    plt.plot(adc.real, "b", label="I")
    plt.plot(adc.imag, "r", label="Q")
    plt.axvline(delay, color="k", linestyle="--", label="TOF")
    plt.xlabel("Time [ns]")
    plt.legend()
    plt.grid("all")
    plt.tight_layout()
    plt.show()
    return fig


######################
# Network parameters #
######################
qop_ip = "127.0.0.1"  # Write the QM router IP address
cluster_name = None  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220


#####################
# OPX configuration #
#####################
con = "con1"
lf_fem = 5
mw_fem = 1
# Set octave_config to None if no octave are present
octave_config = None

#############################################
#                  Qubits                   #
#############################################
sampling_rate = int(1e9)  # or, int(2e9)

qubit_LO_q1 = 3.95 * u.GHz
# Qubits IF
qubit_IF_q1 = 50 * u.MHz
qubit_power = 1  # power in dBm at waveform_amp = 1 (steps of 3 dB)

# Note: amplitudes can be -1..1 and are scaled up to `qubit_power` at amp=1
# CW pulse parameter
const_len = 1000
const_amp = 0.35
# Saturation_pulse
saturation_len = 10 * u.us
saturation_amp = 0.35

##########################################
#               Flux line                #
##########################################
flux_point = 0.0

const_flux_len = 200
const_flux_amp = 0.45

#############################################
#                Resonators                 #
#############################################
resonator_LO = 6.35 * u.GHz
# Resonators IF
resonator_IF_q1 = 75 * u.MHz
resonator_power = 1  # power in dBm at waveform_amp = 1 (steps of 3 dB)

# Note: amplitudes can be -1..1 and are scaled up to `qubit_power` at amp=1
# Readout pulse parameters
readout_len = 4000
readout_amp_q1 = 0.35

# TOF and depletion time
time_of_flight = 28  # must be a multiple of 4
depletion_time = 2 * u.us


#############################################
#                  Config                   #
#############################################

controller_config = {
        "controllers": {
        con: {
            "type": "opx1000",
            "fems": {
                mw_fem: {
                    "type": "MW",
                    "analog_outputs": {
                        # Resonator XY
                        1: {
                            "band": 2,
                            "full_scale_power_dbm": resonator_power,
                            "upconverters": {1: {"frequency": resonator_LO}},
                        },
                        # Qubit 1 XY
                        2: {
                            "band": 1,
                            "full_scale_power_dbm": qubit_power,
                            "upconverters": {1: {"frequency": qubit_LO_q1}},
                        }
                    },
                    "digital_outputs": {},
                    "analog_inputs": {
                        1: {"band": 2, "downconverter_frequency": resonator_LO},  # for down-conversion
                    },
                },
                lf_fem: {
                    "type": "LF",
                    "analog_outputs": {
                        # Q1 flux line
                        1: {
                            "offset": flux_point,
                            "output_mode": "amplified",
                            "sampling_rate": sampling_rate,
                            "upsampling_mode": "pulse",
                            "delay": 141 * u.ns,
                            "filter":{
                                'feedforward': [0.8, 0.3], 
                                "exponential": [(0.1, 0.1 * u.us)],
                            }
                        },
                    },
                    "digital_outputs": {
                        1: {},
                    },
                },
            },
        }
    },
}
logical_config = {
    "elements": {
        "rr1": {
            "MWInput": {
                "port": (con, mw_fem, 1),
                "upconverter": 1,
            },
            "intermediate_frequency": resonator_IF_q1,  # frequency at offset ch7
            "operations": {
                "cw": "const_pulse",
                "readout": "readout_pulse_q1",
            },
            "MWOutput": {
                "port": (con, mw_fem, 1),
            },
            "time_of_flight": time_of_flight,
            "smearing": 0,
        },
        "q1_xy": {
            "MWInput": {
                "port": (con, mw_fem, 2),
                "upconverter": 1,
            },
            "intermediate_frequency": qubit_IF_q1,  # frequency at offset ch7 (max freq)
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
            },
        },
        "q1_z": {
            "singleInput": {
                "port": (con, lf_fem, 1),
            },
            "operations": {
                "const": "const_flux_pulse",
            },
        }
    },
    "pulses": {
        "const_flux_pulse": {
            "operation": "control",
            "length": const_flux_len,
            "waveforms": {
                "single": "const_flux_wf",
            },
        },
        "const_pulse": {
            "operation": "control",
            "length": const_len,
            "waveforms": {
                "I": "const_wf",
                "Q": "zero_wf",
            },
        },
        "saturation_pulse": {
            "operation": "control",
            "length": saturation_len,
            "waveforms": {
                "I": "saturation_wf",
                "Q": "zero_wf",
            },
        },
        "readout_pulse_q1": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "I": "readout_wf_q1",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights"
            },
            "digital_marker": "ON",
        }
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": const_amp},
        "saturation_wf": {"type": "constant", "sample": saturation_amp},
        "const_flux_wf": {"type": "constant", "sample": const_flux_amp},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "readout_wf_q1": {"type": "constant", "sample": readout_amp_q1}
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
        "minus_sine_weights": {
            "cosine": [(0.0, readout_len)],
            "sine": [(-1.0, readout_len)],
        }
    },
}
full_config = controller_config | logical_config
