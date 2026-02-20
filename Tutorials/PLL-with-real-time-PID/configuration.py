"""
QUA-Config supporting OPX1000 with LF-FEM
"""
from pathlib import Path
import numpy as np
import plotly.io as pio
from qualang_tools.units import unit

pio.renderers.default = "browser"

###########################################
#            AUXILIARY FUNCTIONS          #
###########################################
u = unit(coerce_to_integer=True)


###########################################
#           Network parameters            #
###########################################


qop_ip = "127.0.0.1"  # Write the QM router IP address
cluster_name = None  # Write your cluster_name if version >= QOP220
qop_port = None  # Write the QOP port if version < QOP220

###########################################
# #               Save Path                 #
###########################################
# Path to save data
save_dir = Path(__file__).parent.resolve() / "Data"
save_dir.mkdir(exist_ok=True)



###########################################
#         OPX port configuration          #
###########################################
con = "con1"

fem1 = 5   # Should be the LF-FEM index, e.g., 1
AOM_port = 7 # OPX analog output port
Detector_port = 1 # OPX analog input port


#############################################
#                Detector                   #
#############################################
IF_Detector = 10 * u.MHz
readout_len = 10000 #ns
time_of_flight = 28

#############################################
#                  AOM                      #
#############################################

IF_AOM = 10 * u.MHz
const_len_AOM = readout_len
const_amp_AOM = 0.1

#############################################
#                General                    #
#############################################

sampling_rate = int(1e9)  # or, int(2e9)
rotation_angle = (0 / 180) * np.pi

#############################################
#                  Config                   #
#############################################
config = {
    # "version": 1,
    "controllers": {
        con: {
            "type": "opx1000",
            "fems": {
                fem1: {
                    "type": "LF",
                    "analog_outputs": {
                        AOM_port: {
                            "offset": 0.0,
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
                            "upsampling_mode": "pulse",
                        },
                        "2": {
                            "offset": 0.0,
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
                            "upsampling_mode": "pulse",
                        },
                    },
                    "digital_outputs": {
                        "1": {},
                    },
                    "analog_inputs": {
                        Detector_port : {"offset": 0, "gain_db": 0, "sampling_rate": sampling_rate}, # PD
                    },
                },
            }
        },
    },
    "elements": {
        "AOM": {
            "singleInput": {
                "port": (con, fem1, AOM_port),
            },
            "intermediate_frequency": IF_AOM,
            "operations": {
                "cw": "const_pulse_AOM",
            },
        },
        'Detector': { # lower sideband
            "singleInput": {"port": (con, fem1, 2)},  # not used
            "digitalInputs": {  # for visualization in simulation
                "marker": {
                    "port": (con, fem1, 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": IF_Detector,
            "outputs": {
                "out1": (con, fem1, Detector_port),
            },
            "time_of_flight": time_of_flight,
            "operations": {
                "readout": "readout_pulse_PD",
            },
        },
    },
    "pulses": {
        "const_pulse_AOM": {
            "operation": "control",
            "length": const_len_AOM,
            "waveforms": {
                "single": "const_wf_AOM",
            },
        },
        "readout_pulse_PD": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf_AOM": {"type": "constant", "sample": const_amp_AOM},
        "zero_wf": {"type": "constant", "sample": 0.0},

    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
        "OFF": {"samples": [(0, 0)]},
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
        },
    },   
}