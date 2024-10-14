import numpy as np


def blackman(t, v_start, v_end):
    """
    Amplitude waveform that minimizes the amount of side lobes in the Fourier domain.
    :param t: pulse duration [ns] (int)
    :param v_start: start amplitude [V] (float)
    :param v_end: end amplitude [V] (float)
    :return:
    """
    time_vector = np.asarray([x * 1.0 for x in range(int(t))])
    black = v_start + (
        time_vector / t
        - (25 / (42 * np.pi)) * np.sin(2 * np.pi * time_vector / t)
        + (1 / (21 * np.pi)) * np.sin(4 * np.pi * time_vector / t)
    ) * (v_end - v_start)
    return black


def print_2d(matrix):
    """
    Nicely prints a 2D array
    :param matrix: 2D python array
    """
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            print(f"{matrix[i][j]}\t", end="")
        print("")


#############
# VARIABLES #
#############
qop_ip = "127.0.0.1"
cluster_name = "my_cluster"
qop_port = 80

octave_config = None
con = "con1"
fem = 1  # This should be the index of the LF-FEM module, e.g., 1

sampling_rate = int(1e9)  # or, int(2e9)

# Units
ms = 1e6

# --> Array geometry
# Number of columns
number_of_columns = 7
# Number of rows
number_of_rows = 7
# Maximum number of tweezers available
max_number_of_tweezers = 7
# Number of configured tweezers, if it increases don't forget to update the "align" in the QUA program
n_tweezers = 7
n_segment_python = 50
# --> Chirp pulse
# Amplitude of each individual tweezer
# WARNING: total output cannot exceed 0.5V
constant_pulse_amplitude = 0.45 / max_number_of_tweezers  # Must be < 0.49/max_number_of_tweezers
# Duration of tweezer frequency chirp
constant_pulse_length = 1 * ms
# Analog readout threshold discriminating between atom and no-atom [V]
threshold = -0.0015

# --> Blackman pulses
# Amplitude of the Blackman pulse which should match the amplitude of the tweezers during frequency chirps
blackman_amplitude = constant_pulse_amplitude
# Duration of the Blackman pulse for ramping up and down the tweezers power
blackman_pulse_length = 0.3 * ms
# Reduced sampling rate for generating long pulses without memory issues
sampling_rate = 100e6  # Used for Blackman_long_pulse_length

# Tweezer column phases
phases_list = [0.1, 0.4, 0.9, 0.3, 0.7, 0.2, 0.5, 0.8, 0.0, 0.6]
# --> Column frequencies
column_spacing = -0.76e6  # in Hz
column_if_first_site = 76.56e6  # in Hz
column_if = [int(column_if_first_site + column_spacing * i) for i in range(number_of_columns)]
# --> Row frequencies
row_spacing = 0.76e6  # in Hz
row_selector_if = 68.6e6  # in Hz
row_frequencies_list = [row_selector_if + row_spacing * x for x in range(number_of_rows)]

# Readout time of the occupation matrix sent by fpga
readout_fpga_len = 60
# Readout duration for acquiring the spectrographs
readout_pulse_length = blackman_pulse_length * 2 + constant_pulse_length

# --> Microwave qubit addressing
qubit_LO = 9.4e9  # Hz
qubit_IF = 100e6  # Hz
constant_mw_pulse_length = 100e3  # ns
constant_mw_pulse_Amp = 0.3  # V

# Voltage offset the column and row analog outputs
row_selector_voltage_offset = 0.0
column_selector_voltage_offset = 0.0
mw_I_voltage_offset = 0.0
mw_Q_voltage_offset = 0.0

# Analog output connected to the column AOD
column_channel = 10
# Analog output connected to the row AOD
row_channel = 1
# Analog output connected the the mw I port
mw_I = 3
# Analog output connected the the mw Q port
mw_Q = 4


config = {
    "version": 1,
    "controllers": {
        con: {
            "type": "opx1000",
            "fems": {
                fem: {
                    "type": "LF",
                    "analog_outputs": {
                        # The "output_mode" can be used to tailor the max voltage and frequency bandwidth, i.e.,
                        #   "direct":    1Vpp (-0.5V to 0.5V), 750MHz bandwidth (default)
                        #   "amplified": 5Vpp (-2.5V to 2.5V), 330MHz bandwidth
                        # "output_mode": "direct",
                        # The "sampling_rate" can be adjusted by using more FEM cores, i.e.,
                        #   1 GS/s: uses one core per output (default)
                        #   2 GS/s: uses two cores per output
                        # NOTE: duration parameterization of arb. waveforms, sticky elements and chirping
                        #       aren't yet supported in 2 GS/s.
                        # "sampling_rate": sampling_rate,
                        # At 1 GS/s, use the "upsampling_mode" to optimize output for
                        #   modulated pulses (optimized for modulated pulses):      "mw"    (default)
                        #   unmodulated pulses (optimized for clean step response): "pulse"
                        # "upsampling_mode": "mw",
                        row_channel: {"offset": row_selector_voltage_offset},  # Row AOD tone
                        column_channel: {"offset": column_selector_voltage_offset},  # Column AOD tone
                        mw_I: {"offset": mw_I_voltage_offset},  # MW I port
                        mw_Q: {"offset": mw_Q_voltage_offset},  # MW Q port
                        7: {"offset": 0.0},  # Fake port for measurement
                    },
                    "digital_outputs": {
                        1: {},
                    },  # Not used yet
                    "analog_inputs": {
                        1: {"offset": 0.0},  # Analog input 1 used for fpga readout
                        2: {"offset": 0.0},  # Not used yet
                    },
                },
            },
        }
    },
    "elements": {
        "qubit": {
            "mixInputs": {
                # Connect the I qubit mixer component to output "mw_I" of the OPX
                "I": (con, fem, mw_I),
                # Connect the Q qubit mixer component to output "mw_Q" of the OPX
                "Q": (con, fem, mw_Q),
                # Qubit local oscillator frequency in Hz (int)
                "lo_frequency": qubit_LO,
                # Associate a mixer entity to control the IQ mixing process
                "mixer": "mixer_qubit",
            },
            # Resonant frequency of the qubit
            "intermediate_frequency": qubit_IF,
            # Define the set of operations doable on the qubit, each operation is related to a pulse
            "operations": {"constant_mw": "constant_mw_pulse"},
        },
        # fpga is used to read the occupation matrix sent by fpga
        "fpga": {
            "singleInput": {"port": (con, fem, 7)},  # Fake output port for measurement
            "intermediate_frequency": 0,
            "operations": {
                "readout_fpga": "readout_fpga_pulse",
            },
            "outputs": {"out1": (con, fem, 1)},
            "time_of_flight": 24,
            "smearing": 0,
        },
        # detector is used to acquire the spectrographs for debuging
        "detector": {
            "singleInput": {
                "port": (con, fem, 7),
            },
            "intermediate_frequency": 0,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {"out1": (con, fem, 2)},
            "time_of_flight": 24,
            "smearing": 0,
        },
        # row_selector is used to control the row AOD
        "row_selector": {
            "singleInput": {
                "port": (con, fem, row_channel),
            },
            "intermediate_frequency": row_selector_if,
            "operations": {
                "blackman_up": "blackman_up_pulse",
                "blackman_down": "blackman_down_pulse",
                "constant": "constant_pulse",
            },
        },
    },
    "pulses": {
        "readout_fpga_pulse": {
            "operation": "measurement",
            "length": readout_fpga_len,
            "waveforms": {"single": "zero_wf"},
            "digital_marker": "ON",
            "integration_weights": {
                "constant": "constant_weights",
            },
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_pulse_length,
            "waveforms": {
                "single": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
            },
            "digital_marker": "ON",
        },
        "blackman_up_pulse": {
            "operation": "control",
            "length": blackman_pulse_length,
            "waveforms": {
                "single": "blackman_up_wf",
            },
            "digital_marker": "ON",
        },
        "blackman_down_pulse": {
            "operation": "control",
            "length": blackman_pulse_length,
            "waveforms": {
                "single": "blackman_down_wf",
            },
            "digital_marker": "ON",
        },
        "constant_pulse": {
            "operation": "control",
            "length": constant_pulse_length,
            "waveforms": {
                "single": "constant_wf",
            },
            "digital_marker": "ON",
        },
        "constant_mw_pulse": {
            "operation": "control",
            "length": constant_mw_pulse_length,
            "waveforms": {"I": "constant_mw_wf", "Q": "zero_wf"},
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "blackman_up_wf": {
            "type": "arbitrary",
            "samples": blackman(blackman_pulse_length / (1e9 / sampling_rate), 0, blackman_amplitude),
            "sampling_rate": sampling_rate,
        },
        "blackman_down_wf": {
            "type": "arbitrary",
            "samples": blackman(blackman_pulse_length / (1e9 / sampling_rate), blackman_amplitude, 0),
            "sampling_rate": sampling_rate,
        },
        "constant_wf": {
            "type": "constant",
            "sample": constant_pulse_amplitude,
        },
        "constant_mw_wf": {
            "type": "constant",
            "sample": constant_mw_pulse_Amp,
        },
        "zero_wf": {
            "type": "constant",
            "sample": 0.0,
        },
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "constant_weights": {
            "cosine": [(1.0, readout_fpga_len)],
            "sine": [(0.0, readout_fpga_len)],
        },
        "cosine_weights": {
            "cosine": [(1.0, readout_pulse_length)],
            "sine": [(0.0, readout_pulse_length)],
        },
        "sine_weights": {
            "cosine": [(0.0, readout_pulse_length)],
            "sine": [(1.0, readout_pulse_length)],
        },
    },
    "mixers": {
        "mixer_qubit": [
            {
                "intermediate_frequency": qubit_IF,
                "lo_frequency": qubit_LO,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
# Iteratively add the column tweezers
for i in range(1, n_tweezers + 1):
    config["elements"][f"column_{i}"] = {
        "singleInput": {
            "port": (con, fem, column_channel),
        },
        "intermediate_frequency": column_if_first_site,
        "operations": {
            "blackman_up": "blackman_up_pulse",
            "blackman_down": "blackman_down_pulse",
            "constant": "constant_pulse",
        },
    }
