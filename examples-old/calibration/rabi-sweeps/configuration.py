import numpy as np

## Definition of the sample for Gaussian pulse
gauss_pulse_len = 20  # nsec
Amp = 0.2  # Pulse Amplitude
gauss_arg = np.linspace(-3, 3, gauss_pulse_len)
gauss_wf = np.exp(-(gauss_arg**2) / 2)
gauss_wf = Amp * gauss_wf / np.max(gauss_wf)
readout_pulse_len = 20

## Setting up the configuration of the experimental setup
## Embedded in a Python dictionary
config = {
    "version": 1,
    "controllers": {  # Define the QM device and its outputs, in this case:
        "con1": {  # 2 analog outputs for the in-phase and out-of phase components
            "type": "opx1",  # of the qubit (I & Q), and 2 other analog outputs for the coupled readout resonator
            "analog_outputs": {
                1: {"offset": 0.05},
                2: {"offset": 0.05},
                3: {"offset": -0.024},
                4: {"offset": 0.115},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {  # Define the elements composing the quantum system, i.e the qubit+ readout resonator (RR)
        "qubit": {
            "mixInputs": {
                "I": ("con1", 1),  # Connect the component to one output of the OPX
                "Q": ("con1", 2),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",  ##Associate a mixer entity to control the IQ mixing process
            },
            "intermediate_frequency": 5.15e3,  # Resonant frequency of the qubit
            "operations": {  # Define the set of operations doable on the qubit, each operation is related
                "gauss_pulse": "gauss_pulse_in"  # to a pulse
            },
        },
        "RR": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": 6.00e7,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 6.12e3,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 28,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
    },
    "pulses": {  # Pulses definition
        "meas_pulse_in": {
            "operation": "measurement",
            "length": readout_pulse_len,
            "waveforms": {
                "I": "exc_wf",  # Decide what pulse to apply for each component
                "Q": "zero_wf",
            },
            "integration_weights": {
                "integW1": "integW1",
                "integW2": "integW2",
            },
            "digital_marker": "marker1",
        },
        "gauss_pulse_in": {
            "operation": "control",
            "length": gauss_pulse_len,
            "waveforms": {"I": "gauss_wf", "Q": "zero_wf"},
        },
    },
    "waveforms": {  # Specify the envelope type of the pulses defined above
        "zero_wf": {"type": "constant", "sample": 0.0},
        "exc_wf": {"type": "constant", "sample": 0.479},
        "gauss_wf": {"type": "arbitrary", "samples": gauss_wf.tolist()},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {"cosine": [4.0] * 27, "sine": [0.0] * 27},
        "integW2": {"cosine": [0.0] * 27, "sine": [4.0] * 27},
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 6.12e3,
                "lo_frequency": 6.00e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": 5.15e3,
                "lo_frequency": 5.10e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}
