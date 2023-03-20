import numpy as np

from qualang_tools.units import unit

#######################
# AUXILIARY FUNCTIONS #
#######################


#############
# VARIABLES #
#############

u = unit()
host = "129.67.84.123"

measurement_length = 2048  # clock cycles (4 ns)
rf_frequency = 116e6

## waveform parameters

const_amplitude = 0.2

ramp_wf_start = 0.0
ramp_wf_end = 0.5
ramp_wf_samples = 100
ramp_waveform = np.linspace(ramp_wf_start, ramp_wf_end, ramp_wf_samples)

measure_amplitude = 0.05

top_hat_low = 0.0
top_hat_high = 0.5
top_hat_waveform = [top_hat_low] * 1 + [top_hat_high] * 14 + [top_hat_low] * 1

ramp_pulse_length = 100
constant_pulse_length = 16
jump_pulse_length = 16
top_hat_pulse_length = 16


##########
# CONFIG #
##########

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "analog_outputs": {
                1: {"offset": 0.0, "filter": {"feedforward": [-1]}},
                2: {"offset": 0.0, "filter": {"feedforward": [-1]}},
                3: {"offset": 0.0, "filter": {"feedforward": [-1]}},  # dot3
                4: {"offset": 0.0, "filter": {"feedforward": [-1]}},  # charge_sensor gate
                5: {"offset": 0.0},  # rf_in
                6: {"offset": 0.0, "filter": {"feedforward": [-1]}},
            },
            "digital_outputs": {},
            "analog_inputs": {1: {"offset": 0.0}, 2: {"offset": 0.0}},
        }
    },
    "elements": {
        "gate1": {
            "singleInput": {"port": ("con1", 3)},
            "hold_offset": {"duration": 200},
            "operations": {"constant": "constant", "top_hat": "top_hat", "ramp": "ramp"},
        },
        "gate2": {
            "singleInput": {"port": ("con1", 4)},
            "hold_offset": {"duration": 200},
            "operations": {"constant": "constant", "top_hat": "top_hat", "ramp": "ramp"},
        },
        "charge_sensor_ohmic": {
            "singleInput": {"port": ("con1", 1)},
            "time_of_flight": 236,
            "smearing": 0,
            "intermediate_frequency": rf_frequency,
            "outputs": {"out1": ("con1", 1)},
            "operations": {"measure": "measure"},
        },
    },
    "pulses": {
        "ramp": {"operation": "control", "length": ramp_pulse_length, "waveforms": {"single": "ramp"}},
        "constant": {
            "operation": "control",
            "length": constant_pulse_length,
            "waveforms": {"single": "constant"},
        },
        "jump": {
            "operation": "control",
            "length": jump_pulse_length,
            "waveforms": {"single": "constant"},
        },
        "top_hat": {
            "operation": "control",
            "length": top_hat_pulse_length,
            "waveforms": {"single": "top_hat"},
        },
        "measure": {
            "operation": "measurement",
            "length": measurement_length,
            "waveforms": {"single": "measure"},
            "digital_marker": "ON",
            "integration_weights": {
                "integW1": "measure_I",
                "integW2": "measure_Q",
            },
        },
    },
    "waveforms": {
        "constant": {"type": "constant", "sample": const_amplitude},
        "ramp": {"type": "arbitrary", "samples": ramp_waveform},
        "measure": {"type": "constant", "sample": measure_amplitude},
        "top_hat": {"type": "arbitrary", "samples": top_hat_waveform},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {
        "measure_I": {"cosine": [(1.0, measurement_length)], "sine": [(0.0, measurement_length)]},
        "measure_Q": {"cosine": [(0.0, measurement_length)], "sine": [(1.0, measurement_length)]},
    },
    "mixers": {},
}
