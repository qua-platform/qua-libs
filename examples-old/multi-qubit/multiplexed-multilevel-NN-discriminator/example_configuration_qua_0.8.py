import numpy as np


############################
# simulation helpers #
############################


def simulate_pulse(IF_freq, chi, k, Ts, Td, power):
    I = [0]
    Q = [0]
    # solve numerically a simplified version of the readout resonator
    for t in range(Ts):
        I.append(I[-1] + (power / 2 - k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (power / 2 - k * Q[-1] - I[-1] * chi))

    for t in range(Td - 1):
        I.append(I[-1] + (-k * I[-1] + Q[-1] * chi))
        Q.append(Q[-1] + (-k * Q[-1] - I[-1] * chi))

    I = np.array(I)
    Q = np.array(Q)
    t = np.arange(len(I))

    S = I * np.cos(2 * np.pi * IF_freq * t * 1e-9) + Q * np.sin(2 * np.pi * IF_freq * t * 1e-9)

    return t, I, Q, S


###########################
# config params #
###########################

lo_freq = 7.1e9
lo_freq_qb = 5.1e9

rr_num = 10

qb_freqs = [
    51.223e6,
    63.635e6,
    72.11e6,
    81.784e6,
    95.533e6,
    102.109e6,
    110.662e6,
    124.235e6,
    129.1e6,
    140e6,
]
freqs = [
    51.223e6,
    63.635e6,
    72.11e6,
    81.784e6,
    95.533e6,
    102.109e6,
    110.662e6,
    124.235e6,
    129.1e6,
    140e6,
]

time_of_flight = 100
readout_len = 480

i_port = 10
q_port = 5

############################
# generate waveforms #
############################

Ts = readout_len - 200
Td = 200
power = 0.2
k = 0.04
chi = 0.023 * np.array([-1, 1, 5])

t_, I_, Q_, S_ = [""] * rr_num, [""] * rr_num, [""] * rr_num, [""] * rr_num
state = [0, 1, 1, 2, 0]
num_of_levels = 3
for i in range(num_of_levels):
    t_[i], I_[i], Q_[i], S_[i] = simulate_pulse(freqs[1], chi[i], k, Ts, Td, power)

divide_signal_factor = 60  # scale to get the signal within -0.5,0.5 range
############################
############################
############################

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {i: {"offset": 0} for i in range(1, 11)},
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        },
        "con2": {
            "type": "opx1",
            "analog_outputs": {i: {"offset": 0} for i in range(1, 11)},
            "digital_outputs": {
                1: {},
            },
            "analog_inputs": {
                1: {"offset": 0},
                2: {"offset": 0},
            },
        },
    },
    "elements": {},
    "pulses": {
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "zero_wf", "Q": "const_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "const_wf": {"type": "constant", "sample": 0.1},
    },
    "digital_waveforms": {"ON": {"samples": [(1, 0)]}},
    "integration_weights": {},
    "mixers": {},
}

for i in range(rr_num):
    config["elements"]["rr" + str(i)] = {
        "group": str(i),
        "mixInputs": {
            "I": ("con" + str(int(i / 5) + 1), i_port),
            "Q": ("con" + str(int(i / 5) + 1), q_port),
            "lo_frequency": lo_freq,
            "mixer": "mixer_rr" + str(i),
        },
        "intermediate_frequency": freqs[i],
        "operations": {
            "readout": "readout_pulse_" + str(i),
        },
        "outputs": {
            "out1": ("con" + str(int(i / 5) + 1), 1),
            "out2": ("con" + str(int(i / 5) + 1), 2),
        },
        "time_of_flight": time_of_flight,
        "smearing": 0,
    }
    config["pulses"]["readout_pulse_" + str(i)] = {
        "operation": "measurement",
        "length": readout_len,
        "waveforms": {"I": "const_wf", "Q": "const_wf"},
        "integration_weights": {},
        "digital_marker": "ON",
    }
    config["waveforms"]["const_wf"] = {"type": "constant", "sample": 0.2}
    config["elements"]["qb" + str(i)] = {
        "group": str(i),
        "mixInputs": {
            "I": (
                "con" + str(int(i / 5) + 1),
                (2 * i + 1) % 10 + ((2 * i + 1) % 10 == 0) * 10,
            ),
            "Q": (
                "con" + str(int(i / 5) + 1),
                (2 * i + 2) % 10 + ((2 * i + 2) % 10 == 0) * 10,
            ),
            "lo_frequency": lo_freq_qb,
            "mixer": "mixer_qb" + str(i),
        },
        "intermediate_frequency": qb_freqs[i],
        "operations": {
            "prepare0": "prepare_pulse_0",
            "prepare1": "prepare_pulse_1",
            "prepare2": "prepare_pulse_2",
        },
        "outputs": {
            "out1": ("con" + str(int(i / 5) + 1), 1),
            "out2": ("con" + str(int(i / 5) + 1), 2),
        },
        "time_of_flight": time_of_flight,
        "smearing": 0,
    }
    config["mixers"]["mixer_rr" + str(i)] = [
        {
            "intermediate_frequency": freqs[i],
            "lo_frequency": lo_freq,
            "correction": [1, 0, 0, 1],
        },
    ]
    config["mixers"]["mixer_qb" + str(i)] = [
        {
            "intermediate_frequency": qb_freqs[i],
            "lo_frequency": lo_freq_qb,
            "correction": [1, 0, 0, 1],
        },
    ]

for i in range(num_of_levels):
    config["pulses"]["prepare_pulse_" + str(i)] = {
        "operation": "measurement",
        "length": readout_len,
        "waveforms": {"I": "I_wf_" + str(i), "Q": "Q_wf_" + str(i)},
        "integration_weights": {},
        "digital_marker": "ON",
    }
    config["waveforms"]["I_wf_" + str(i)] = {
        "type": "arbitrary",
        "samples": [float(arg / divide_signal_factor) for arg in I_[i]],
    }
    config["waveforms"]["Q_wf_" + str(i)] = {
        "type": "arbitrary",
        "samples": [float(arg / divide_signal_factor) for arg in Q_[i]],
    }
    config["waveforms"]["const_wf"] = {"type": "constant", "sample": 0.2}
