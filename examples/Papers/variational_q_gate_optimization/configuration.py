import numpy as np
from qm.qua import *
import time
import math
readout_len = 200
π = np.pi
def gauss(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_wave = amplitude * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_wave = gauss_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_wave]


def gauss_der(amplitude, mu, sigma, delf, length):
    t = np.linspace(-length / 2, length / 2, length)
    gauss_der_wave = (
        amplitude * (-2 * (t - mu)) * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))
    )
    # Detuning correction Eqn. (4) in Chen et al. PRL, 116, 020501 (2016)
    gauss_der_wave = gauss_der_wave * np.exp(2 * np.pi * delf * t)
    return [float(x) for x in gauss_der_wave]


# X90 definition

x90amp = 0.4
x90std = 0.2
x90mean = 0
x90duration = 1000
x90detuning = 0
x90waveform = gauss(
    x90amp, x90mean, x90std, x90detuning, x90duration
)  # Assume you have calibration for a X90 pulse
lmda = 0.5  # Define scaling parameter for Drag Scheme
alpha = -1  # Define anharmonicity parameter
x90der_waveform = gauss_der(x90amp, x90mean, x90std, x90detuning, x90duration)

# Y180 definition
y180waveform = gauss(
    2 * x90amp, x90mean, x90std, x90detuning, x90duration
)  # Assume you have calibration for a X90 pulse
y180der_waveform = gauss_der(2 * x90amp, x90mean, x90std, x90detuning, x90duration)

config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
                3: {"offset": +0.0},
                4: {"offset": +0.0},
                5: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
                2: {"offset": +0.0},
            }
        }
    },
    "elements": {
        "qubit0": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 0,  # 5.2723e9,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
            },
        },
        "CPW0": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": 6.00e7,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 0,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 1)},
        },
        "qubit1": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": 5.10e7,
                "mixer": "mixer_qubit",
            },
            "intermediate_frequency": 0,  # 5.2122e9,
            "operations": {
                "X90": "X90_pulse",
                "Y90": "Y90_pulse",
                "Y180": "Y180_pulse",
            },
        },
        "CPW1": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": 6.00e7,
                "mixer": "mixer_res",
            },
            "intermediate_frequency": 0,  # 6.12e7,
            "operations": {
                "meas_pulse": "meas_pulse_in",
            },
            "time_of_flight": 180,  # Measurement parameters
            "smearing": 0,
            "outputs": {"out1": ("con1", 2)},
        },
        "coupler01": {
            "singleInput": {"port": ("con1", 5)},
            "digitalInputs": {
                "digital_input1": {
                    "port": ("con1", 1),
                    "delay": 0,
                    "buffer": 0,
                },
            },
            "intermediate_frequency": 5e6,
            "operations": {
                "CR_Op": "constPulse",
            },
        }
    },
    "pulses": {
        "meas_pulse_in": {  # Readout pulse
            "operation": "measurement",
            "length": readout_len,
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
        "X90_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "x90_wf", "Q": "x90_der_wf"},
        },
        "Y90_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "x90_der_wf", "Q": "x90_wf"},
        },
        "Y180_pulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"I": "y180_der_wf", "Q": "y180_wf"},
        },
        "constPulse": {
            "operation": "control",
            "length": 1000,
            "waveforms": {"single": "const_wf"}
        }
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
        "zero_wf": {"type": "constant", "sample": 0.0},
        "x90_wf": {"type": "arbitrary", "samples": x90waveform},
        "x90_der_wf": {"type": "arbitrary", "samples": x90der_waveform},
        "y180_wf": {"type": "arbitrary", "samples": y180waveform},
        "y180_der_wf": {"type": "arbitrary", "samples": y180der_waveform},
        "exc_wf": {"type": "constant", "sample": 0.479},
    },
    "digital_waveforms": {"marker1": {"samples": [(1, 4), (0, 2), (1, 1), (1, 0)]}},
    "integration_weights": {  # Define integration weights for measurement demodulation
        "integW1": {
            "cosine": [4.0] * readout_len,

            "sine": [0.0] * readout_len

        },
        "integW2": {
            "cosine": [0.0] * readout_len,
            "sine": [4.0] * readout_len

        },
    },
    "mixers": {  # Potential corrections to be brought related to the IQ mixing scheme
        "mixer_res": [
            {
                "intermediate_frequency": 0,
                "lo_frequency": 6.00e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
        "mixer_qubit": [
            {
                "intermediate_frequency": 0,
                "lo_frequency": 5.10e7,
                "correction": [1.0, 0.0, 0.0, 1.0],
            }
        ],
    },
}

def generate_binary(n):  # Define a function to generate a list of binary strings (Python function, not related to QUA)

    # 2^(n-1)  2^n - 1 inclusive
    bin_arr = range(0, int(math.pow(2, n)))
    bin_arr = [bin(i)[2:] for i in bin_arr]

    # Prepending 0's to binary strings
    max_len = len(max(bin_arr, key=len))
    bin_arr = [i.zfill(max_len) for i in bin_arr]

    return bin_arr


def cost_function_C(x, G):  # Cost function for MaxCut problem, needs to be adapted to the considered optimization problem

    E = G.edges()
    if len(x) != len(G.nodes()):
        return np.nan

    C = 0
    for edge in E:
        e1 = edge[0]
        e2 = edge[1]
        w = G[e1][e2]["weight"]
        C = C + w * x[e1] * (1 - x[e2]) + w * x[e2] * (1 - x[e1])

    return C


# QUA macros (pulse definition of quantum gates)

def CR(ctrl, tgt):  # gate created based on the implementation on IBM in the following paper : https://arxiv.org/abs/2004.06755
    return None


def Rz(𝜆, tgt):
    frame_rotation(-𝜆, tgt)


def Ry(𝜆, tgt):
    play("Y90"*amp(𝜆/(π/2)),tgt)


def Rx(𝜆, tgt):
    play("X90"*amp(𝜆/(π/2)),tgt)



def measurement(RR, I, Q):  # Simple measurement command, could be changed/generalized
    measure("meas_pulse", RR, None, ("integW1", I), ("integW2", Q))


# Stream processing QUA macros
def raw_saving(I, Q, I_stream, Q_stream):
    # Saving command
    save(I, I_stream)
    save(Q, Q_stream)


def state_saving(
        I, Q, state_estimate, stream
):  # Do state estimation protocol in QUA, and save the associated state
    # Define coef a & b defining the line separating states 0 & 1 in the IQ Plane (calibration required), here a & b are arbitrary
    a = declare(fixed, value=1.0)
    b = declare(fixed, value=1.0)
    with if_(Q - a * I - b > 0):
        assign(state_estimate, 1)
    with else_():
        assign(state_estimate, 0)
    save(state_estimate, stream)


# Main Python functions for launching QAOA algorithm


def encode_angles_in_IO(γ, β):  # Insert angles values using IO1 by keeping track of where we are in the QUA program
    if len(γ) != len(β):
        raise IndexError
    for i in range(len(γ)):
        while not (job.is_paused()):
            time.sleep(0.01)
        QM.set_io1_value(γ[i])
        QM.set_io2_value(β[i])
        job.resume()


def quantum_avg_computation(angles):  # Calculate Hamiltonian expectation value (cost function to optimize)

    γ = angles[0: 2 * p: 2]
    β = angles[1: 2 * p: 2]
    job.resume()

    encode_angles_in_IO(γ, β)

    # Implement routine here to set IO values and resume the program, and adapt prior part (or remove it)

    while not (job.is_paused()):
        time.sleep(0.0001)
    results = job.result_handles
    # time.sleep()
    output_states = []
    # Those commands do retrieve the results in generic variables called state#i with #i being a number between 0 and n-1
    # Those results are then stored in a bigger array called output_states
    timing = results.timing.fetch_all()["value"]
    print(timing[-1])
    for d in range(n):
        output_states.append(results.get("state" + str(d)).fetch_all())

    counts = (
        {}
    )  # Dictionary containing statistics of measurement of each bitstring obtained
    for i in range(2 ** n):
        counts[generate_binary(n)[i]] = 0

    for i in range(
            N_shots
    ):  # Here we build in the statistics by picking line by line the bistrings obtained in measurements
        bitstring = ""
        for j in range(len(output_states)):
            bitstring += str(output_states[j][i])
        counts[bitstring] += 1

    avr_C = 0

    for bitstr in list(
            counts.keys()
    ):  # Computation of expectation value according to simulation results

        # use sampled bit string x to compute C(x)
        x = [int(num) for num in list(bitstr)]
        tmp_eng = cost_function_C(x)

        # compute the expectation value and energy distribution
        avr_C = avr_C + counts[bitstr] * tmp_eng

    Mp_sampled = avr_C / N_shots

    return -Mp_sampled  # minus sign because optimizer will minimize the function,
    # to get the maximum value one takes the absolute value of the yielded result of  optimization
