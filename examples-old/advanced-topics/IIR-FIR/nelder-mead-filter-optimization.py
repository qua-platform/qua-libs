import time
from typing import List
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import (
    SimulationConfig,
    QuantumMachinesManager,
    LoopbackInterface,
)
import numpy as np
import scipy.signal as signal
import scipy.optimize as opti  # Requires SciPy 1.7.0 or above.

qmm = QuantumMachinesManager()

with program() as filter_optimization:
    stream = declare_stream(adc_trace=True)
    measure("readoutOp", "flux1", stream)
    with stream_processing():
        stream.input1().save("adc")

pulse_len = 128
tof = 248

waveform = [0.0] * 30 + [0.2] * (pulse_len - 60) + [0.0] * 30

# We use an arbitrarily selected filter for distorting the signal
distorted_waveform = signal.lfilter(np.array([1]), np.array([0.95, -0.15, 0.1]), waveform)


def perform(params: List[float]):
    # This is the script which will be called by the optimizer. if bCalc=True, it will do a calculation based on SciPy
    # signal module, otherwise, it will simulate using the OPX
    params = np.array(params)

    feedback_filter = params[:M]
    feedforward_filter = params[M:]
    print("feedback:", feedback_filter)
    print("feedforward:", feedforward_filter)
    if bCalc:  # Use the signal module to simulate the filter behavior.
        scipy_feedback_filter = np.insert(-feedback_filter, 0, np.array([1]))  # SciPy uses a different notation.
        corrected_signal = signal.lfilter(feedforward_filter, scipy_feedback_filter, distorted_waveform)
    else:  # Use the OPX to get the real data (simulated in this case).
        config = {
            "version": 1,
            "controllers": {
                "con1": {
                    "type": "opx1",
                    "analog_outputs": {
                        1: {
                            "offset": +0.0,
                            "filter": {
                                "feedback": feedback_filter,
                                "feedforward": feedforward_filter,
                            },
                        },
                    },
                    "analog_inputs": {
                        1: {"offset": +0.0},
                    },
                },
            },
            "elements": {
                "flux1": {
                    "singleInput": {"port": ("con1", 1)},
                    "outputs": {"output1": ("con1", 1)},
                    "intermediate_frequency": 10,
                    "operations": {
                        "readoutOp": "readoutPulse",
                    },
                    "time_of_flight": tof,
                    "smearing": 0,
                },
            },
            "pulses": {
                "readoutPulse": {
                    "operation": "measure",
                    "length": pulse_len,
                    "waveforms": {"single": "const_wf"},
                    "digital_marker": "ON",
                    "integration_weights": {"x": "xWeights", "y": "yWeights"},
                },
            },
            "waveforms": {
                "const_wf": {"type": "arbitrary", "samples": distorted_waveform},
            },
            "digital_waveforms": {
                "ON": {"samples": [(1, 0)]},
            },
            "integration_weights": {
                "xWeights": {
                    "cosine": [1.0] * (pulse_len // 4),
                    "sine": [0.0] * (pulse_len // 4),
                },
                "yWeights": {
                    "cosine": [0.0] * (pulse_len // 4),
                    "sine": [1.0] * (pulse_len // 4),
                },
            },
        }

        job = qmm.simulate(
            config,
            filter_optimization,
            SimulationConfig(
                duration=150,
                simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)], latency=200),
            ),
        )
        job.result_handles.wait_for_all_values()
        corrected_signal = -job.result_handles.adc.fetch_all() / 4096  # This converts ADC units into volts

    # The correlation is used to calculate the "loss": Check whether the resulting output matches the required waveform,
    # taking into account added delays. Check the readme for more information
    corr = np.correlate(corrected_signal, waveform, "full") / (
        np.sqrt(np.correlate(corrected_signal, corrected_signal) * np.correlate(waveform, waveform))
    )

    loss = 1 - np.max(corr)

    print("loss:", loss)
    if bPlot:
        plt.plot(waveform)
        plt.plot(distorted_waveform)
        plt.plot(corrected_signal * np.sum(waveform) / np.sum(corrected_signal), "--")
        plt.legend(["Target waveform", "Distorted waveform", "Corrected waveform"])
        print(f"delay ~ {np.argmax(corr) - len(waveform) + 1}ns")
    return loss


####################
## Set Parameters ##
####################
n_coeff = 10
M = 2  # number of feedback taps 0, 1, 2.
eps = 1e-10
bCalc = True

initial_simplex = np.zeros([n_coeff + 1, n_coeff])
initial_simplex[0, :] = np.array([0.01] * M + [0.8] + [0.01] * (n_coeff - 1 - M))
for i in range(n_coeff):
    initial_simplex[i + 1, :] = initial_simplex[0, :]
    initial_simplex[i + 1, i] = initial_simplex[0, i] + 0.2

fatol_calc = 1e-7
xatol_calc = 1e-3
fatol = 1e-5
xatol = 1e-2
start_time = time.time()

######################
## Optimize Offline ##
######################
bPlot = False
solver_calc = opti.minimize(
    perform,
    # Ignored when there is a simplex, but this is also a good starting point.
    np.array([0] * M + [1] + [0] * (n_coeff - 1 - M)) + 0.2 * np.random.rand(n_coeff),
    method="Nelder-Mead",
    options={
        "initial_simplex": initial_simplex,
        "fatol": fatol_calc,
        "xatol": xatol_calc,
        "adaptive": False,
    },
    bounds=opti.Bounds(
        [-2 + eps] * (M > 0)
        + [-1 + eps] * (M > 1)  # First feedback tap is bounded at (-2,2)
        + [-1]  # Second feedback tap is bounded at (-1,1)
        * (n_coeff - (M > 0) - (M > 1)),  # feedforward taps are bounded at [-1,1]
        [2 - eps] * (M > 0)
        + [1 - eps] * (M > 1)  # First feedback tap is bounded at (-2,2)
        + [1]  # Second feedback tap is bounded at (-1,1)
        * (n_coeff - (M > 0) - (M > 1)),  # feedforward taps are bounded at [-1,1]
    ),
)

###################
## Plotting Part ##
###################
plt.figure()
scipy_feedback_filter = np.insert(-solver_calc.x[:M], 0, np.array([1]))  # SciPy uses a different notation.

corrected_waveform = signal.lfilter(np.array(solver_calc.x[M:]), scipy_feedback_filter, distorted_waveform)
norm = np.sum(waveform) / np.sum(corrected_waveform)

corrected_waveform = signal.lfilter(np.array(solver_calc.x[M:] * norm), scipy_feedback_filter, distorted_waveform)

plt.plot(waveform)
plt.plot(distorted_waveform)
plt.plot(corrected_waveform, "--")
plt.legend(["Target waveform", "Distorted waveform", "Corrected waveform"])
plt.title("Output according to the SciPy signal module - 1st iteration")

bPlot = True
bCalc = False
plt.figure()
perform(solver_calc.x)
plt.title("Output according to the OPX - 1st iteration")
plt.show()

#####################
## Optimize Online ##
#####################

bPlot = False
solver = opti.minimize(
    perform,
    solver_calc.x,
    method="Nelder-Mead",
    options={
        "fatol": fatol,
        "xatol": xatol,
        "adaptive": False,
    },
    bounds=opti.Bounds(
        [-2 + eps] * (M > 0)
        + [-1 + eps] * (M > 1)  # First feedback tap is bounded at (-2,2)
        + [-1]  # Second feedback tap is bounded at (-1,1)
        * (n_coeff - (M > 0) - (M > 1)),  # feedforward taps are bounded at [-1,1]
        [2 - eps] * (M > 0)
        + [1 - eps] * (M > 1)  # First feedback tap is bounded at (-2,2)
        + [1]  # Second feedback tap is bounded at (-1,1)
        * (n_coeff - (M > 0) - (M > 1)),  # feedforward taps are bounded at [-1,1]
    ),
)

###################
## Plotting Part ##
###################
plt.figure()
scipy_feedback_filter = np.insert(-solver_calc.x[:M], 0, np.array([1]))  # SciPy uses a different notation.
corrected_waveform = signal.lfilter(np.array(solver_calc.x[M:]), scipy_feedback_filter, distorted_waveform)
norm = np.sum(waveform) / np.sum(corrected_waveform)
corrected_waveform = signal.lfilter(np.array(solver_calc.x[M:] * norm), scipy_feedback_filter, distorted_waveform)

plt.plot(waveform)
plt.plot(distorted_waveform)
plt.plot(corrected_waveform, "--")
plt.legend(["Target waveform", "Distorted waveform", "Corrected waveform"])
plt.title("Output according to the SciPy signal module - 2nd iteration")

bPlot = True
bCalc = False
plt.figure()
perform(solver_calc.x)
plt.title("Output according to the OPX - 2nd iteration")

#########
print(f"Full optimization took {int((time.time() - start_time)//60)}:{int((time.time() - start_time)%60)} minutes")
