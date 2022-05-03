# """
# filters_with_sp.py: Demonstrate applying filters and fft on the Stream Processor
# Author: Yoav Romach - Quantum Machines
# Created: 31/12/2020
# Created on QUA version: 0.7.411
# """

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as signal
from qm import LoopbackInterface
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *

pulse_len = 20000  # ns
config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},
            },
            "analog_inputs": {
                1: {"offset": +0.0},
            },
        }
    },
    "elements": {
        "qe1": {
            "singleInput": {"port": ("con1", 1)},
            "outputs": {"output1": ("con1", 1)},
            "intermediate_frequency": 5.5e6,
            "operations": {
                "readoutOp": "readoutPulse",
            },
            "time_of_flight": 184,
            "smearing": 0,
        },
        "qe2": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 15e6,
            "operations": {
                "readoutOp": "readoutPulse2",
            },
        },
    },
    "pulses": {
        "readoutPulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
        },
        "readoutPulse2": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"single": "const_wf"},
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.
QM1 = QMm.open_qm(config)

with program() as filter_sp:
    data_stream = declare_stream(adc_trace=True)
    play("readoutOp", "qe2")  # Plays a 15MHz frequency waveform
    measure("readoutOp", "qe1", data_stream)  # Plays a 5.5MHz frequency waveform and readout the data

    # Creates a 5th order Butterworth LPF filter at 7.5MHz.
    # For other filters, see https://docs.scipy.org/doc/scipy/reference/signal.html or any other source
    # 'y' is the filter's impulse response.
    butter = signal.dlti(*signal.butter(5, 7.5e6, btype="low", analog=False, output="ba", fs=1e9))
    t, y = signal.dimpulse(butter, n=pulse_len)
    y = np.squeeze(y)

    # A simple example of the simplest 'filter' - which does nothing.
    # unity = np.zeros(pulse_len, int)
    # unity[0] = 1
    # y = unity

    with stream_processing():
        data_stream.input1().save_all("raw_data")
        data_stream.input1().fft().save_all("fft_raw_data")
        data_stream.input1().convolution(y).save_all("filtered_data")
        data_stream.input1().convolution(y).fft().save_all("fft_filter_data")

# Simulate the program with the feedback and the realistic 184ns latency.
job = QM1.simulate(
    filter_sp,
    SimulationConfig(
        7500,
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)], latency=184),
    ),
)

samples = job.get_simulated_samples()
res = job.result_handles

# Fetching data, converting the 12 bit ADC value to voltage and removing extra dimensions.
raw_adc = np.squeeze(res.raw_data.fetch_all()["value"] / 2**12)
filter_adc = np.squeeze(res.filtered_data.fetch_all()["value"] / 2**12)

ax1 = plt.subplot(211)
plt.plot(raw_adc, ".")
plt.title("Raw ADC Input")
plt.subplot(212, sharex=ax1)
plt.plot(filter_adc, ".")
plt.title("Filtered ADC input")
plt.axis(ymin=-0.2, ymax=0.2)
plt.xlabel("t [ns]")
plt.axis(xmin=0, xmax=1.1 * pulse_len)
plt.tight_layout()
plt.show()

# Fetching data, converting the 12 bit ADC value to voltage and removing extra dimensions.
fft_raw_adc = np.squeeze(res.fft_raw_data.fetch_all()["value"] / 2**12)
fft_filter_adc = np.squeeze(res.fft_filter_data.fetch_all()["value"] / 2**12)

# Sqrt of Sum of Squares (gives absolute value).
# Normalize by the length of the original sine wave such that we will get the result in real units.
fft_raw_adc = np.sqrt(np.sum(np.squeeze(fft_raw_adc) ** 2, axis=1)) / pulse_len
fft_filter_adc = np.sqrt(np.sum(np.squeeze(fft_filter_adc) ** 2, axis=1)) / pulse_len
plt.figure()
ax2 = plt.subplot(211)
# When doing fft, the frequency goes from -1/(2*dt) to 1/(2*dt) in jumps of  1/(maxT).
# In our case, dt = 1ns and maxT = pulse_len.
# In addition, the resulting fft data is shifted such that the order is [0:1/(2*dt), -1/(2*dt):0].
# Here, instead of shifting the data, we just look only at the positive part.
plt.plot(np.arange(0, 0.5, 1 / pulse_len), fft_raw_adc[: int(np.ceil(len(fft_raw_adc) / 2))])
plt.title("FFT on Raw Data")
plt.subplot(212, sharex=ax2)
# Here, maxT = 2*pulse_len because of the convolution
plt.plot(
    np.arange(0, 0.5, 1 / (2 * pulse_len)),
    fft_filter_adc[: int(np.ceil(len(fft_filter_adc) / 2))],
)
plt.title("FFT on Filtered Data")
plt.xlabel("f [GHz]")
plt.axis(xmin=0, xmax=0.02)
plt.tight_layout()
plt.show()
