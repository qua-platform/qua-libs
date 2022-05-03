# """
# filters_with_windows.py: Demonstrate applying filters directly on the input using windows
# Author: Yoav Romach - Quantum Machines
# Created: 31/12/2020
# Created on QUA version: 0.7.411
# """

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import LoopbackInterface
from qm import SimulationConfig
import matplotlib.pyplot as plt
import numpy as np

chunk_size = 50  # This also decimates the data & reduces the sampling from 1GHz to 1GHz/(4*chunk_size)
pulse_len = 20000  # Needs to be dividable by (4 * chunk_size)
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
            "intermediate_frequency": 20e6,
            "operations": {
                "readoutOp": "readoutPulse",
            },
            "time_of_flight": 184,
            "smearing": 0,
        },
        "qe2": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 20.1e6,
            "operations": {
                "readoutOp": "readoutPulse",
            },
        },
        "qe3": {
            "singleInput": {"port": ("con1", 1)},
            "intermediate_frequency": 0.1e6,
            "operations": {
                "readoutOp": "readoutPulse",
            },
        },
    },
    "pulses": {
        "readoutPulse": {
            "operation": "measure",
            "length": pulse_len,
            "waveforms": {"single": "const_wf"},
            "integration_weights": {
                "sine_weight": "sine",
                "cosine_weight": "cosine",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "const_wf": {"type": "constant", "sample": 0.2},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "cosine": {
            "cosine": [1.0] * int(np.ceil(pulse_len / 4)),
            "sine": [0.0] * int(np.ceil(pulse_len / 4)),
        },
        "sine": {
            "cosine": [0.0] * int(np.ceil(pulse_len / 4)),
            "sine": [1.0] * int(np.ceil(pulse_len / 4)),
        },
    },
}

# Open communication with the server.
QMm = QuantumMachinesManager()

# Create a quantum machine based on the configuration.
QM1 = QMm.open_qm(config)

# This program plays a 20 MHz & a 100 kHz pulse and then uses an LPF based on a moving window to remove the 20 MHz.
with program() as DC_filter:
    i = declare(int)
    data_stream = declare_stream(adc_trace=True)
    data = declare(fixed, size=int(np.ceil(pulse_len / (4 * chunk_size))))
    play("readoutOp", "qe3")  # Plays a 100 kHz RF
    # Plays a 20 MHz RF and measure the data, both directly into the steam and using a moving window integration.
    measure(
        "readoutOp",
        "qe1",
        data_stream,
        integration.moving_window("cosine_weight", data, chunk_size, 1),
    )

    # Saves the filtered data in "data"
    with for_(i, 0, i < data.length(), i + 1):
        save(data[i], "data")

    with stream_processing():
        data_stream.input1().save_all("raw_data")

# This program plays a 20 MHz & a 20.1 MHz pulse, performs a demod by multiplying the result with a sin wave, and uses
# an LPF filter to keep the 100 kHz envelope.
with program() as IF_filter:
    i = declare(int)
    data_stream = declare_stream(adc_trace=True)
    data = declare(fixed, size=int(np.ceil(pulse_len / (4 * chunk_size))))
    play("readoutOp", "qe2")  # Plays a 20.1 MHz RF
    # Plays a 20 MHz RF and measure the data, both directly into the steam and using a moving window demod integration.
    measure(
        "readoutOp",
        "qe1",
        data_stream,
        demod.moving_window("sine_weight", data, chunk_size, 1),
    )

    # Saves the filtered data in "data"
    with for_(i, 0, i < data.length(), i + 1):
        save(data[i], "data")

    with stream_processing():
        data_stream.input1().save_all("raw_data")

job = QM1.simulate(
    DC_filter,
    SimulationConfig(
        7500,
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)], latency=184),
    ),
)

samples = job.get_simulated_samples()
res = job.result_handles

# Fetching data, converting the 12 bit ADC value to voltage and removing the extra dimension.
raw_adc = np.squeeze(res.raw_data.fetch_all()["value"] / 2**12)
filter_adc = res.data.fetch_all()["value"] * (2**12)  # For 'demod' the correction is the other way around

plt.figure()
ax1 = plt.subplot(211)
plt.plot(raw_adc)
plt.axis(ymin=-0.4, ymax=0.4)
plt.title("Raw ADC Input - 20 MHz & 100 kHz")
plt.subplot(212, sharex=ax1)
plt.plot(np.arange(0, pulse_len, 4 * chunk_size), filter_adc / (4 * chunk_size), ".")  # Data is scaled by 4*chunk_size
plt.title("Filtered ADC input")
plt.xlabel("t [ns]")
plt.tight_layout()
plt.show()

job = QM1.simulate(
    IF_filter,
    SimulationConfig(
        7500,
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)], latency=184),
    ),
)

samples = job.get_simulated_samples()
res = job.result_handles

# Fetching data, converting the 12 bit ADC value to voltage and removing the extra dimension.
raw_adc = np.squeeze(res.raw_data.fetch_all()["value"] / 2**12)
filter_adc = res.data.fetch_all()["value"] * (2**12)  # For 'demod' the correction is the other way around

plt.figure()
ax1 = plt.subplot(211)
plt.plot(raw_adc)
plt.axis(ymin=-0.4, ymax=0.4)
plt.title("Raw ADC Input - 20 MHz & 20.1 MHz")
plt.subplot(212, sharex=ax1)
plt.plot(np.arange(0, pulse_len, 4 * chunk_size), filter_adc / (2 * chunk_size), ".")  # Data is scaled by 2*chunk_size
plt.title("Filtered ADC input")
plt.xlabel("t [ns]")
plt.tight_layout()
plt.show()
