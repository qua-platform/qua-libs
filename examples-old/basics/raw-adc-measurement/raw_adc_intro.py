"""
raw_adc_intro.py: Demonstrate recording raw analog input
Author: Gal Winer - Quantum Machines
Created: 7/11/2020
Created on QUA version: 0.5.138
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import LoopbackInterface
from qm import SimulationConfig
import matplotlib.pyplot as plt

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
            "intermediate_frequency": 1e6,
            "operations": {
                "readoutOp": "readoutPulse",
            },
            "time_of_flight": 184,
            "smearing": 0,
        },
    },
    "pulses": {
        "readoutPulse": {
            "operation": "measure",
            "length": 1000,
            "waveforms": {"single": "const_wf"},
            "digital_marker": "ON",
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

with program() as raw_adc_prog:
    measure("readoutOp", "qe1", "raw_adc")

# In the OPX, the analog signal starts 184 after the play command. In order to simulate it, we added the same latency
# here, and this is the time_of_flight in the configuration file
job = QM1.simulate(
    raw_adc_prog,
    SimulationConfig(
        500,
        simulation_interface=LoopbackInterface([("con1", 1, "con1", 1)], latency=184),
    ),
)

samples = job.get_simulated_samples()
res = job.result_handles
raw_adc = res.raw_adc_input1.fetch_all()["value"]

ax1 = plt.subplot(211)
plt.plot(samples.con1.analog["1"])
plt.title("Simulated samples")
plt.subplot(212, sharex=ax1)
plt.plot(raw_adc / 2**12)  # Converting the 12 bit ADC value to voltage
plt.title("Raw ADC input")
plt.xlabel("Sample number")
plt.tight_layout()
plt.show()
