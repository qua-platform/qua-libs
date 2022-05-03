import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import gaussian
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig

##############
# parameters #
##############

gauss_len = 32
gauss_sig = 7
gauss_amp = 0.2
swap_len = 52
readout_len = 400
readout_amp = 0.28

LO_transmon = 4.4e9
transmon_IF = 79e6
LO_cubic = 3.6e9
cubic_IF = 33e6
LO_hybrid = 0.8e9
ge_eg_IF = 46e6
ee_gf_IF = -118e6
rr_LO = 6.6e9
rr_transmon_IF = -91e6
rr_cubic_IF = 176e6


config = {
    "version": 1,
    "controllers": {
        "con1": {
            "type": "opx1",
            "analog_outputs": {
                1: {"offset": +0.0},  # transmon I
                2: {"offset": +0.0},  # transmon Q
                3: {"offset": +0.0},  # cubic ge I
                4: {"offset": +0.0},  # cubic ge Q
                5: {"offset": +0.0},  # I hybrid
                6: {"offset": +0.0},  # Q hybrid
                7: {"offset": +0.0},  # Readout resonator
                8: {"offset": +0.0},  # Readout resonator
            },
            "digital_outputs": {},
            "analog_inputs": {1: {"offset": +0.0}, 2: {"offset": +0.0}},
        }
    },
    "elements": {
        "transmon_ge": {
            "mixInputs": {
                "I": ("con1", 1),
                "Q": ("con1", 2),
                "lo_frequency": LO_transmon,
                "mixer": "mixer_transmon_ge",
            },
            "intermediate_frequency": transmon_IF,
            "operations": {
                "pi": "pi_pulse",
                "pi2": "pi2_pulse",
            },
        },
        "cubic_ge": {
            "mixInputs": {
                "I": ("con1", 3),
                "Q": ("con1", 4),
                "lo_frequency": LO_cubic,
                "mixer": "mixer_cubic_ge",
            },
            "intermediate_frequency": cubic_IF,
            "operations": {
                "pi": "pi_pulse",
                "pi2": "pi2_pulse",
            },
        },
        "ge_eg": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": LO_hybrid,
                "mixer": "mixer_hybrid",
            },
            "intermediate_frequency": ge_eg_IF,
            "operations": {"pi": "pi_pulse", "swap": "swap_ge_eg_pulse"},
        },
        "ee_gf": {
            "mixInputs": {
                "I": ("con1", 5),
                "Q": ("con1", 6),
                "lo_frequency": LO_hybrid,
                "mixer": "mixer_hybrid",
            },
            "intermediate_frequency": ee_gf_IF,
            "operations": {"pi": "pi_pulse", "swap": "swap_ee_gf_pulse"},
        },
        "rr_transmon": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": rr_LO,
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": rr_transmon_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": 28,
            "smearing": 0,
        },
        "rr_cubic": {
            "mixInputs": {
                "I": ("con1", 7),
                "Q": ("con1", 8),
                "lo_frequency": rr_LO,
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": rr_cubic_IF,
            "operations": {
                "readout": "readout_pulse",
            },
            "outputs": {
                "out1": ("con1", 1),
            },
            "time_of_flight": 28,
            "smearing": 0,
        },
    },
    "pulses": {
        "pi_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi_wf", "Q": "zero_wf"},
        },
        "pi2_pulse": {
            "operation": "control",
            "length": gauss_len,
            "waveforms": {"I": "pi2_wf", "Q": "zero_wf"},
        },
        "swap_ge_eg_pulse": {
            "operation": "control",
            "length": swap_len,
            "waveforms": {"I": "swap_ge_eg_wf", "Q": "zero_wf"},
        },
        "swap_ee_gf_pulse": {
            "operation": "control",
            "length": swap_len,
            "waveforms": {"I": "swap_ee_gf_wf", "Q": "zero_wf"},
        },
        "readout_pulse": {
            "operation": "measurement",
            "length": readout_len,
            "waveforms": {"I": "readout_wf", "Q": "zero_wf"},
            "integration_weights": {
                "integW_cos": "integW_cos",
                "integW_sin": "integW_sin",
            },
            "digital_marker": "ON",
        },
    },
    "waveforms": {
        "zero_wf": {"type": "constant", "sample": 0.0},
        "pi_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in gauss_amp * gaussian(gauss_len, gauss_sig)],
        },
        "pi2_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in 0.5 * gauss_amp * gaussian(gauss_len, gauss_sig)],
        },
        "swap_ge_eg_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in np.linspace(0, 0.1, 6)]
            + [0.1] * 40
            + [float(arg) for arg in np.linspace(0.1, 0.0, 6)],
        },
        "swap_ee_gf_wf": {
            "type": "arbitrary",
            "samples": [float(arg) for arg in np.linspace(0.0, 0.1, 6)]
            + [0.1] * 14
            + [float(arg) for arg in np.linspace(0.1, 0.0, 6)]
            + [float(arg) for arg in np.linspace(0.0, 0.1, 6)]
            + [0.1] * 14
            + [float(arg) for arg in np.linspace(0.1, 0.0, 6)],
        },
        "readout_wf": {"type": "constant", "sample": readout_amp},
    },
    "digital_waveforms": {
        "ON": {"samples": [(1, 0)]},
    },
    "integration_weights": {
        "integW_cos": {
            "cosine": [1.0] * int(readout_len / 4),
            "sine": [0.0] * int(readout_len / 4),
        },
        "integW_sin": {
            "cosine": [0.0] * int(readout_len / 4),
            "sine": [1.0] * int(readout_len / 4),
        },
    },
    "mixers": {
        "mixer_transmon_ge": [
            {
                "intermediate_frequency": transmon_IF,
                "lo_frequency": LO_transmon,
                "correction": [1, 0, 0, 1],
            }
        ],
        "mixer_cubic_ge": [
            {
                "intermediate_frequency": cubic_IF,
                "lo_frequency": LO_cubic,
                "correction": [1, 0, 0, 1],
            }
        ],
        "mixer_hybrid": [
            {
                "intermediate_frequency": ge_eg_IF,
                "lo_frequency": LO_hybrid,
                "correction": [1, 0, 0, 1],
            },
            {
                "intermediate_frequency": ee_gf_IF,
                "lo_frequency": LO_hybrid,
                "correction": [1, 0, 0, 1],
            },
        ],
        "mixer_rr": [
            {
                "intermediate_frequency": rr_transmon_IF,
                "lo_frequency": rr_LO,
                "correction": [1, 0, 0, 1],
            },
            {
                "intermediate_frequency": rr_cubic_IF,
                "lo_frequency": rr_LO,
                "correction": [1, 0, 0, 1],
            },
        ],
    },
}


with program() as swap:

    n = declare(int)
    phi = declare(fixed)

    I = declare(fixed)
    I_stream = declare_stream()

    Q = declare(fixed)
    Q_stream = declare_stream()

    with for_(n, 0, n < 10000, n + 1):

        with for_(phi, -np.pi, phi < np.pi, phi + np.pi / 50):

            play("pi", "cubic_ge")
            align("cubic_ge", "transmon_ge")
            play("pi2", "transmon_ge")

            align("transmon_ge", "ge_eg", "ee_gf")
            play("swap", "ge_eg")
            play("swap", "ee_gf")

            align("cubic_ge", "ge_eg", "ee_gf")
            frame_rotation(phi, "cubic_ge")
            play("pi2", "cubic_ge")
            align("cubic_ge", "rr_cubic")
            measure(
                "readout",
                "rr_cubic",
                None,
                demod.full("integW_cos", I),
                demod.full("integW_sin", Q),
            )
            save(I, I_stream)
            save(Q, Q_stream)

qmm = QuantumMachinesManager()
sim_len = 700
job = qmm.simulate(config, swap, SimulationConfig(sim_len))

samples = job.get_simulated_samples().con1.analog
transmon_I = samples["1"]
transmon_Q = samples["2"]
cubic_ge_I = samples["3"]
cubic_ge_Q = samples["4"]
hybrid_I = samples["5"]
hybrid_Q = samples["6"]
Readout_resonator_I = samples["7"]
Readout_resonator_Q = samples["8"]
t = np.arange(sim_len * 4)

fig, axs = plt.subplots(3, 1)
axs[0].plot(t, transmon_I, label="transmon_I")
axs[0].plot(t, transmon_Q, label="transmon_Q")
axs[1].plot(t, cubic_ge_I, label="cubic_ge_I")
axs[1].plot(t, cubic_ge_Q, label="cubic_ge_Q")
axs[1].plot(t, hybrid_I, label="hybrid_I")
axs[1].plot(t, hybrid_Q, label="hybrid_Q")
axs[2].plot(t, Readout_resonator_Q, label="Readout_resonator_Q")
axs[2].plot(t, Readout_resonator_Q, label="Readout_resonator_Q")
axs[0].legend()
axs[1].legend()
axs[2].legend()
