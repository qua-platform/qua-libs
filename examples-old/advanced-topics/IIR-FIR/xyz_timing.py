import scipy.signal as sig
import numpy as np
import matplotlib.pyplot as plt
from qm.qua import *
from qm.QuantumMachinesManager import (
    SimulationConfig,
    QuantumMachinesManager,
)

qmm = QuantumMachinesManager()
pulse_len = 60

ntaps = 40  # max is 40
delays = np.arange(20, 22, 0.25)

for delay in delays:
    with program() as xyz_timing:
        n = declare(int)
        I = declare(fixed)
        I_st = declare_stream()
        Q = declare(fixed)
        Q_st = declare_stream()
        with for_(n, 0, n < 1, n + 1):
            wait(5, "q_z")  # 20ns offest
            play("const", "q_z")
            play("X", "q_xy")
            align()
            measure(
                "readout",
                "rr",
                None,
                dual_demod.full("integW_cos", "out1", "integW_sin", "out2", I),
                dual_demod.full("integW_minus_sin", "out1", "integW_cos", "out2", Q),
            )
            save(I, I_st)
            save(Q, Q_st)
        with stream_processing():
            I_st.average().save("I")
            Q_st.average().save("Q")

    feedforward_filter = np.sinc(np.linspace(0, ntaps, ntaps + 1)[0:-1] - delay)

    print("feedforward taps:", feedforward_filter)
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "type": "opx1",
                "analog_outputs": {
                    1: {
                        "offset": +0.0,
                        "filter": {"feedforward": feedforward_filter},
                    },  # q_xy
                    2: {
                        "offset": +0.0,
                        "filter": {"feedforward": feedforward_filter},
                    },  # q_xy
                    3: {"offset": +0.0},  # rr
                    4: {"offset": +0.0},  # rr
                    5: {"offset": +0.0},  # flux
                },
                "analog_inputs": {
                    1: {"offset": 0.0},  # RL14_I
                    2: {"offset": 0.0},  # RL14_Q
                },
            },
        },
        "elements": {
            "q_z": {
                "singleInput": {"port": ("con1", 5)},
                "intermediate_frequency": 10,
                "operations": {
                    "const": "const_pulse",
                },
            },
            "rr": {
                "mixInputs": {
                    "I": ("con1", 3),
                    "Q": ("con1", 4),
                    "lo_frequency": 1e9,
                    "mixer": "mixer",
                },
                "intermediate_frequency": 50e6,
                "outputs": {
                    "out1": ("con1", 1),
                    "out2": ("con1", 2),
                },
                "time_of_flight": 32,
                "smearing": 0,
                "operations": {
                    "readout": "readout_pulse",
                },
            },
            "q_xy": {
                "mixInputs": {
                    "I": ("con1", 1),
                    "Q": ("con1", 2),
                    "lo_frequency": 1e9,
                    "mixer": "mixer",
                },
                "intermediate_frequency": 50e6,
                "outputs": {
                    "out1": ("con1", 1),
                    "out2": ("con1", 2),
                },
                "time_of_flight": 32,
                "smearing": 0,
                "operations": {
                    "X": "gaussian_pulse",
                },
            },
        },
        "pulses": {
            "gaussian_pulse": {
                "operation": "control",
                "length": pulse_len,
                "waveforms": {"I": "gaussian_wf", "Q": "zero_wf"},
            },
            "const_pulse": {
                "operation": "control",
                "length": 32,
                "waveforms": {"single": "const_wf"},
            },
            "readout_pulse": {
                "operation": "measurement",
                "length": pulse_len,
                "waveforms": {"I": "gaussian_wf", "Q": "zero_wf"},
                "integration_weights": {
                    "integW_cos": "integW_cos",
                    "integW_sin": "integW_sin",
                    "integW_minus_sin": "integW_minus_sin",
                },
                "digital_marker": "ON",
            },
        },
        "waveforms": {
            "gaussian_wf": {
                "type": "arbitrary",
                "samples": 0.25 * sig.gaussian(pulse_len, 5),
            },
            "const_wf": {
                "type": "constant",
                "sample": 0.4,
            },
            "zero_wf": {
                "type": "constant",
                "sample": 0,
            },
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
        },
        "integration_weights": {
            "integW_cos": {
                "cosine": [1.0] * (pulse_len // 4),
                "sine": [0.0] * (pulse_len // 4),
            },
            "integW_sin": {
                "cosine": [0.0] * (pulse_len // 4),
                "sine": [1.0] * (pulse_len // 4),
            },
            "integW_minus_sin": {
                "cosine": [0.0] * (pulse_len // 4),
                "sine": [-1.0] * (pulse_len // 4),
            },
        },
        "mixers": {
            "mixer": [
                {
                    "intermediate_frequency": 50e6,
                    "lo_frequency": 1e9,
                    "correction": [1, 0, 0, 1],
                }
            ]
        },
    }

    job = qmm.simulate(
        config,
        xyz_timing,
        SimulationConfig(duration=int(150), include_analog_waveforms=True),
    )
    job.result_handles.wait_for_all_values()
    for i in ["1", "2", "5"]:
        plt.plot(job.get_simulated_samples().con1.analog.get(i))

plt.axis([290, 350, -0.15, 0.45])
