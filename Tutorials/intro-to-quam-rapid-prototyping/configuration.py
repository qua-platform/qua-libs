from quam import QuAM
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit

#######################
# AUXILIARY FUNCTIONS #
#######################
u = unit(coerce_to_integer=True)


# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports. (unit-less), set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians), set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]


def build_config(quam: QuAM):
    x180_I_wf = []
    x180_Q_wf = []
    x90_I_wf = []
    x90_Q_wf = []
    minus_x90_I_wf = []
    minus_x90_Q_wf = []
    y180_I_wf = []
    y180_Q_wf = []
    y90_I_wf = []
    y90_Q_wf = []
    minus_y90_I_wf = []
    minus_y90_Q_wf = []
    # No DRAG when alpha=0, it's just a gaussian.
    for i in range(len(quam.qubits)):
        # x180
        x180_wf, x180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].xy.pi_amp,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        x180_I_wf.append(x180_wf)
        x180_Q_wf.append(x180_der_wf)
        # x90
        x90_wf, x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].xy.pi_amp / 2,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        x90_I_wf.append(x90_wf)
        x90_Q_wf.append(x90_der_wf)
        # -x90
        minus_x90_wf, minus_x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].xy.pi_amp / 2,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        minus_x90_I_wf.append(minus_x90_wf)
        minus_x90_Q_wf.append(minus_x90_der_wf)
        # y180
        y180_wf, y180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].xy.pi_amp,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        y180_I_wf.append((-1) * y180_der_wf)
        y180_Q_wf.append(y180_wf)
        # y90
        y90_wf, y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].xy.pi_amp / 2,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        y90_I_wf.append((-1) * y90_der_wf)
        y90_Q_wf.append(y90_wf)
        # -y90
        minus_y90_wf, minus_y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].xy.pi_amp / 2,
                quam.qubits[i].xy.pi_length,
                quam.qubits[i].xy.pi_length / 5,
                quam.qubits[i].xy.drag_coefficient,
                quam.qubits[i].xy.anharmonicity,
                quam.qubits[i].xy.ac_stark_detuning,
            )
        )
        minus_y90_I_wf.append((-1) * minus_y90_der_wf)
        minus_y90_Q_wf.append(minus_y90_wf)

    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": quam.qubits[0].xy.wiring.mixer_correction.offset_I},  # I qubit1 XY
                    2: {"offset": quam.qubits[0].xy.wiring.mixer_correction.offset_Q},  # Q qubit1 XY
                    3: {"offset": quam.qubits[1].xy.wiring.mixer_correction.offset_I},  # I qubit2 XY
                    4: {"offset": quam.qubits[1].xy.wiring.mixer_correction.offset_Q},  # Q qubit2 XY
                    5: {"offset": quam.resonators[0].wiring.mixer_correction.offset_I},  # I readout line
                    6: {"offset": quam.resonators[0].wiring.mixer_correction.offset_Q},  # Q readout line
                    7: {
                        "offset": quam.qubits[0].z.max_frequency_point,
                        "filter": {
                            "feedforward": quam.qubits[0].z.wiring.filter.fir_taps,
                            "feedback": quam.qubits[0].z.wiring.filter.iir_taps,
                        },
                    },  # qubit1 Z
                    8: {
                        "offset": quam.qubits[1].z.max_frequency_point,
                        "filter": {
                            "feedforward": quam.qubits[1].z.wiring.filter.fir_taps,
                            "feedback": quam.qubits[1].z.wiring.filter.iir_taps,
                        },
                    },  # qubit2 Z
                },
                "digital_outputs": {
                    1: {},
                },
                "analog_inputs": {
                    1: {
                        "offset": quam.global_parameters.downconversion_offset_I,
                        "gain_db": 0,
                    },  # I from down-conversion
                    2: {
                        "offset": quam.global_parameters.downconversion_offset_Q,
                        "gain_db": 0,
                    },  # Q from down-conversion
                },
            },
        },
        "elements": {
            **{
                f"rr{i}": {
                    "mixInputs": {
                        "I": ("con1", quam.resonators[i].wiring.I),
                        "Q": ("con1", quam.resonators[i].wiring.Q),
                        "lo_frequency": quam.local_oscillators.readout[0].freq,
                        "mixer": f"mixer_rr{i}",
                    },
                    "intermediate_frequency": (quam.resonators[i].f_opt - quam.local_oscillators.readout[0].freq),
                    "operations": {
                        "cw": "const_pulse",
                        "readout": f"readout_pulse_q{i}",
                    },
                    "outputs": {
                        "out1": ("con1", 1),
                        "out2": ("con1", 2),
                    },
                    "time_of_flight": quam.global_parameters.time_of_flight,
                    "smearing": 0,
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"q{i}_xy": {
                    "mixInputs": {
                        "I": ("con1", quam.qubits[i].xy.wiring.I),
                        "Q": ("con1", quam.qubits[i].xy.wiring.Q),
                        "lo_frequency": quam.local_oscillators.qubits[0].freq,
                        "mixer": f"mixer_q{i}_xy",
                    },
                    "intermediate_frequency": (quam.qubits[i].xy.f_01 - quam.local_oscillators.qubits[0].freq),
                    "operations": {
                        "cw": "const_pulse",
                        "x180": f"x180_pulse{i}",
                        "x90": f"x90_pulse{i}",
                        "-x90": f"-x90_pulse{i}",
                        "y90": f"y90_pulse{i}",
                        "y180": f"y180_pulse{i}",
                        "-y90": f"-y90_pulse{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"q{i}_z": {
                    "singleInput": {
                        "port": ("con1", quam.qubits[i].z.wiring.port),
                    },
                    "operations": {
                        "const": f"const_flux_pulse{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
        },
        "pulses": {
            "const_pulse": {
                "operation": "control",
                "length": 1000,
                "waveforms": {
                    "I": "const_wf",
                    "Q": "zero_wf",
                },
            },
            **{
                f"const_flux_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].z.flux_pulse_length,
                    "waveforms": {
                        "single": f"const_flux{i}_wf",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"readout_pulse_q{i}": {
                    "operation": "measurement",
                    "length": quam.resonators[i].readout_pulse_length,
                    "waveforms": {
                        "I": f"readout{i}_wf",
                        "Q": "zero_wf",
                    },
                    "integration_weights": {
                        "cos": f"cosine_weights{i}",
                        "sin": f"sine_weights{i}",
                        "minus_sin": f"minus_sine_weights{i}",
                        "rotated_cos": f"rotated_cosine_weights{i}",
                        "rotated_sin": f"rotated_sine_weights{i}",
                        "rotated_minus_sin": f"rotated_minus_sine_weights{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"x90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"x90_I_wf{i}",
                        "Q": f"x90_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"x180_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"x180_I_wf{i}",
                        "Q": f"x180_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"-x90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"minus_x90_I_wf{i}",
                        "Q": f"minus_x90_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"y90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"y90_I_wf{i}",
                        "Q": f"y90_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"y180_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"y180_I_wf{i}",
                        "Q": f"y180_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
            **{
                f"-y90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].xy.pi_length,
                    "waveforms": {
                        "I": f"minus_y90_I_wf{i}",
                        "Q": f"minus_y90_Q_wf{i}",
                    },
                }
                for i in range(len(quam.qubits))
            },
        },
        "waveforms": {
            "zero_wf": {"type": "constant", "sample": 0.0},
            "const_wf": {"type": "constant", "sample": 0.1},
            **{
                f"readout{i}_wf": {"type": "constant", "sample": quam.resonators[i].readout_pulse_amp}
                for i in range(len(quam.resonators))
            },
            **{
                f"const_flux{i}_wf": {"type": "constant", "sample": quam.qubits[i].z.flux_pulse_amp}
                for i in range(len(quam.resonators))
            },
            **{f"x90_I_wf{i}": {"type": "arbitrary", "samples": x90_I_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{f"x90_Q_wf{i}": {"type": "arbitrary", "samples": x90_Q_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{
                f"x180_I_wf{i}": {"type": "arbitrary", "samples": x180_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"x180_Q_wf{i}": {"type": "arbitrary", "samples": x180_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_x90_I_wf{i}": {"type": "arbitrary", "samples": minus_x90_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_x90_Q_wf{i}": {"type": "arbitrary", "samples": minus_x90_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{f"y90_I_wf{i}": {"type": "arbitrary", "samples": y90_I_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{f"y90_Q_wf{i}": {"type": "arbitrary", "samples": y90_Q_wf[i].tolist()} for i in range(len(quam.qubits))},
            **{
                f"y180_I_wf{i}": {"type": "arbitrary", "samples": y180_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"y180_Q_wf{i}": {"type": "arbitrary", "samples": y180_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_y90_I_wf{i}": {"type": "arbitrary", "samples": minus_y90_I_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
            **{
                f"minus_y90_Q_wf{i}": {"type": "arbitrary", "samples": minus_y90_Q_wf[i].tolist()}
                for i in range(len(quam.qubits))
            },
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
        },
        "integration_weights": {
            **{
                f"cosine_weights{i}": {
                    "cosine": [(1.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(0.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(1.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"minus_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].readout_pulse_length)],
                    "sine": [(-1.0, quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_cosine_weights{i}": {
                    "cosine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_sine_weights{i}": {
                    "cosine": [(np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"rotated_minus_sine_weights{i}": {
                    "cosine": [(-np.sin(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                    "sine": [(-np.cos(quam.resonators[i].rotation_angle), quam.resonators[i].readout_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
        },
        "mixers": {
            **{
                f"mixer_q{i}_xy": [
                    {
                        "intermediate_frequency": (quam.qubits[i].xy.f_01 - quam.local_oscillators.qubits[0].freq),
                        "lo_frequency": quam.local_oscillators.qubits[0].freq,
                        "correction": IQ_imbalance(
                            quam.qubits[i].xy.wiring.mixer_correction.gain,
                            quam.qubits[i].xy.wiring.mixer_correction.phase,
                        ),
                    },
                ]
                for i in range(len(quam.qubits))
            },
            **{
                f"mixer_rr{i}": [
                    {
                        "intermediate_frequency": (quam.resonators[i].f_opt - quam.local_oscillators.readout[0].freq),
                        "lo_frequency": quam.local_oscillators.readout[0].freq,
                        "correction": IQ_imbalance(
                            quam.resonators[i].wiring.mixer_correction.gain,
                            quam.resonators[i].wiring.mixer_correction.phase,
                        ),
                    },
                ]
                for i in range(len(quam.resonators))
            },
        },
    }
    return config
