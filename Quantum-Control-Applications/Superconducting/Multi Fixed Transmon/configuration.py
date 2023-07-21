from quam import QuAM
import numpy as np
from qualang_tools.config.waveform_tools import drag_gaussian_pulse_waveforms
from qualang_tools.units import unit
from set_octave import OctaveUnit, octave_declaration

############################
# Set octave configuration #
############################
# Custom port mapping example
# port_mapping_1 = [
#     {
#         ("con1", 1): ("octave1", "I1"),
#         ("con1", 2): ("octave1", "Q1"),
#         ("con1", 3): ("octave1", "I2"),
#         ("con1", 4): ("octave1", "Q2"),
#         ("con1", 5): ("octave1", "I3"),
#         ("con1", 6): ("octave1", "Q3"),
#         ("con1", 7): ("octave1", "I4"),
#         ("con1", 8): ("octave1", "Q4"),
#         ("con1", 9): ("octave1", "I5"),
#         ("con1", 10): ("octave1", "Q5"),
#     }
# ]
# port_mapping_2 = [
#     {
#         ("con2", 1): ("octave2", "I1"),
#         ("con2", 2): ("octave2", "Q1"),
#         ("con2", 3): ("octave2", "I2"),
#         ("con2", 4): ("octave2", "Q2"),
#         ("con2", 5): ("octave2", "I3"),
#         ("con2", 6): ("octave2", "Q3"),
#         ("con2", 7): ("octave2", "I4"),
#         ("con2", 8): ("octave2", "Q4"),
#         ("con2", 9): ("octave2", "I5"),
#         ("con2", 10): ("octave2", "Q5"),
#     }
# ]

machine = QuAM('quam_state.json', flat_data=False)
octave_1 = OctaveUnit("octave1", machine.network.octave1_ip, port=machine.network.qop_port, con="con1", clock="External_1000MHz")
octave_2 = OctaveUnit("octave2", machine.network.octave2_ip, port=machine.network.qop_port, con="con2", clock="External_1000MHz")

# Add the octaves
# octaves = [octave_1]
octaves = [octave_1, octave_2]
# Configure the Octaves
octave_config = octave_declaration(octaves)

#####################
# OPX configuration #
#####################

u = unit(coerce_to_integer=True)

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
                quam.qubits[i].pi_amp,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        x180_I_wf.append(x180_wf)
        x180_Q_wf.append(x180_der_wf)
        # x90
        x90_wf, x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].pi_amp / 2,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        x90_I_wf.append(x90_wf)
        x90_Q_wf.append(x90_der_wf)
        # -x90
        minus_x90_wf, minus_x90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].pi_amp / 2,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        minus_x90_I_wf.append(minus_x90_wf)
        minus_x90_Q_wf.append(minus_x90_der_wf)
        # y180
        y180_wf, y180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].pi_amp,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        y180_I_wf.append((-1) * y180_der_wf)
        y180_Q_wf.append(y180_wf)
        # y90
        y90_wf, y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                quam.qubits[i].pi_amp / 2,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        y90_I_wf.append((-1) * y90_der_wf)
        y90_Q_wf.append(y90_wf)
        # -y90
        minus_y90_wf, minus_y90_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                -quam.qubits[i].pi_amp / 2,
                quam.qubits[i].pi_length,
                quam.qubits[i].pi_length / 5,
                quam.qubits[i].drag_coefficient,
                quam.qubits[i].anharmonicity,
                quam.qubits[i].ac_stark_detuning,
            )
        )
        minus_y90_I_wf.append((-1) * minus_y90_der_wf)
        minus_y90_Q_wf.append(minus_y90_wf)

    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                    1: {"offset": 0.0},
                    2: {"offset": 0.0},
                    3: {"offset": 0.0},
                    4: {"offset": 0.0},
                    5: {"offset": 0.0},
                    6: {"offset": 0.0},
                    7: {"offset": 0.0},
                    8: {"offset": 0.0}, 
                    9: {"offset": 0.0}, 
                    10: {"offset": 0.0}, 
                },
                "digital_outputs": {
                    1: {},
                    2: {},
                    3: {},
                    4: {},
                    5: {},
                    6: {},
                    7: {},
                    8: {},
                    9: {},
                    10: {},
                },
                "analog_inputs": {
                    1: {
                        "offset": quam.global_parameters.con1_downconversion_offset_I,
                        "gain_db": quam.global_parameters.con1_downconversion_gain,
                    },  
                    2: {
                        "offset": quam.global_parameters.con1_downconversion_offset_Q,
                        "gain_db": quam.global_parameters.con1_downconversion_gain,
                    },  
                },
            },
            "con2": {
                "analog_outputs": {
                    1: {"offset": 0.0},
                    2: {"offset": 0.0},
                    3: {"offset": 0.0},
                    4: {"offset": 0.0},
                    5: {"offset": 0.0},
                    6: {"offset": 0.0},
                    7: {"offset": 0.0},
                    8: {"offset": 0.0}, 
                    9: {"offset": 0.0}, 
                    10: {"offset": 0.0}, 
                },
                "digital_outputs": {
                    1: {},
                    2: {},
                    3: {},
                    4: {},
                    5: {},
                    6: {},
                    7: {},
                    8: {},
                    9: {},
                    10: {},
                },
                "analog_inputs": {
                    1: {
                        "offset": quam.global_parameters.con2_downconversion_offset_I,
                        "gain_db": quam.global_parameters.con2_downconversion_gain,
                    },  
                    2: {
                        "offset": quam.global_parameters.con2_downconversion_offset_Q,
                        "gain_db": quam.global_parameters.con2_downconversion_gain,
                    },  
                },
            },
        },
        "elements": {
            **{
                quam.resonators[i].name: {
                    "mixInputs": {
                        "I": ("con1", quam.resonators[i].wiring.I),
                        "Q": ("con1", quam.resonators[i].wiring.Q),
                        "lo_frequency": quam.resonators[i].lo,
                        "mixer": "octave_octave1_1",
                    },
                    "intermediate_frequency": (quam.resonators[i].f_readout - quam.resonators[i].lo),
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
                quam.qubits[i].name: {
                    "mixInputs": {
                        "I": (quam.qubits[i].wiring.controller, quam.qubits[i].wiring.I),
                        "Q": (quam.qubits[i].wiring.controller, quam.qubits[i].wiring.Q),
                        "lo_frequency": quam.qubits[i].lo,
                        "mixer": quam.qubits[i].mixer_name,
                    },
                    "intermediate_frequency": (quam.qubits[i].f_01 - quam.qubits[i].lo),
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
                        "opt_cos": f"opt_cosine_weights{i}",
                        "opt_sin": f"opt_sine_weights{i}",
                        "opt_minus_sin": f"opt_minus_sine_weights{i}",
                    },
                    "digital_marker": "ON",
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"x90_pulse{i}": {
                    "operation": "control",
                    "length": quam.qubits[i].pi_length,
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
                    "length": quam.qubits[i].pi_length,
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
                    "length": quam.qubits[i].pi_length,
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
                    "length": quam.qubits[i].pi_length,
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
                    "length": quam.qubits[i].pi_length,
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
                    "length": quam.qubits[i].pi_length,
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
            **{
                f"opt_cosine_weights{i}": {
                    "cosine": [(1.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"opt_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(1.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
            **{
                f"opt_minus_sine_weights{i}": {
                    "cosine": [(0.0, quam.resonators[i].optimal_pulse_length)],
                    "sine": [(-1.0, quam.resonators[i].optimal_pulse_length)],
                }
                for i in range(len(quam.resonators))
            },
        },
        "mixers": {
            **{
                quam.qubits[i].mixer_name: [
                    {
                        "intermediate_frequency": (quam.qubits[i].f_01 - quam.qubits[i].lo),
                        "lo_frequency": quam.qubits[i].lo,
                        "correction": [1.0, 0.0, 0.0, 1.0],
                    },
                ]
                for i in range(len(quam.qubits))
            },
            "octave_octave1_1": [
                {
                    "intermediate_frequency": (quam.resonators[i].f_readout - quam.resonators[i].lo),
                    "lo_frequency": quam.resonators[i].lo,
                    "correction": [1.0, 0.0, 0.0, 1.0],
                } 
                for i in range(len(quam.resonators))
            ]
        },
    }
    return config
