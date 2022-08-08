"""
The process to generate the initial state and the build config function:
1. write the build_config function with references to the state
2. copy the config as a key in the state dict variable
3. replace the references in the state by placeholders
4. clear the state from not required fields
5. open a qm
6. generate the bootstrap_state_old.json
"""
import numpy as np
from typing import List, Dict
from scipy.signal.windows import gaussian
from scipy import interpolate

# IQ imbalance matrix
def IQ_imbalance(g, phi):
    """
    Creates the correction matrix for the mixer imbalance caused by the gain and phase imbalances, more information can
    be seen here:
    https://docs.qualang.io/libs/examples/mixer-calibration/#non-ideal-mixer
    :param g: relative gain imbalance between the I & Q ports (unit-less). Set to 0 for no gain imbalance.
    :param phi: relative phase imbalance between the I & Q ports (radians). Set to 0 for no phase imbalance.
    """
    c = np.cos(phi)
    s = np.sin(phi)
    N = 1 / ((1 - g**2) * (2 * c**2 - 1))
    return [float(N * x) for x in [(1 - g) * c, (1 + g) * s, (1 - g) * s, (1 + g) * c]]

# layer 1: bare state
state = {
    "OPX_config": {
        "controllers": {
            "con1": {
                "analog_outputs": {
                    "1": {"offset": 0.0},  # q1 I
                    "2": {"offset": 0.0},  # q1 Q
                    "3": {"offset": 0.0},  # q2 Q
                    "4": {"offset": 0.0},  # q2 Q
                    "5": {"offset": 0.0},  # q2 Q
                    "6": {"offset": 0.0},  # q2 Q
                    "7": {"offset": 0.0},  # q2 Q
                    "8": {"offset": 0.0},  # q2 Q
                    "9": {"offset": 0.0},  # I rrs
                    "10": {"offset": 0.0},  # Q rrs
                },
                "digital_outputs": {},
                "analog_inputs": {
                    "1": {"offset": 0.0, "gain_db": 0},
                    "2": {"offset": 0.0, "gain_db": 0},
                },
            }
        },
        "elements": {},
        "pulses": {
            "const_pulse": {
                "operation": "control",
                "length": 100,
                "waveforms": {
                    "I": "const_wf",
                    "Q": "zero_wf",
                },
            },
            "saturation_pulse": {
                "operation": "control",
                "length": 100000,
                "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
            },
        },
        "waveforms": {
            "const_wf": {"type": "constant", "sample": 0.2},
            "saturation_drive_wf": {"type": "constant", "sample": 0.2},
            "zero_wf": {"type": "constant", "sample": 0.0},
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
        },
        "integration_weights": {},
        "mixers": {},
    },
    "readout_length": 2e-6,  # Sec
    "readout_lo_freq":  6.57e9,
    "readout_resonators": [
        {
            "f_res": 6.45218e9,  # Hz
            "q_factor": None,
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_res": 6.53269e9,  # Hz
            "q_factor": None,
            "readout_length": 2e-6,  # Sec
            "readout_regime": "low_power",
            "readout_amplitude": 0.2,
            "opt_readout_frequency": 4.52503e9,
            "rotation_angle": 41.3,  # degrees
            "readout_fidelity": 0.84,
            "chi": 1e6,
            "wiring": {
                "time_of_flight": 260,
                "I": ["con1", 9],
                "Q": ["con1", 10],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
    ],
    "qubits": [
        {
            "f_01": 4.52503e9,  # Hz
            "anharmonicity": None,
            "rabi_freq": None,
            "t1": 18e-6,
            "t2": None,
            "t2*": 5e-6,
            "driving": {
                "gate_len": 60e-9,  # Sec
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"90": 0.1, "180": 0.2},
            },
            "wiring": {
                "lo_freq": 4.59e9,
                "I": ["con1", 1],
                "Q": ["con1", 2],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
        {
            "f_01": 4.65097e9,  # Hz
            "anharmonicity": None,
            "rabi_freq": None,
            "t1": 18e-6,
            "t2": None,
            "t2*": 5e-6,
            "driving": {
                "gate_len": 60e-9,  # Sec
                "gate_sigma": 20e-9,
                "gate_shape": "gaussian",
                "angle2volt": {"90": 0.1, "180": 0.2},
            },
            "wiring": {
                "lo_freq": 4.72e9,
                "I": ["con1", 3],
                "Q": ["con1", 4],
                "correction_matrix": IQ_imbalance(0, 0),
            },
        },
    ],
    "single_qubit_operations": [
        {"direction": "x", "angle": 180},
        {"direction": "x", "angle": -180},
        {"direction": "x", "angle": 90},
        {"direction": "x", "angle": -90},
        {"direction": "y", "angle": 180},
        {"direction": "y", "angle": -180},
        {"direction": "y", "angle": 90},
        {"direction": "y", "angle": -90},
    ],
    "two_qubit_gates": {
        "cr_c0t1": {
            "control": 0,
            "target": 1,
            "gate_time": None,
            "correction_matrix": IQ_imbalance(0, 0),
            "interaction strength": {
                "ix": None,
                "iy": None,
                "iz": None,
                "zx": None,
                "zy": None,
                "zz": None,
            },
        },
        "cr_c1t0": {
            "control": 1,
            "target": 0,
            "gate_time": None,
            "correction_matrix": IQ_imbalance(0, 0),
            "interaction strength": {
                "ix": None,
                "iy": None,
                "iz": None,
                "zx": None,
                "zy": None,
                "zz": None,
            },
        },
    },
    "running strategy": {"running": True, "start": [], "end": []},
}


# layer 2: parameters modifications over bare state
# state["OPX_config"]["elements"]["rr1"]["intermediate_frequency"] = int(52e6)
# state["OPX_config"]["mixers"]["mixer_rr"][0]["intermediate_frequency"] = int(
#     52e6
# )


# layer 3: structure change to bare state


def add_qubits(state: Dict, config: Dict):
    for q in range(len(state["qubits"])):
        wiring = state["qubits"][q]["wiring"]
        config["elements"][f"q{q}"] = {
            "mixInputs": {
                "I": (wiring["I"][0], wiring["I"][1]),
                "Q": (wiring["Q"][0], wiring["Q"][1]),
                "lo_frequency": wiring["lo_freq"],
                "mixer": f"mixer_q{q}",
            },
            "intermediate_frequency": round(state["qubits"][q]["f_01"])
            - wiring["lo_freq"],
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
            },
        }
        if f"mixer_q{q}" not in config["mixers"]:
            config["mixers"][f"mixer_q{q}"] = []
        config["mixers"][f"mixer_q{q}"].append(
            {
                "intermediate_frequency": round(state["qubits"][q]["f_01"])
                - wiring["lo_freq"],
                "lo_frequency": wiring["lo_freq"],
                "correction": wiring["correction_matrix"],
            }
        )


def add_readout_resonators(state, config):
    for r, v in enumerate(state["readout_resonators"]):  # r - idx, v - value
        config["elements"][f"rr{r}"] = {
            "mixInputs": {
                "I": (v["wiring"]["I"][0], v["wiring"]["I"][1]),
                "Q": (v["wiring"]["Q"][0], v["wiring"]["Q"][1]),
                "lo_frequency": round(state["readout_lo_freq"]),
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": round(
                v["f_res"] - state["readout_lo_freq"]
            ),
            "operations": {
                "cw": "const_pulse",
                "readout": f"readout_pulse_rr{r}",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": v["wiring"]["time_of_flight"],
            "smearing": 0,
        }
        if "mixer_rr" not in config["mixers"]:
            config["mixers"]["mixer_rr"] = []
        config["mixers"]["mixer_rr"].append(
            {
                "intermediate_frequency": round(
                    v["f_res"] - state["readout_lo_freq"]
                ),
                "lo_frequency": state["readout_lo_freq"],
                "correction": v["wiring"]["correction_matrix"],
            }
        )
        config["waveforms"][f"readout_wf_rr{r}"] = {
            "type": "constant",
            "sample": v["readout_amplitude"],
        }
        config["pulses"][f"readout_pulse_rr{r}"] = {
            "operation": "measurement",
            "length": round(state["readout_length"] * 1e9),
            "waveforms": {
                "I": f"readout_wf_rr{r}",
                "Q": "zero_wf",
            },
            "integration_weights": {
                "cos": "cosine_weights",
                "sin": "sine_weights",
                "minus_sin": "minus_sine_weights",
                "rotated_cos": f"rotated_cosine_weights_rr{r}",
                "rotated_sin": f"rotated_sine_weights_rr{r}",
                "rotated_minus_sin": f"rotated_minus_sine_weights_rr{r}",
            },
            "digital_marker": "ON",
        }
        rot_angle_in_pi = v["rotation_angle"] / 180. * np.pi
        config["integration_weights"]["cosine_weights"] = {
            "cosine": [(1.0, round(state["readout_length"] * 1e9))],
            "sine": [(0.0, round(state["readout_length"] * 1e9))],
        }
        config["integration_weights"]["sine_weights"] = {
            "cosine": [(0.0, round(state["readout_length"] * 1e9))],
            "sine": [(1.0, round(state["readout_length"] * 1e9))],
        }
        config["integration_weights"]["minus_sine_weights"] = {
            "cosine": [(0.0, round(state["readout_length"] * 1e9))],
            "sine": [(-1.0, round(state["readout_length"] * 1e9))],
        }
        config["integration_weights"][f"rotated_cosine_weights_rr{r}"] = {
            "cosine": [(np.cos(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
            "sine": [(-np.sin(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
        }
        config["integration_weights"][f"rotated_sine_weights_rr{r}"] = {
            "cosine": [(np.sin(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
            "sine": [(np.cos(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
        }
        config["integration_weights"][f"rotated_minus_sine_weights_rr{r}"] = {
            "cosine": [(-np.sin(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
            "sine": [(-np.cos(rot_angle_in_pi), round(state["readout_length"] * 1e9))],
        }


def add_qb_rot(
    state: Dict,
    config: Dict,
    q: int,
    angle: int,
    direction: str,
    wf_I: List,
    wf_Q: List = None,
):
    """Add single qubit operation

    Args:
        state (Dict): state of the system
        config (Dict): OPX config we are building/editing
        q (int): index of qubit to whom we are adding operation
        angle (int): angle in degrees
        direction (str):  "x" or "y"
        wf_I: I waveform
        wf_Q: Optional, Q waveform
    """
    if direction not in ["x", "y"]:
        raise ValueError(
            f"Only x and y are accepted directions, received {direction}"
        )
    if q >= len(state["qubits"]):
        raise ValueError(
            f"Qubit {q} is not configured in state. Please add qubit q first."
        )
    if type(angle) != int:
        raise ValueError("Only integers are accepted as angle.")

    if direction == "x":
        direction_angle = 0
    elif direction == "y":
        direction_angle = np.pi / 2

    if wf_Q is None:
        wf_Q = np.zeros(len(wf_I))
    if len(wf_I) != len(wf_Q):
        raise ValueError("wf_I and wf_Q should have same lengths!")

    wv = np.sign(angle) * (
        wf_I * np.cos(direction_angle) + wf_Q * np.sin(direction_angle)
    )
    if np.all((wv == 0)):
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"] = {
            "type": "constant"
        }
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"]["sample"] = 0
    else:
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"] = {
            "type": "arbitrary"
        }
        config["waveforms"][f"{direction}{angle}_I_wf_q{q}"]["samples"] = wv

    wv = np.sign(angle) * (
        (-wf_I) * np.sin(direction_angle) + wf_Q * np.cos(direction_angle)
    )
    if np.all((wv == 0)):
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"] = {
            "type": "constant"
        }
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"]["sample"] = 0
    else:
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"] = {
            "type": "arbitrary"
        }
        config["waveforms"][f"{direction}{angle}_Q_wf_q{q}"]["samples"] = wv

    config["pulses"][f"{direction}{angle}_pulse_q{q}"] = {
        "operation": "control",
        "length": len(wf_I),
        "waveforms": {
            "I": f"{direction}{angle}_I_wf_q{q}",
            "Q": f"{direction}{angle}_Q_wf_q{q}",
        },
    }
    config["elements"][f"q{q}"]["operations"][
        f"{direction}{angle}"
    ] = f"{direction}{angle}_pulse_q{q}"


def add_cross_resonance_gates(state, config):
    for gate, v in state["two_qubit_gates"].items():
        config["elements"][gate] = {
            "mixInputs": {
                "I": (
                    config["elements"][f"q{v['control']}"]["mixInputs"]["I"][0],
                    config["elements"][f"q{v['control']}"]["mixInputs"]["I"][1],
                ),
                "Q": (
                    config["elements"][f"q{v['control']}"]["mixInputs"]["Q"][0],
                    config["elements"][f"q{v['control']}"]["mixInputs"]["Q"][1],
                ),
                "lo_frequency": config["elements"][f"q{v['control']}"][
                    "mixInputs"
                ]["lo_frequency"],
                "mixer": f"mixer_q{v['control']}",
            },
            "intermediate_frequency": round(
                state["qubits"][v["target"]]["f_01"]
            )
            - state["qubits"][v["control"]]["wiring"]["lo_freq"],
            "operations": {
                "cw": "const_pulse",
            },
        }
        config["mixers"][f"mixer_q{v['control']}"].append(
            {
                "intermediate_frequency": round(
                    state["qubits"][v["target"]]["f_01"]
                )
                - state["qubits"][v["control"]]["wiring"]["lo_freq"],
                "lo_frequency": state["qubits"][v["control"]]["wiring"][
                    "lo_freq"
                ],
                "correction": v["correction_matrix"],
            }
        )


def build_config(state):
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {
                },
                "digital_outputs": {},
                "analog_inputs": {
                    1: {
                        "offset": state["OPX_config"]["controllers"]["con1"][
                            "analog_inputs"
                        ]["1"]["offset"],
                        "gain_db": state["OPX_config"]["controllers"]["con1"][
                            "analog_inputs"
                        ]["1"]["gain_db"],
                    },
                    2: {
                        "offset": state["OPX_config"]["controllers"]["con1"][
                            "analog_inputs"
                        ]["2"]["offset"],
                        "gain_db": state["OPX_config"]["controllers"]["con1"][
                            "analog_inputs"
                        ]["2"]["gain_db"],
                    },
                },
            }
        },
        "elements": {},
        "pulses": {
            "const_pulse": {
                "operation": "control",
                "length": state["OPX_config"]["pulses"]["const_pulse"][
                    "length"
                ],
                "waveforms": {
                    "I": "const_wf",
                    "Q": "zero_wf",
                },
            },
            "saturation_pulse": {
                "operation": "control",
                "length": state["OPX_config"]["pulses"]["saturation_pulse"][
                    "length"
                ],
                "waveforms": {"I": "saturation_drive_wf", "Q": "zero_wf"},
            },
        },
        "waveforms": {
            "const_wf": {
                "type": "constant",
                "sample": state["OPX_config"]["waveforms"]["const_wf"][
                    "sample"
                ],
            },
            "saturation_drive_wf": {
                "type": "constant",
                "sample": state["OPX_config"]["waveforms"][
                    "saturation_drive_wf"
                ]["sample"],
            },
            "zero_wf": {"type": "constant", "sample": 0.0},
        },
        "digital_waveforms": {
            "ON": {"samples": [(1, 0)]},
        },
        "integration_weights": {},
        "mixers": {},
    }
    for key, value in state["OPX_config"]["controllers"]["con1"]["analog_outputs"].items():
        config["controllers"]["con1"]["analog_outputs"][key] = {
            "offset": state["OPX_config"]["controllers"]["con1"][
                "analog_outputs"
            ][key]["offset"]
        }

    add_qubits(state, config)

    add_readout_resonators(state, config)

    for single_qubit_operation in state["single_qubit_operations"]:
        for q in range(len(state["qubits"])):
            if state["qubits"][q]["driving"]["gate_shape"] == "gaussian":
                add_qb_rot(
                    state,
                    config,
                    q,
                    single_qubit_operation["angle"],
                    single_qubit_operation["direction"],
                    state["qubits"][q]["driving"]["angle2volt"][
                        str(abs(single_qubit_operation["angle"]))
                    ]
                    * gaussian(
                        round(state["qubits"][q]["driving"]["gate_len"] * 1e9),
                        round(
                            state["qubits"][q]["driving"]["gate_sigma"] * 1e9
                        ),
                    ),
                )  # +180 and -180 have same amplitude
            else:
                raise ValueError(
                    f'Gate shape {state["qubits"][q]["driving"]["gate_shape"]} not recognized.'
                )

    add_cross_resonance_gates(state, config)
    return config


def generate_bootstrap_state():
    config = build_config(state)
    # from qm.QuantumMachinesManager import QuantumMachinesManager
    # qmm = QuantumMachinesManager(host="172.16.2.103", port=82)
    # qm = qmm.open_qm(config)

    import json

    with open("bootstrap_state.json", "w") as fp:
        json.dump(state, fp, indent=2)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return json.JSONEncoder.default(self, obj)

    with open("configuration_parsed.json", "w") as fp:
        json.dump(build_config(state), fp, cls=NumpyEncoder, indent=2)
        # convert_file.write(json.dumps(build_config(state)))


if __name__ == "__main__":
    generate_bootstrap_state()
