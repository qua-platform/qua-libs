from quam import QuAM
import numpy as np
from scipy.signal.windows import gaussian


def find_lo_freq(state: QuAM, qubit_index: int):
    for x in state.drive_line:
        if qubit_index in x.qubits:
            return x.freq
    raise ValueError(f"Qubit {qubit_index} is not associated with any LO in system state!")


def add_qubits(state: QuAM, config: dict):
    for q in range(len(state.qubits)):
        wiring = state.qubits[q].wiring
        lo_freq = find_lo_freq(state, q)
        config["elements"][f"q{q}"] = {
            "mixInputs": {
                "I": (wiring.I[0], wiring.I[1]),
                "Q": (wiring.Q[0], wiring.Q[1]),
                "lo_frequency": lo_freq,
                "mixer": f"mixer_drive_line{q}",
            },
            "intermediate_frequency": round(state.qubits[q].f_01)
                                      - lo_freq,
            "operations": {
                "cw": "const_pulse",
                "saturation": "saturation_pulse",
            },
        }

        # add flux element
        config["elements"][f"q{q}_flux"] = {
            "singleInput": {
                "port": (wiring.flux_line[0], wiring.flux_line[1])
            },
            "operations": {
                "cw": "const_flux_pulse",
            },
        }
        # add operations for flux line
        for op in state.qubits[0].sequence_states:
            config["elements"][f"q{q}_flux"]["operations"][op.name] = f"q{q}_flux_{op.name}"

            # add pulse
            config["pulses"][f"q{q}_flux_{op.name}"] = {
                "operation": "control",
                "length": op.length,
                "waveforms": {
                    "single": f"q{q}_flux_{op.name}_wf"
                },
            }
            config["waveforms"][f"q{q}_flux_{op.name}_wf"] = {
                "type": "constant",
                "sample": op.amplitude,
            }
        config["controllers"][wiring.flux_line[0]]["analog_outputs"][str(wiring.flux_line[1])]["filter"] = {
            "feedforward": wiring.flux_filter_coef.feedforward,
            "feedback": wiring.flux_filter_coef.feedback
        }

    # add cross talk
    for i in range(len(state.crosstalk_matrix.fast)):
        crosstalk = {}
        q_i = state.qubits[i]
        for j in range(len(state.crosstalk_matrix.fast[i])):
            q_j = state.qubits[j]
            crosstalk[q_j.wiring.flux_line[1]] = state.crosstalk_matrix.fast[i][j]
        config["controllers"][q_i.wiring.flux_line[0]]["analog_outputs"][str(q_i.wiring.flux_line[1])][
            "crosstalk"] = crosstalk


def add_mixers(state: QuAM, config: dict):
    for q in range(len(state.drive_line)):
        lo_freq = state.drive_line[q].freq

        if f"mixer_drive_line{q}" not in config["mixers"]:
            config["mixers"][f"mixer_drive_line{q}"] = []

        for j in state.drive_line[q].qubits:
            config["mixers"][f"mixer_drive_line{q}"].append(
                {
                    "intermediate_frequency": round(state.qubits[j].f_01)
                                              - lo_freq,
                    "lo_frequency": lo_freq,
                    "correction": state.qubits[j].wiring.correction_matrix,
                }
            )

        for z in range(len(state.qubits)):
            for t in range(len(state.drive_line[q].qubits)):
                if z == state.drive_line[q].qubits[t]:
                    config["elements"][f"q{z}"]["mixInputs"]["mixer"] = f"mixer_drive_line{q}"


def add_readout_resonators(state: QuAM, config: dict):
    for r, v in enumerate(state.readout_resonators):  # r - idx, v - value
        readout_line = state.readout_lines[v.wiring.readout_line_index]
        config["elements"][f"rr{r}"] = {
            "mixInputs": {
                "I": (v.wiring.I[0], v.wiring.I[1]),
                "Q": (v.wiring.Q[0], v.wiring.Q[1]),
                "lo_frequency": round(readout_line.lo_freq),
                "mixer": "mixer_rr",
            },
            "intermediate_frequency": round(
                v.f_res - readout_line.lo_freq
            ),
            "operations": {
                "cw": "const_pulse",
                "readout": f"readout_pulse_rr{r}",
            },
            "outputs": {
                "out1": ("con1", 1),
                "out2": ("con1", 2),
            },
            "time_of_flight": v.wiring.time_of_flight,
            "smearing": 0,
        }
        if "mixer_rr" not in config["mixers"]:
            config["mixers"]["mixer_rr"] = []
        config["mixers"]["mixer_rr"].append(
            {
                "intermediate_frequency": round(
                    v.f_res - readout_line.lo_freq
                ),
                "lo_frequency": readout_line.lo_freq,
                "correction": v.wiring.correction_matrix,
            }
        )
        config["waveforms"][f"readout_wf_rr{r}"] = {
            "type": "constant",
            "sample": v.readout_amplitude,
        }
        config["pulses"][f"readout_pulse_rr{r}"] = {
            "operation": "measurement",
            "length": round(readout_line.length * 1e9),
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
        rot_angle_in_pi = v.rotation_angle / 180.0 * np.pi
        config["integration_weights"]["cosine_weights"] = {
            "cosine": [(1.0, round(readout_line.length * 1e9))],
            "sine": [(0.0, round(readout_line.length * 1e9))],
        }
        config["integration_weights"]["sine_weights"] = {
            "cosine": [(0.0, round(readout_line.length * 1e9))],
            "sine": [(1.0, round(readout_line.length * 1e9))],
        }
        config["integration_weights"]["minus_sine_weights"] = {
            "cosine": [(0.0, round(readout_line.length * 1e9))],
            "sine": [(-1.0, round(readout_line.length * 1e9))],
        }
        config["integration_weights"][f"rotated_cosine_weights_rr{r}"] = {
            "cosine": [
                (np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
            "sine": [
                (-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
        }
        config["integration_weights"][f"rotated_sine_weights_rr{r}"] = {
            "cosine": [
                (np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
            "sine": [
                (np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
        }
        config["integration_weights"][f"rotated_minus_sine_weights_rr{r}"] = {
            "cosine": [
                (-np.sin(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
            "sine": [
                (-np.cos(rot_angle_in_pi), round(readout_line.length * 1e9))
            ],
        }


def add_qb_rot(
        state: QuAM,
        config: dict,
        q: int,
        angle: int,
        direction: str,
        wf_I: list,
        wf_Q: list = None,
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
    if q >= len(state.qubits):
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
            wf_I * np.cos(direction_angle) - wf_Q * np.sin(direction_angle)
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
            wf_I * np.sin(direction_angle) + wf_Q * np.cos(direction_angle)
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


def add_control_operation_single(config, element, operation_name, wf):
    pulse_name = element + "_" + operation_name + "_in"
    config["waveforms"][pulse_name + "_single"] = {
        "type": "arbitrary",
        "samples": list(wf),
    }
    config["pulses"][pulse_name] = {
        "operation": "control",
        "length": len(wf),
        "waveforms": {"I": pulse_name + "_single"},
    }
    config["elements"][element]["operations"][operation_name] = pulse_name


def add_control_operation_iq(config, element, operation_name, wf_i, wf_q):
    pulse_name = element + "_" + operation_name + "_in"
    config["waveforms"][pulse_name + "_i"] = {
        "type": "arbitrary",
        "samples": list(wf_i),
    }
    config["waveforms"][pulse_name + "_q"] = {
        "type": "arbitrary",
        "samples": list(wf_q),
    }
    config["pulses"][pulse_name] = {
        "operation": "control",
        "length": len(wf_i),
        "waveforms": {"I": pulse_name + "_i", "Q": pulse_name + "_q"},
    }
    config["elements"][element]["operations"][operation_name] = pulse_name


def add_analog_outputs(state: QuAM, config: dict):
    for o in state.analog_outputs:
        if o.controller not in config["controllers"]:
            config["controllers"][o.controller] = {}
        if "analog_outputs" not in config["controllers"][o.controller]:
            config["controllers"][o.controller]["analog_outputs"] = {}
        config["controllers"][o.controller]["analog_outputs"][
            str(o.output)
        ] = {"offset": o.offset}


def add_analog_inputs(state: QuAM, config: dict):
    for i in state.analog_inputs:
        if i.controller not in config["controllers"]:
            config["controllers"][i.controller] = {}
        if "analog_inputs" not in config["controllers"][i.controller]:
            config["controllers"][i.controller]["analog_inputs"] = {}
        config["controllers"][i.controller]["analog_inputs"][
            str(i.input)
        ] = {
            "offset": i.offset,
            "gain_db": i.gain_db,
        }


def add_analog_waveforms(state: QuAM, config):
    for wf in state.analog_waveforms:
        if wf.type == "constant":
            if len(wf.samples) != 1:
                raise ValueError(
                    f'Constant analog waveform {wf.name} has to have samples length of 1 (currently {len(wf.samples)})'
                )

            config["waveforms"][wf.name] = {
                "type": wf.type,
                "sample": wf.samples[0],
            }
        else:
            if len(wf.samples) <= 1:
                raise ValueError(
                    f'Analog waveform {wf.name} has single sample, and should be then of type "constant" instead of {wf.type}.'
                )
            config["waveforms"][wf["name"]] = {
                "type": wf.type,
                "samples": wf.samples,
            }


def add_digital_waveforms(state: QuAM, config: dict):
    for wf in state.digital_waveforms:
        config["digital_waveforms"][wf.name] = {"samples": wf.samples}


def add_pulses(state: QuAM, config: dict):
    for pulse in state.pulses:
        config["pulses"][pulse.name] = {
            "operation": pulse.operation,
            "length": pulse.length,
            "waveforms": {
                "I": pulse.waveforms.I,
                "Q": pulse.waveforms.Q,
            },
        }

    for pulse in state.pulses_single:
        config["pulses"][pulse.name] = {
            "operation": pulse.operation,
            "length": pulse.length,
            "waveforms": {
                "single": pulse.waveforms.single
            },
        }


def build_config(state: QuAM):
    config = {
        "version": 1,
        "controllers": {
            "con1": {
                "analog_outputs": {},
                "digital_outputs": {},
                "analog_inputs": {},
            }
        },
        "elements": {},
        "pulses": {},
        "waveforms": {},
        "digital_waveforms": {},
        "integration_weights": {},
        "mixers": {},
    }

    add_analog_outputs(state, config)

    add_analog_inputs(state, config)

    add_analog_waveforms(state, config)

    add_digital_waveforms(state, config)

    add_pulses(state, config)

    add_qubits(state, config)

    add_readout_resonators(state, config)

    add_mixers(state, config)

    for single_qubit_operation in state.single_qubit_operations:
        for q in range(len(state.qubits)):
            if state.qubits[q].driving.gate_shape == "gaussian":
                if abs(single_qubit_operation.angle) == 180:
                    amplitude = state.qubits[q].driving.angle2volt.deg180
                elif abs(single_qubit_operation.angle) == 90:
                    amplitude = state.qubits[q].driving.angle2volt.deg90
                else:
                    raise ValueError(
                        "Unknown angle for single qubit operation"
                        f" {single_qubit_operation.angle}"
                    )
                add_qb_rot(
                    state,
                    config,
                    q,
                    single_qubit_operation.angle,
                    single_qubit_operation.direction,
                    amplitude
                    * gaussian(
                        round(state.qubits[q].driving.gate_len * 1e9),
                        round(
                            state.qubits[q].driving.gate_sigma * 1e9
                        ),
                    ),
                )  # +180 and -180 have same amplitude
            else:
                raise ValueError(
                    f'Gate shape {state.qubits[q].driving.gate_shape} not recognized.'
                )

    return config


if __name__ == "__main__":
    # if we execute directly config.py this tests that configuration is ok

    machine = QuAM("quam_bootstrap_state.json")
    config = build_config(machine)