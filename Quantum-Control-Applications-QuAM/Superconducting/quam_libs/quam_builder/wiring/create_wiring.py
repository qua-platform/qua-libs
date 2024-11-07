from typing import List, Dict, Any
from functools import reduce

from qualang_tools.wirer import Connectivity
from qualang_tools.wirer.connectivity.element import QubitPairReference
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from qualang_tools.wirer.instruments.instrument_channel import AnyInstrumentChannel

from .create_analog_ports import create_octave_port, create_mw_fem_port, create_lf_opx_plus_port
from .create_digital_ports import create_digital_output_port
from .paths import *


def create_wiring(connectivity: Connectivity) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references which
    can be used to generate QuAM `port` objects.
    """
    wiring = {}
    for element_id, element in connectivity.elements.items():
        for line_type, channels in element.channels.items():
            if line_type in [WiringLineType.RESONATOR, WiringLineType.DRIVE, WiringLineType.FLUX]:
                for k, v in qubit_wiring(channels).items():
                    set_nested_value_with_path(wiring, f"qubits/{element_id}/{line_type.value}/{k}", v)

            elif line_type == WiringLineType.COUPLER:
                for k, v in qubit_pair_wiring(channels, element_id).items():
                    set_nested_value_with_path(wiring, f"qubit_pairs/{element_id}/{line_type.value}/{k}", v)

            else:
                raise ValueError(f"Unknown line type {line_type}")

    return wiring


def qubit_wiring(channels: List[AnyInstrumentChannel]) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references for a
    list of channels from a single qubit and the same line type.
    """
    qubit_line_wiring = {}
    for channel in channels:
        if not (channel.signal_type == "digital" and channel.io_type == "input"):
            key, reference = get_channel_port(channel, channels)
            qubit_line_wiring[key] = reference

    return qubit_line_wiring


def qubit_pair_wiring(channels: List[AnyInstrumentChannel], element_id: QubitPairReference) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references for a
    list of channels from a single qubit and the same line type.
    """
    qubit_pair_line_wiring = {
        "control_qubit": f"{QUBITS_BASE_JSON_PATH}/q{element_id.control_index}",
        "target_qubit": f"{QUBITS_BASE_JSON_PATH}/q{element_id.target_index}",
    }
    for channel in channels:
        if not (channel.signal_type == "digital" and channel.io_type == "input"):
            key, reference = get_channel_port(channel, channels)
            qubit_pair_line_wiring[key] = reference

    return qubit_pair_line_wiring


def get_channel_port(channel: AnyInstrumentChannel, channels: List[AnyInstrumentChannel]) -> tuple:
    if channel.signal_type == "digital":
        key, reference = create_digital_output_port(channel)
    else:
        if channel.instrument_id == "octave":
            key, reference = create_octave_port(channel)
        elif channel.instrument_id == "mw-fem":
            key, reference = create_mw_fem_port(channel)
        elif channel.instrument_id in ["lf-fem", "opx+"]:
            key, reference = create_lf_opx_plus_port(channel, channels)
        else:
            raise ValueError(f"Unknown instrument type {channel.instrument_id}")

    return key, reference


def set_nested_value_with_path(d: Dict, path: str, value: Any):
    """Set a value in a nested dictionary using a '/' separated path."""
    keys = path.split("/")  # Split the path into keys
    reduce(lambda d, key: d.setdefault(key, {}), keys[:-1], d)[keys[-1]] = value
