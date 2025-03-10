from typing import List, Dict, Any
from functools import reduce
from qualang_tools.wirer import Connectivity
from qualang_tools.wirer.connectivity.element import QubitPairReference, QubitReference
from qualang_tools.wirer.connectivity.wiring_spec import WiringLineType
from qualang_tools.wirer.instruments.instrument_channel import AnyInstrumentChannel
from quam_builder.builder.qop_connectivity.create_analog_ports import (
    create_octave_port,
    create_mw_fem_port,
    create_lf_opx_plus_port,
    create_external_mixer_reference,
)
from quam_builder.builder.qop_connectivity.create_digital_ports import (
    create_digital_output_port,
)
from quam_builder.builder.qop_connectivity.paths import *


def create_wiring(connectivity: Connectivity) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references which can be used to generate QuAM `port` objects.

    Parameters:
    connectivity (Connectivity): The connectivity configuration.

    Returns:
    dict: A dictionary containing QuAM-compatible JSON references.

    Raises:
    ValueError: If an unknown line type is encountered.
    """
    wiring = {}
    for element_id, element in connectivity.elements.items():
        for line_type, channels in element.channels.items():
            if line_type in [
                WiringLineType.RESONATOR,
                WiringLineType.DRIVE,
                WiringLineType.FLUX,
            ]:
                for k, v in qubit_wiring(channels, element_id, line_type).items():
                    set_nested_value_with_path(wiring, f"qubits/{element_id}/{line_type.value}/{k}", v)

            elif line_type in [
                WiringLineType.COUPLER,
                WiringLineType.CROSS_RESONANCE,
                WiringLineType.ZZ_DRIVE,
            ]:
                for k, v in qubit_pair_wiring(channels, element_id).items():
                    set_nested_value_with_path(wiring, f"qubit_pairs/{element_id}/{line_type.value}/{k}", v)

            else:
                raise ValueError(f"Unknown line type {line_type}")

    return wiring


def qubit_wiring(
    channels: List[AnyInstrumentChannel],
    element_id: QubitReference,
    line_type: WiringLineType,
) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references for a list of channels from a single qubit and the same line type.

    Parameters:
    channels (List[AnyInstrumentChannel]): The list of instrument channels.
    element_id (QubitReference): The ID of the qubit element.
    line_type (WiringLineType): The type of wiring line.

    Returns:
    dict: A dictionary containing QuAM-compatible JSON references.
    """
    qubit_line_wiring = {}
    for channel in channels:
        if channel.instrument_id == "external-mixer":
            key, reference = create_external_mixer_reference(channel, element_id, line_type)
            qubit_line_wiring[key] = reference
        elif not (channel.signal_type == "digital" and channel.io_type == "input"):
            key, reference = get_channel_port(channel, channels)
            qubit_line_wiring[key] = reference

    return qubit_line_wiring


def qubit_pair_wiring(channels: List[AnyInstrumentChannel], element_id: QubitPairReference) -> dict:
    """
    Generates a dictionary containing QuAM-compatible JSON references for a list of channels from a single qubit pair and the same line type.

    Parameters:
    channels (List[AnyInstrumentChannel]): The list of instrument channels.
    element_id (QubitPairReference): The ID of the qubit pair element.

    Returns:
    dict: A dictionary containing QuAM-compatible JSON references.
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
    """
    Determines the key and JSON reference for a given channel.

    Parameters:
    channel (AnyInstrumentChannel): The instrument channel for which the reference is created.
    channels (List[AnyInstrumentChannel]): A list of all instrument channels.

    Returns:
    tuple: A tuple containing the key and the JSON reference.

    Raises:
    ValueError: If the instrument type is unknown.
    """
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
    """
    Sets a value in a nested dictionary using a '/' separated path.

    Parameters:
    d (Dict): The dictionary in which the value will be set.
    path (str): The '/' separated path to the value.
    value (Any): The value to set.
    """
    keys = path.split("/")  # Split the path into keys
    reduce(lambda d, key: d.setdefault(key, {}), keys[:-1], d)[keys[-1]] = value
