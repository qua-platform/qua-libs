from typing import List

from .paths import OCTAVES_BASE_JSON_PATH, PORTS_BASE_JSON_PATH
from qualang_tools.wirer.instruments.instrument_channel import AnyInstrumentChannel


def create_octave_port(channel: AnyInstrumentChannel) -> (str, str):
    """
    Generates a key/JSON reference pair from which a QuAM port can be created
    for a single Octave channel.
    """
    if channel.io_type == "output":
        key = "frequency_converter_up"
    elif channel.io_type == "input":
        key = "frequency_converter_down"
    else:
        raise ValueError(f"Unknown IO type {channel.io_type}")

    # todo: handle "octave" in OPX+ vs "oct" in new QOP?
    reference = OCTAVES_BASE_JSON_PATH
    reference += f"/oct{channel.con}"
    reference += f"/RF_{channel.io_type}s"
    reference += f"/{channel.port}"

    return key, reference


def create_mw_fem_port(channel: AnyInstrumentChannel) -> (str, str):
    """
    Generates a key/JSON reference pair from which a QuAM port can be created
    for a mw-fem channel
    """
    reference = PORTS_BASE_JSON_PATH
    reference += f"/mw_{channel.io_type}s"

    reference += f"/con{channel.con}"
    reference += f"/{channel.slot}"
    reference += f"/{channel.port}"

    key = f"opx_{channel.io_type}"

    return key, reference


def create_lf_opx_plus_port(channel: AnyInstrumentChannel, channels: List[AnyInstrumentChannel]) -> (str, str):
    """
    Generates a key/JSON reference pair from which a QuAM port can be created
    for a single non-octave channel.
    """
    reference = PORTS_BASE_JSON_PATH
    reference += f"/analog_{channel.io_type}s"  # process IO type
    if channel.instrument_id == "opx+":
        reference += f"/con{channel.con}"
        reference += f"/{channel.port}"
    else:
        reference += f"/con{channel.con}"
        reference += f"/{channel.slot}"
        reference += f"/{channel.port}"

    # process whether single I/O channel or mixed I/O.
    channels_with_same_type = get_objects_with_same_type(channel, channels)
    if len(channels_with_same_type) == 1:
        key = f"opx_{channel.io_type}"

    elif len(channels_with_same_type) == 2:
        # assuming lower port number corresponds to I- and higher to Q-quadrature
        if channel.port == min([channel_with_same_type.port for channel_with_same_type in channels_with_same_type]):
            key = f"opx_{channel.io_type}_I"
        else:
            key = f"opx_{channel.io_type}_Q"

    else:
        raise NotImplementedError(f"Can't handle when channel number is not 1 or 2, got {len(channels_with_same_type)}")

    return key, reference


def get_objects_with_same_type(obj, lst):
    """Returns all objects in the list that have the same type as the given object."""
    return [item for item in lst if isinstance(item, type(obj))]  # Return items with the same type
