from typing import Literal

from .wirer_assign_channels_to_spec import assign_channels_to_spec
from .wirer_exception import WirerException
from ..connectivity.wiring_spec import WiringSpec
from ..instruments import Instruments
from .wirer_channel_map import get_channel_mapping


def allocate_dc_channels(spec: WiringSpec, instruments: Instruments):
    """
    Try to allocate DC channels to an LF-FEM or OPX+ to satisfy the spec.
    """
    if not (
        try_allocate_channels(spec, instruments, "lf-fem")
        or try_allocate_channels(spec, instruments, "opx+")
    ):
        raise WirerException(spec)


def allocate_rf_channels(spec: WiringSpec, instruments: Instruments):
    """
    Try to allocate RF channels to a MW-FEM. If that doesn't work, look for a
    combination of LF-FEM I/Q and Octave channels, or OPX+ I/Q and Octave
    channels.
    """
    if not try_allocate_channels(spec, instruments, "mw-fem"):
        if not (
            try_allocate_channels(spec, instruments, "octave")
            and (
                try_allocate_channels(spec, instruments, "lf-fem", num=2)
                or try_allocate_channels(spec, instruments, "opx+", num=2)
            )
        ):
            raise WirerException(spec)


def try_allocate_channels(
    spec: WiringSpec,
    instruments: Instruments,
    module: Literal["mw-fem", "lf-fem", "opx+", "octave"],
    num: int = 1,
) -> bool:
    """
    Identify which channel types would satisfy the spec for a specific QM
    module, then try to assign such channels to the spec, returning True
    if it succeeds.
    """
    channel_types = get_channel_mapping(module, spec.io_spec.type, num=num)
    return assign_channels_to_spec(
        spec, instruments, channel_types, same_con=True, same_slot=True
    )
