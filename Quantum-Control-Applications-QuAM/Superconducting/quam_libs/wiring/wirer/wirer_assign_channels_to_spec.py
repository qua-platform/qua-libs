from typing import List, Type

from quam_libs.wiring.connectivity.wiring_spec import WiringSpec
from quam_libs.wiring.instruments import Instruments
from quam_libs.wiring.instruments.instrument_channels import InstrumentChannel


def assign_channels_to_spec(
    spec: WiringSpec,
    instruments: Instruments,
    channel_types: List[Type[InstrumentChannel]],
    same_con: bool = False,
    same_slot: bool = False,
):

    candidate_channels = _assign_channels_to_spec(
        spec, instruments, channel_types, same_con, same_slot
    )

    # if candidate channels satisfy all the required channel types
    if len(candidate_channels) == len(channel_types):
        for channel in candidate_channels:
            # remove candidate channel from stack of available channels
            instruments.available_channels[type(channel)].remove(channel)
            for element in spec.elements:
                # assign channel to the specified element
                if spec.line_type not in element.channels:
                    element.channels[spec.line_type] = []
                element.channels[spec.line_type].append(channel)

    return len(candidate_channels) == len(channel_types)


def _assign_channels_to_spec(
    spec: WiringSpec,
    instruments: Instruments,
    channel_types: List[Type[InstrumentChannel]],
    same_con: bool,
    same_slot: bool,
    allocated_channels=None,
):
    """
    Recursive function to find any valid combination of channel allocations
    given a wiring specification, a stack of available channels in the
    instruments setup, and a list of desired channel types.
    """
    if allocated_channels is None:
        allocated_channels = []

    # extract the lead/initial channel type
    target_channel_type = channel_types[0]

    # filter available channels according to the specification
    available_channels = list(
        filter(
            spec.io_spec.make_channel_filter(),  # filter function
            instruments.available_channels.get(target_channel_type, []),  # iterable
        )
    )

    candidate_channels = []
    for channel in available_channels:
        # make sure to not re-allocate a channel
        if channel in allocated_channels:
            continue

        candidate_channels = [channel]

        # base case: all channels allocated properly
        if len(channel_types) == 1:
            break

        # recursive case: allocate remaining channels
        else:
            # require that all future channels have the same `con` and `slot` as the candidate
            if same_con:
                spec.io_spec.con = channel.con
            if same_slot:
                spec.io_spec.slot = channel.slot

            # recursively allocate the remaining channels
            subsequent_channels = _assign_channels_to_spec(
                spec,
                instruments,
                channel_types[1:],
                same_con,
                same_slot,
                allocated_channels=candidate_channels,
            )

            candidate_channels.extend(subsequent_channels)

            # without a candidate channel for every type, try the next available channel
            if len(candidate_channels) == len(channel_types):
                break

            # otherwise, a successful allocation has been made
            else:
                continue

    return candidate_channels
