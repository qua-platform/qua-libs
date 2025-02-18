from typing import List

from pydantic import Field
from qualibrate.parameters import RunnableParameters

from quam_experiments.parameters.batchable_list import BatchableList


class MultiplexableNodeParameters(RunnableParameters):
    multiplexed: bool = Field(
        default=False,
        description="Whether to play control pulses, readout pulses and active/thermal reset "
                    "at the same time for all qubits (True) or to play the experiment sequentially"
                    "for each qubit (False)"
    )


def make_batchable_list_from_multiplexed(items: List, multiplexed: bool) -> BatchableList:
    if multiplexed:
        batched_groups = [[i for i in range(len(items))]]
    else:
        batched_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batched_groups)
