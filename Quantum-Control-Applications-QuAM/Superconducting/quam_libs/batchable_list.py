from typing import List, Any, Dict, Union
from collections.abc import MutableSequence
from quam_libs.components.superconducting.qubit import FixedFrequencyTransmon, FluxTunableTransmon
# from quam_experiments.node_parameters import QubitsExperimentNodeParameters, MultiplexableNodeParameters


class BatchableList(MutableSequence):
    def __init__(self, items: List[Any], multiplexed: bool = True):
        self._items = items
        self.multiplexed = multiplexed

    # Required methods to implement for MutableSequence
    def __getitem__(self, index: int) -> Union[FixedFrequencyTransmon, FluxTunableTransmon]:
        return self._items[index]

    def __setitem__(self, index: int, value: Union[FixedFrequencyTransmon, FluxTunableTransmon]) -> None:
        self._items[index] = value

    def __delitem__(self, index: int) -> None:
        del self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, index: int, value: Union[FixedFrequencyTransmon, FluxTunableTransmon]) -> None:
        self._items.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._items)

    def batch(self) -> List[Dict[int, Any]]:
        if self.multiplexed:
            return [{i: item for i, item in enumerate(self._items)}]
        else:
            return [{i: item} for i, item in enumerate(self._items)]


# def make_batchable_list(items, node_parameters: QubitsExperimentNodeParameters) -> BatchableList:
#     if isinstance(node_parameters, MultiplexableNodeParameters):
#         multiplexed = node_parameters.multiplexed
#     else:
#         multiplexed = False
#
#     return BatchableList(items, multiplexed)
