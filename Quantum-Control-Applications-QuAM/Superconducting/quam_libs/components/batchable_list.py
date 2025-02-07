from typing import List, Any, Dict
from collections.abc import MutableSequence
from quam_libs.components import Transmon


class BatchableList(MutableSequence):
    def __init__(self, items: List[Any], multiplexed: bool = True):
        self._items = items
        self.multiplexed = multiplexed

    # Required methods to implement for MutableSequence
    def __getitem__(self, index: int) -> Transmon:
        return self._items[index]

    def __setitem__(self, index: int, value: Transmon) -> None:
        self._items[index] = value

    def __delitem__(self, index: int) -> None:
        del self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, index: int, value: Transmon) -> None:
        self._items.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._items)

    def batch(self) -> List[Dict[int, Any]]:
        if self.multiplexed:
            return [{i: item for i, item in enumerate(self._items)}]
        else:
            return [{i: item} for i, item in enumerate(self._items)]
