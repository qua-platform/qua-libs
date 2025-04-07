from typing import List, Dict, TypeVar
from collections.abc import MutableSequence

T = TypeVar("T")


class BatchableList(MutableSequence[T]):
    def __init__(self, items: List[T], batch_groups: List[List[int]]):
        """
        A batchable list is a data-structure that behaves exactly like a list
        when interfaced with like a list. However, it privately contains an
        internal batch_groups attribute, defining how it can be "batched", that is,
        its elements can be arbitrarily grouped into "batches". The list of batches
        can be obtained by calling the .batch() method.

        Each "batch" is an unordered group of items represented by a dictionary,
        whose key is the item's original index in the list, and whose value is
        the item itself.

        Example:
            >>> items = ["a", "b", "c", "d"]
            >>> batch_groups = [[0, 2], [1, 3]]
            >>> batchable_items = BatchableList(items, batch_groups)
            >>> batchable_items.batch()
            [
                {
                    0: "a",
                    2: "c"
                },
                {
                    1: "b",
                    3: "d"
                }
            ]
        """
        self._items = items
        self._batch_groups = batch_groups

        # Validate that batch_groups covers all indices exactly once
        expected_indices = set(range(len(items)))
        actual_indices = {idx for group in batch_groups for idx in group}
        if actual_indices != expected_indices:
            raise ValueError("batch_groups must cover each index exactly once")

    # Required methods to implement for MutableSequence
    def __getitem__(self, index: int) -> T:
        return self._items[index]

    def __setitem__(self, index: int, value: T):
        self._items[index] = value

    def __delitem__(self, index: int) -> None:
        del self._items[index]

    def __len__(self) -> int:
        return len(self._items)

    def insert(self, index: int, value: T) -> None:
        self._items.insert(index, value)

    def __repr__(self) -> str:
        return repr(self._items)

    def batch(self) -> List[Dict[int, T]]:
        """
        todo: clear docstring
        Examples:
            [["q1", "q2", "q3", "q4"]] --> fully multiplexed
            [["q1"], ["q2"], ["q3"], ["q4"]] --> fully sequential
            [["q1", "q3"], ["q2", "q4"]] --> multiplexed by batches
        :return:
        """
        batched_items = []
        for group in self._batch_groups:
            batch = {idx: self._items[idx] for idx in group}
            batched_items.append(batch)
        return batched_items

    def get_names(self):
        names = []
        for q in self._items:
            names.append(q.name)
        return names
