from collections.abc import Sequence
from typing import Any, List


def broadcast_param_to_list(value: Any, n: int) -> List[Any]:
    """
    Normalize a parameter to a list of length n.

    - Scalar -> [value] * n
    - Sequence (not str/bytes):
        * len == n  -> list(value)
        * len == 1  -> [value[0]] * n
        * otherwise -> ValueError
    """
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        seq = list(value)
        if len(seq) == n:
            return seq
        if len(seq) == 1:
            return seq * n
        raise ValueError(f"Expected length 1 or {n}, got {len(seq)}.")
    return [value] * n
