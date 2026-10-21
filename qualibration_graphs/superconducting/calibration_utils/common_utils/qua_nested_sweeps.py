"""Nested QUA sweep context managers (Cartesian product over ``for_`` variables)."""

from contextlib import contextmanager

from qm.qua import for_


@contextmanager
def nested_sweep(loop_vars, idx=0, *, upper=2):
    """Recursively nest QUA ``for_`` loops from 0 to ``upper - 1`` over each variable."""
    if idx == len(loop_vars):
        yield
        return

    var = loop_vars[idx]
    with for_(var, 0, var < upper, var + 1):
        with nested_sweep(loop_vars, idx + 1, upper=upper):
            yield
