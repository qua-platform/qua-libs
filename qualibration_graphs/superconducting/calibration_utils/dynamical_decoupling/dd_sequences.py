"""Dynamical decoupling (DD) sequences built from evenly spaced pi pulses, played window by window.

A sequence is defined by the pi pulses of one block (e.g. XY4 = X Y X Y). One idle window of duration T_w holds N pulses
(a multiple of the block size), each pulse sitting in the middle of an equal cell of free evolution:
    tau - P1 - 2tau - P2 - ... - 2tau - PN - tau,   with  N * (2 tau + t_pi) = T_w
Repeating the window keeps the 2tau spacing across window boundaries, as in consecutive QEC rounds. The 4 ns rounding
of tau is spread over the cells (they differ by at most one clock cycle), so there is no lump of unprotected time.

To add a sequence, add an entry to DD_SEQUENCES and its name to the `sequence` Literal in parameters.py.
A pulse is either the name of a qubit.xy operation (e.g. "y180"), or an (operation, phase in degrees) pair. The phase is
applied as a virtual Z rotation of the frame around that pulse only, with the same sign convention as the pulse axis
angle, so ("x180", 90) is a Y180 and ("x180", 180) is a -X180.
"""

from contextlib import nullcontext
from dataclasses import dataclass
from typing import List, Tuple, Union

from qm.qua import for_, strict_timing_

Pulse = Union[str, Tuple[str, float]]

# Minimum QUA wait duration in clock cycles (4 ns each)
MIN_WAIT_CC = 4


@dataclass(frozen=True)
class DDSequence:
    name: str
    pulses: Tuple[Pulse, ...]
    """Pi pulses of one block."""
    prep: str = "x90"
    """Operation preparing (and, played again at the end, mapping back) the superposition."""

    @property
    def pulses_per_block(self) -> int:
        return len(self.pulses)

    @property
    def operations(self) -> List[str]:
        """Names of the qubit.xy operations used by the pi pulses."""
        return sorted({_split_pulse(p)[0] for p in self.pulses})

    def pulse_train(self, n_pulses: int) -> List[Tuple[str, float]]:
        """The N pi pulses of one window, as (operation, phase_deg) pairs: the block repeated N / block size times."""
        if n_pulses <= 0 or n_pulses % self.pulses_per_block:
            raise ValueError(f"{self.name}: the number of pulses must be a multiple of {self.pulses_per_block}.")
        return [_split_pulse(p) for p in self.pulses] * (n_pulses // self.pulses_per_block)

    def window_steps(self, n_pulses: int, free_cc: int) -> List[tuple]:
        """One window as a list of ("wait", clock_cycles) and ("play", operation, phase_deg) steps.

        The free evolution time free_cc (window minus pulses, in clock cycles) is split into N cells that differ by at
        most one clock cycle; each pulse sits in the middle of its cell.
        """
        train = self.pulse_train(n_pulses)
        cells = [(i + 1) * free_cc // n_pulses - i * free_cc // n_pulses for i in range(n_pulses)]
        before = [c // 2 for c in cells]
        after = [c - b for c, b in zip(cells, before)]
        if min(before) < MIN_WAIT_CC:
            raise ValueError(
                f"{self.name}: {n_pulses} pulses leave {4 * min(cells)} ns between pulses, below the "
                f"{8 * MIN_WAIT_CC} ns minimum."
            )
        steps = []
        for i, (operation, phase) in enumerate(train):
            steps.append(("wait", before[0] if i == 0 else after[i - 1] + before[i]))
            steps.append(("play", operation, phase))
        steps.append(("wait", after[-1]))
        return steps

    def play(self, qubit, n_windows, loop_var, n_pulses: int, free_cc: int, strict: bool = True):
        """prep - [window] x n_windows - prep, where n_windows can be a QUA variable.

        One window (N pulses and free_cc clock cycles of free evolution) is unrolled in Python; only the number of
        windows is swept in real time.
        """
        steps = self.window_steps(n_pulses, free_cc)
        with strict_timing_() if strict else nullcontext():
            qubit.xy.play(self.prep)
            with for_(loop_var, 0, loop_var < n_windows, loop_var + 1):
                for step in steps:
                    if step[0] == "wait":
                        qubit.xy.wait(step[1])
                    else:
                        _, operation, phase = step
                        if phase:
                            qubit.xy.frame_rotation_2pi(phase / 360)
                        qubit.xy.play(operation)
                        if phase:
                            qubit.xy.frame_rotation_2pi(-phase / 360)
            qubit.xy.play(self.prep)


def _split_pulse(pulse: Pulse) -> Tuple[str, float]:
    if isinstance(pulse, str):
        return pulse, 0.0
    operation, phase = pulse
    return operation, float(phase)


_XY4 = ("x180", "y180", "x180", "y180")
# XY4 followed by its time-reversed copy
_XY8 = _XY4 + _XY4[::-1]
# XY8 followed by XY8 with all pulses phase-inverted (-X, -Y)
_XY16 = _XY8 + tuple((p, 180) for p in _XY8)

DD_SEQUENCES = {
    "CPMG": DDSequence("CPMG", ("y180", "y180")),
    "XY4": DDSequence("XY4", _XY4),
    "XY8": DDSequence("XY8", _XY8),
    "XY16": DDSequence("XY16", _XY16),
}


def get_dd_sequence(name: str) -> DDSequence:
    if name not in DD_SEQUENCES:
        raise ValueError(f"Unknown DD sequence '{name}'. Available: {list(DD_SEQUENCES)}")
    return DD_SEQUENCES[name]
