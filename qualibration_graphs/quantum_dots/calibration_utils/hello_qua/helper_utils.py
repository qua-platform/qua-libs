from typing import Sequence

__all__ = [
    "extract_longest_readout_time",
    "extract_vgs_id",
]


def extract_longest_readout_time(sensors: Sequence, pulse_name: str = "readout"):
    """Extract the length of the longest readout pulse in our sensors list."""
    longest_readout_pulse_length = max([s.readout_resonator.operations[pulse_name].length for s in sensors])
    return longest_readout_pulse_length


def extract_vgs_id(quantum_dots: Sequence):
    """
    Extract the name of the virtual gate set ID associated with a Sequence of QDs.
    This will return the first one it finds.
    """
    vgs_set = {qd.voltage_sequence.gate_set.name for qd in quantum_dots}
    if len(vgs_set) > 1:
        raise NotImplementedError("QDs from multiple VirtualGateSets found.")
    return next(iter(vgs_set))
