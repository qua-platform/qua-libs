import math

import numpy as np
from typing import Sequence, List

from quam.components import MWChannel, IQChannel
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from calibration_utils.readout_optimization_3d.parameters import Parameters


def get_max_accumulated_readouts(qubits: Sequence[AnyTransmon], node_parameters: Parameters) -> int:
    """
    In order to perform simultaneous, I/Q based accumulated demodulation during
    qubit readout, each qubit requires four `demod.accumulated` processing blocks.
    Each of these consumes a resource on the PPU, up to a maximum of 16 for the
    MW-FEM and 20 for the OPX+, leading to a maximum of 3 qubits on the MW-FEM
    and 4 on the OPX+ if the qubits are measured simultaneously. If the measurement
    isn't multiplexed, the limit is never reached.
    """
    res_per_demod = 4

    if node_parameters.multiplexed:
        if isinstance(qubits[0].resonator, MWChannel):
            resource_limit = 16
        elif isinstance(qubits[0].resonator, IQChannel):
            resource_limit = 20
        else:
            raise TypeError("Unrecognized resonator type {type(qubits[0].resonator)}, couldn't")
    else:
        return len(qubits)

    # Since we have to play the readout pulse on every non-measured resonator,
    # this will also occupy a thread. So, we have to make sure that
    # `max_accumulated_readouts` * 4 + `leftover_qubits` < limit
    max_reads = 0
    while max_reads < len(qubits):
        if ((max_reads + 1) * res_per_demod + len(qubits) - (max_reads + 1)) > resource_limit:
            break
        max_reads += 1

    return max_reads


def generate_measurement_batches(qubits: Sequence[AnyTransmon], max_accumulated_readouts: int):
    """
    Generate fair measurement groups ensuring all qubits are measured equally.

    Returns:
        list of lists: Sequence of measurement groups.
    """
    subsets_indices = _generate_balanced_subsets_indices(len(qubits), max_accumulated_readouts)

    measurement_batches = []
    for subset_indices in subsets_indices:
        measurement_batches.append([qubits[i] for i in subset_indices])

    return measurement_batches


def _generate_balanced_subsets_indices(N: int, S: int) -> List[List[int]]:
    """
    Suppose we have N objects that we want to distribute into K subsets of
    fixed size S, such that each of the N objects appear exactly J times
    across all subsets.

    This can always be satisfied using:
     - K = LCM(N, S) / S subsets
     - J = LCM(N, S) / N occurrences

    """
    K = math.lcm(N, S) // S

    objects = list(range(N)) * K
    subsets = [objects[i : i + S] for i in range(K)]

    return subsets
