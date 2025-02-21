import math
import numpy as np
from typing import Sequence

from quam.components import MWChannel, IQChannel
from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_experiments.experiments.readout_optimization_3d.parameters import Parameters


def get_max_accumulated_readouts(qubits: Sequence[AnyTransmon], node_parameters: Parameters) -> int:
    """
    In order to perform simultaneous, I/Q based accumulated demodulation during
    qubit readout, each qubit requires four `demod.accumulated` processing blocks.
    Each of these consumes a resource on the PPU, up to a maximum of 16 for the
    MW-FEM and 20 for the OPX+, leading to a maximum of 4 qubits on the MW-FEM
    and 5 on the OPX+ if the qubits are measured simultaneously. If the measurement
    isn't multiplexed, the limit is never reached.
    """
    if node_parameters.multiplexed:
        if isinstance(qubits[0].resonator, MWChannel):
            max_accumulated_readouts = 4
        elif isinstance(qubits[0].resonator, IQChannel):
            max_accumulated_readouts = 5
        else:
            raise TypeError("Unrecognized resonator type {type(qubits[0].resonator)}, couldn't")
    else:
        max_accumulated_readouts = np.inf

    return max_accumulated_readouts


def generate_measurement_batches(qubits: Sequence[AnyTransmon], max_accumulated_readouts: int):
    """
    Generate fair measurement groups ensuring all qubits are measured equally.

    Returns:
        list of lists: Sequence of measurement groups.
    """
    skip_group_size = math.gcd(len(qubits), max_accumulated_readouts)
    skip_groups = [qubits[i:i+skip_group_size] for i in range(0, len(qubits), skip_group_size)]

    groups = []
    for skip_group in skip_groups:
        groups.append([qubit for qubit in qubits if qubit not in skip_group])

    return groups
