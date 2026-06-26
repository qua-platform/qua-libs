"""Parameters for pair-targeted single-qubit randomized benchmarking.

This is the qubit-pair analogue of the single-qubit RB node (node 14): it
reuses the same node-specific RB controls (depths, shots, circuits) but
targets ``qubit_pairs`` instead of ``qubits``, running RB on both
``qubit_pair.qubit_target`` and ``qubit_pair.qubit_control`` in turn.

It exists so the ORBIT gate-optimisation graph can verify, on the same
target convention as the ORBIT nodes (``qubit_pairs``), the single-qubit
gate fidelity reached for each member of a pair.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from qualibrate.core import NodeParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitPairExperimentNodeParameters,
)
from calibration_utils.single_qubit_randomized_benchmarking.parameters import (
    NodeSpecificParameters,
)


class PairRBParameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Full parameter set for pair-targeted single-qubit RB.

    Inherits all RB controls (``num_circuits_per_length``, ``num_shots``,
    ``max_circuit_depth``, ``delta_clifford``, ``log_scale``, ``seed``,
    ``operation_x90``, ``operation_x180``) from
    :class:`NodeSpecificParameters`, but resolves targets via ``qubit_pairs``.
    """

    # Operate on qubit pairs (graph target injection writes `qubit_pairs`).
    targets_name: ClassVar[str] = "qubit_pairs"

    def get_depths(self) -> np.ndarray:
        """Generate circuit depths — identical convention to single-qubit RB.

        Depth *d* means *d-1* random Cliffords + 1 recovery (inverse) = *d*
        total Cliffords (standard RB convention).

        - ``log_scale=True``: powers of two 2, 4, 8, ... up to
          ``max_circuit_depth``.
        - ``log_scale=False``: linearly spaced by ``delta_clifford`` up to
          ``max_circuit_depth`` (first value forced to 1).
        """
        if self.log_scale:
            depths = []
            current_depth = 2
            while current_depth <= self.max_circuit_depth:
                depths.append(current_depth)
                current_depth *= 2
            return np.array(depths, dtype=int)

        # Linear spacing by delta_clifford.  Round the upper bound to the nearest
        # whole multiple of delta_clifford (≥ 1 step) instead of requiring exact
        # divisibility, so any (max_circuit_depth, delta_clifford) pair works.
        n_steps = max(1, round(self.max_circuit_depth / self.delta_clifford))
        max_depth = n_steps * self.delta_clifford
        depths_arr = np.arange(0, max_depth + 0.1, self.delta_clifford, dtype=int)
        depths_arr[0] = 1
        return depths_arr