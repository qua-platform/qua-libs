"""Parameter definitions for Cross-Entropy Benchmarking (XEB) experiments.

Supports both single-qubit and two-qubit XEB modes. Single-qubit mode
benchmarks 1Q gate layer fidelity; two-qubit mode interleaves a CZ gate
to benchmark the 2Q gate fidelity.
"""

from __future__ import annotations

from typing import List, Literal, Optional

import numpy as np
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for XEB experiments."""

    n_sequences: int = 20
    """Number of random circuits per depth."""
    n_shots: int = 200
    """Number of measurement shots per circuit."""
    depth_min: int = 5
    """Minimum circuit depth (number of gate cycles)."""
    depth_max: int = 200
    """Maximum circuit depth."""
    depth_step: int = 5
    """Step size for depth sweep."""
    seed: Optional[int] = None
    """Seed for the QUA pseudo-random number generator."""

    gate_set: Literal["sw", "t"] = "sw"
    """Gate set for random 1Q gates. 'sw' = {SX, SY, SW}, 't' = {SX, SY, T}."""

    apply_two_qubit_gate: bool = False
    """If True, interleave a CZ gate between cycles (2Q XEB mode)."""
    cz_macro_name: str = "cz"
    """Name of the CZ macro on the qubit pair (used in 2Q mode)."""

    estimate_2q_unitary: bool = False
    """If True, run Nelder-Mead 2Q unitary estimation after XEB analysis."""

    def get_depths(self) -> np.ndarray:
        """Return array of circuit depths for the XEB sweep."""
        return np.arange(self.depth_min, self.depth_max, self.depth_step, dtype=int)


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Unified parameters for XEB experiments (1Q and 2Q modes).

    When apply_two_qubit_gate=False (default), uses ``qubits`` for targeting.
    When apply_two_qubit_gate=True, uses ``qubit_pairs`` for targeting.
    """

    qubit_pairs: Optional[List[str]] = None
    """List of qubit pair names to benchmark (used in 2Q mode)."""


SingleQubitParameters = Parameters
TwoQubitParameters = Parameters
