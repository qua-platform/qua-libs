"""Parameters module for GHZ Z-basis population measurement."""

from typing import ClassVar, List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from qualibration_libs.parameters.experiment import BaseExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Node-specific parameters for GHZ Z-basis measurement."""

    qubit_groups: Optional[List[str]] = None
    """Ordered qubit chains (3–5 qubits), one dash-separated entry per group, e.g.
    ``["qD2-qD1-qD3"]``. GHZ prep is a linear CZ ladder (pair_01, pair_12, …); order
    must match physical couplers, state bits, and node 38."""

    num_shots: int = 2000
    """Number of shots per group. Default is 2000."""

    operation: Literal["cz_flattop", "cz_unipolar", "cz_bipolar", "cz_flattop_erf", "cz_SNZ"] = "cz_unipolar"
    """CZ macro applied along the chain during GHZ preparation."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    BaseExperimentNodeParameters,
    NodeSpecificParameters,
):
    """Combined parameters for GHZ Z-basis population measurement."""

    targets_name: ClassVar[str] = "qubit_groups"
