"""Parameter definitions for gate set tomography (AIS) experiment."""

from typing import List, Literal, Optional

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """GST-specific parameters for circuit depth and shot count."""

    qubits: Optional[List[str]] = ["q1"]
    """Single qubit to characterize. GST is 1Q-only; default is q1."""
    max_circuit_depth_in_power: int = 9
    """Maximum circuit depth as a power of two: depths are 2**0 .. 2**N. Default is 9."""
    num_shots: int = 100
    """Number of repetitions per GST circuit (germ). Default is 100."""
    reset_type: Literal["thermal", "active", "active_gef"] = "active"
    """Qubit reset method. Default is 'active'."""
    use_state_discrimination: bool = True
    """GST requires state discrimination. Default is True."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    """Combined parameters for gate set tomography (AIS) node."""

    pass
