"""Parameter definitions for two-qubit gate set tomography (AIS)."""

from typing import ClassVar, Literal

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """2Q GST-specific parameters for circuit depth, shots, and CZ macro."""

    max_circuit_depth_in_power: int = 2
    """Maximum germ length as a power of two: lengths are 2**0 .. 2**N. Default is 2 (max length 4).
    2Q GST circuit counts grow quickly; keep this small unless you know the runtime."""
    num_shots: int = 100
    """Number of repetitions per GST circuit. Default is 100."""
    reset_type: Literal["thermal", "active", "active_gef"] = "active"
    """Qubit reset method. Default is 'active'."""
    use_state_discrimination: bool = True
    """GST requires state discrimination. Default is True."""
    use_fiducial_pair_reduction: bool = True
    """Use pyGSTi fiducial-pair reduction when the model pack supports it. Default is True."""
    operation: Literal[
        "cz_flattop",
        "cz_unipolar",
        "cz_bipolar",
        "cz_flattop_erf",
        "cz_SNZ",
        "cz_gaussian_bipolar",
        "cz_gaussian_unipolar",
    ] = "cz_unipolar"
    """CZ macro name on the qubit pair. Default is 'cz_unipolar'."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for two-qubit gate set tomography (AIS)."""

    targets_name: ClassVar[str] = "qubit_pairs"
