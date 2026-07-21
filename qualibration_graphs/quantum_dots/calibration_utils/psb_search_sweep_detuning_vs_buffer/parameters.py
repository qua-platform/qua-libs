from typing import Literal, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters
from calibration_utils.common_utils.experiment import QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of shots per (detuning, buffer_duration) point."""

    qubit_pair_to_initialize: Optional[str] = None
    """If set, initialise using this pair while measuring selected qubit_pairs."""

    qubit_to_pulse: Optional[str] = None
    """Optionally apply x180 to one qubit before measurement."""

    barrier_gate_voltage: float = 0.0
    """Barrier gate voltage applied together with detuning."""

    detuning_min: float = -0.1
    """Minimum detuning value for sweep (V)."""

    detuning_max: float = 0.1
    """Maximum detuning value for sweep (V)."""

    detuning_points: int = 21
    """Number of detuning points."""

    ramp_duration: int = 40
    """Ramp duration to measure point (ns)."""

    buffer_duration_min: int = 16
    """Minimum pre-readout buffer duration (ns, multiple of 4)."""

    buffer_duration_max: int = 400
    """Maximum pre-readout buffer duration (ns, multiple of 4)."""

    buffer_duration_step: int = 16
    """Buffer-duration sweep step (ns, multiple of 4)."""

    initialization_macro: Literal["empty", "initialize"] = "empty"
    """Preparation macro applied before the sweep point."""

    pca_metric: Literal["pc1_std", "iq_trace"] = "pc1_std"
    """Map quantity to highlight in plots; analysis computes both."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Parameter set for 06e_PSB_search_opx_sweep_detuning_vs_buffer."""
