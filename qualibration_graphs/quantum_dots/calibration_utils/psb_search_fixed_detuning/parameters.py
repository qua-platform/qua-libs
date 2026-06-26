from typing import List, Literal, Optional

from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters

from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    QubitsExperimentNodeParameters,
)
from qualibration_libs.parameters import CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 1000
    """Number of shots per preparation state (no-pi and pi-pulse arms). Each shot yields one IQ sample."""
    detuning: Optional[float] = None
    """If set, temporarily overrides the measure macro detuning (V) for this run; reverted in ``update_state``."""
    init_state_label: Literal["decay", "no_decay"] = "decay"
    """Which spin state is prepared WITHOUT the pi pulse.
    'decay'    – no pi pulse loads the decay (triplet T) state; pi pulse gives the non-decay (singlet S) state.
    'no_decay' – no pi pulse loads the non-decay (singlet S) state; pi pulse gives the decay (triplet T) state."""
    analysis_model: Literal["barthel", "gmm"] = "barthel"
    """Which model to use for fitting the labeled IQ shots.
    'barthel' – physics-based Barthel 1D readout model with MCMC (``fit_raw_data``).
    'gmm'     – 2-component Gaussian mixture model via PCA projection + sklearn GMM."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    QubitsExperimentNodeParameters,
    NodeSpecificParameters,
):
    pass
