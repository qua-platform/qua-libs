from typing import Literal
from qualibrate.core import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters

from calibration_utils.common_utils.experiment import (
    HeraldedInitializeParameters,
    ParityDiffAnalysisParameters,
    QubitPairExperimentNodeParameters,
)


class CrossTalkRabiSpecificParameters(RunnableParameters):
    """Parameters specific to 09c_cross_talk_rabi.

    Two degrees of freedom only:
      * the measured ``qubit_pair`` — initialised and read out via PSB, and
      * the ``drive_qubit`` — a qubit outside that pair, driven off-resonance.

    For each measured pair the node sweeps the XY drive amplitude of every qubit
    that is not part of the pair, one driver at a time. The Rabi-like response of
    the pair to each external driver quantifies the microwave cross-talk.

    Example: pick ``qubit_pairs=["q2_q3"]`` and the node will (sequentially) drive
    q1 while reading out q2-q3, then drive q4 while reading out q2-q3, reporting a
    cross-talk amplitude for each. Passing several pairs builds a full
    measured-pair x drive-qubit cross-talk matrix.
    """

    num_shots: int = 300
    """Number of averages to perform. Default is 300."""
    min_amp_factor: float = 0.0
    """Minimum amplitude factor for the drive-qubit operation. Default is 0.0."""
    max_amp_factor: float = 2.0
    """Maximum amplitude factor for the drive-qubit operation (must be < 2). Default is 2.0."""
    amp_factor_step: float = 0.02
    """Step size for the amplitude factor. Default is 0.02."""
    operation: Literal["x180", "x90", "y90"] = "x180"
    """The operation played on each drive qubit."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    CrossTalkRabiSpecificParameters,
    QubitPairExperimentNodeParameters,
    ParityDiffAnalysisParameters,
):
    """Parameter set for 09c_cross_talk_rabi."""
