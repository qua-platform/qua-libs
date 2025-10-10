from typing import ClassVar, List, Literal, Optional

import numpy as np
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters

from quam.core import operation


class NodeSpecificParameters(RunnableParameters):
    """
    Parameters for configuring a CZ phase compensation experiment.

    Attributes:
        num_shots (int): Number of averages to perform. Default is 100.
        num_frames (int): Number of phase frames to sweep. Default is 17.
        operation (Literal["cz_flattop", "cz_unipolar"]): Type of CZ operation to perform. Default is "cz_unipolar".
    """

    num_shots: int = 100
    num_frames: int = 17
    operation: Literal["cz_flattop", "cz_unipolar"] = "cz_unipolar"
    use_state_discrimination: bool = True


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    targets_name: ClassVar[str] = "qubit_pairs"
