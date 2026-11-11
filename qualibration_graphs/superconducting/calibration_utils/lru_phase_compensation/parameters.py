"""Parameters for the f-LRU phase compensation (node 26c)."""

from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitsExperimentNodeParameters
from pydantic import Field

from quam_config.lru_components import LRU_OPERATION


class NodeSpecificParameters(RunnableParameters):
    """Ramsey around the LRU with a swept virtual Z, read as g/e."""

    num_shots: int = 200
    """Number of averages. Default is 200."""
    num_frames: int = 25
    """Number of virtual-Z frames swept over one turn [0, 1). Default is 25."""
    operation: str = LRU_OPERATION
    """Pump pulse on ``qubit.lru``. Default is "lru"."""
    num_lru_repetitions: int = Field(1, ge=1)
    """LRUs between the two x90 pulses, each with the stored correction. More repetitions amplify a small
    residual phase, but the residual times this number must stay within half a turn, so start with 1 when
    no correction is stored. Default is 1."""
    skip_lru_pulse: bool = False
    """Control run: replace each LRU with an idle of the same length on the qubit drive, with no virtual Z.
    The Ramsey contrast then shows the qubit's own decay over the same time, to compare with the pumped run;
    nothing is written to the state. Default is False."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
