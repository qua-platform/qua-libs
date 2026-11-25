from typing import List, Literal, Optional

from pydantic import Field
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters


class NodeSpecificParameters(RunnableParameters):
    pairs: Optional[List[str]] = None
    """Directed pairs to measure, as "source-victim" (e.g. ["qA3-qA2", "qD2-qD3"]): the first qubit plays its gate (the
    source), the second picks up the leak (the victim), so "qA3-qA2" and "qA2-qA3" are two different measurements.
    Both qubits must be among the selected qubits. If None, every pair among the selected qubits. Default is None."""
    num_shots: int = Field(default=200, ge=1)
    """Number of averages per grid point. Default is 200."""
    r_max: float = Field(default=0.2, gt=0, lt=2)
    """Radius of the complex tone grid, in units of the source pulse amplitude; the grid must contain the null -c.
    The starting radius of every pair: it is adapted to the strength of the source pulse and of the victim line, unless
    train_pulses is set. Default is 0.2."""
    center: Literal["zero", "previous"] = "zero"
    """Grid centre: 0, or -c of the coefficient stored by a previous run (to refine it). Default is 'zero'."""
    train_pulses: Optional[int] = Field(default=None, ge=2)
    """Number of x180 of the source qubit in the train, with the grid radius fixed at r_max (debugging). If None, the
    train length and the radius are chosen per pair so that the largest expected signal is about 1 rad.
    Default is None."""
    min_stark_detuning_mhz: float = Field(default=10, ge=0)
    """Pairs detuned by less than this are measured by the excited population, the others by Ramsey (Stark phase);
    pairs whose source is this close to the victim's 1-2 transition are skipped. Default is 10 MHz."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    pass
