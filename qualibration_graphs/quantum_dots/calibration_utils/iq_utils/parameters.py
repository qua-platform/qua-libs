from typing import Literal
from qualibrate.core.parameters import RunnableParameters

__all__ = ["IQParameters", "IQSweepParameters"]

class IQParameters(RunnableParameters): 
    um_shots: int = 2000
    """Number of runs to perform per sweep point. Default is 2000."""
    operation: Literal["readout", "readout_QND"] = "readout"
    """Type of operation to perform. Default is "readout"."""

class IQSweepParameters(IQParameters): 
    sweep_name: str = "detuning"
    """Name of the swept coordinate in ds_raw (e.g. "detuning", "integration_time")."""
    optimization_metric: Literal["fidelity", "visibility"] = "fidelity"
    """Metric used to pick the optimal sweep value for state updates.
    Both the fidelity and visibility optima are always recorded regardless of this choice."""
    labeled_states: bool = False
    """Whether ds_raw contains labelled S/T preparations (Ig,Qg,Ie,Qe, as in a
    Rabi-style IQ-blob experiment) or a single mixed-state acquisition (I,Q,
    as in a PSB search where loading is random). Determines whether a confusion
    matrix is computed. Default False = mixed-state mode."""
