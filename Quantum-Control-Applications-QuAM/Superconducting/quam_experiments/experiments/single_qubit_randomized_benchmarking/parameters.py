from typing import Optional
from pydantic import ConfigDict
from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from quam_experiments.parameters import (
    QubitsExperimentNodeParameters,
    CommonNodeParameters,
)


class NodeSpecificParameters(RunnableParameters):
    use_state_discrimination: bool = False
    """state discrimination flag"""
    use_strict_timing: bool = False
    """state discrimination flag"""
    num_random_sequences: int = 100  # Number of random sequences
    """state discrimination flag"""
    num_averages: int = 20
    """state discrimination flag"""
    max_circuit_depth: int = 1000  # Maximum circuit depth
    """state discrimination flag"""
    delta_clifford: int = 20
    """delta"""
    seed: int = 345324
    """seed"""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitsExperimentNodeParameters,
):
    model_config = ConfigDict(use_attribute_docstrings=True)
    pass
