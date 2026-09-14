from typing import Optional

from qualibrate.core.parameters import RunnableParameters


class HeraldedInitializeParameters(RunnableParameters):
    target_state: Optional[int] = None
    """The state you want to initialize into for heralded initialization."""
    max_loops: int = 100
    """Maximum number of initialization loops for heralded initialization."""
    return_n_loops: bool = False
    """Whether to return the number of times it has looped over the initialise sequence to achieve the desired result."""
