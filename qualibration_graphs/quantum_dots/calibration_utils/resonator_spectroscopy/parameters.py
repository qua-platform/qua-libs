from qualibrate import NodeParameters
from qualibrate.parameters import RunnableParameters
from qualibration_libs.parameters import QubitsExperimentNodeParameters, CommonNodeParameters
from typing import List, Optional

class NodeSpecificParameters(RunnableParameters):
    num_shots: int = 100
    """Number of averages to perform. Default is 100."""
    # quantum_dots: Optional[List[str]] = None
    # """Selection of Quantum Dot names in your VirtualGateSet"""
    virtual_gate_set_id: str = None
    """Name of the associated VirtualGateSet in your QPU. """
    sensor_names: Optional[List[str]] = None
    """Sensors that you would like to measure in your VirtualGateSet"""
    Video_Mode: bool = False
    """Optionally open Video Mode with the qualibration node."""
    frequency_span_in_mhz: int = 30
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
    dc_control: bool = False
    """If an associated external DC offset exists."""




class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
):
    pass
