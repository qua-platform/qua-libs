from qualibrate.parameters import RunnableParameters
from typing import List, Optional

class VideoModeSpecificParameters(RunnableParameters):
    x_axis_name: str = None
    """Name of physical or virtual X axis."""
    y_axis_name: str = None
    """Name of physical or virtual Y axis."""
    sensor_names: Optional[List[str]] = None
    """Sensors that you would like to sweep"""
    frequency_span_in_mhz: float = 30.0
    """Span of frequencies to sweep in MHz. Default is 30 MHz."""
    frequency_step_in_mhz: float = 0.1
    """Step size for frequency sweep in MHz. Default is 0.1 MHz."""
