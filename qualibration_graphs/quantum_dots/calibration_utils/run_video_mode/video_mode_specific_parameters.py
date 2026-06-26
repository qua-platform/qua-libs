from qualibrate.core.parameters import RunnableParameters
from typing import List, Optional, Literal


class VideoModeCommonParameters(RunnableParameters):
    run_in_video_mode: bool = True
    """Optionally open Video Mode with the qualibration node."""
    virtual_gate_set_id: Optional[str] = None
    """Name of the associated VirtualGateSet in your QPU. """
    video_mode_port: int = 8002
    """Localhost port to open VideoMode with"""
    dc_control: bool = False
    """If an associated external DC offset exists."""
    result_type: Literal["I", "Q", "Amplitude", "Phase"] = "I"
