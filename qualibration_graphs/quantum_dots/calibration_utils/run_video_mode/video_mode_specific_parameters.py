from qualibrate.parameters import RunnableParameters
from typing import List, Optional


class VideoModeCommonParameters(RunnableParameters):
    run_in_video_mode: bool = False
    """Optionally open Video Mode with the qualibration node."""
    virtual_gate_set_id: str = None
    """Name of the associated VirtualGateSet in your QPU. """
    video_mode_port: int = 8050
    """Localhost port to open VideoMode with"""
    dc_control: bool = False
    """If an associated external DC offset exists."""
