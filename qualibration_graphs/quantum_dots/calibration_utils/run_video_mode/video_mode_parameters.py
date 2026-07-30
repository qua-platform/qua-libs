from typing import Literal, Optional

from qualibrate.core.parameters import RunnableParameters


class VideoModeParameters(RunnableParameters):
    run_in_video_mode: bool = True
    """Optionally open Video Mode with the qualibration node."""
    virtual_gate_set_id: Optional[str] = None
    """Name of the associated VirtualGateSet in your QPU."""
    video_mode_port: int = 8002
    """Localhost port to open VideoMode with."""
    result_type: Literal["I", "Q", "Amplitude", "Phase"] = "I"
    """Result type to display in Video Mode."""
