from typing import Dict, List, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class BarrierCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for barrier compensation scans."""

    barrier_compensation_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of barrier gate -> list of gates (typically plungers) to scan
    against it for compensation.  Only local cross-talk pairs need to be
    specified.
    Example: {"barrier_12": ["virtual_dot_1", "virtual_dot_2"]}.
    If None, must be generated from the machine (not yet implemented)."""
