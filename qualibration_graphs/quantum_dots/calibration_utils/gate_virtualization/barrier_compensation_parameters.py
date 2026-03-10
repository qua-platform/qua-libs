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
    barrier_gate_span: float = 0.1
    """Total voltage span of the barrier gate (X axis) sweep in volts."""
    barrier_gate_points: int = 201
    """Number of points along the barrier gate (X axis) sweep."""
    compensation_gate_span: float = 0.1
    """Total voltage span of the compensation gate (Y axis) sweep in volts."""
    compensation_gate_points: int = 201
    """Number of points along the compensation gate (Y axis) sweep."""
    barrier_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    compensation_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""
