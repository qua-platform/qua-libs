from typing import Dict, List, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class VirtualPlungerParameters(GateVirtualizationBaseParameters):
    """Parameters for virtual plunger gate calibration."""

    plunger_device_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of plunger gate -> list of device gates (plungers or barriers)
    to scan against it.  Only neighbouring pairs need to be specified.
    Example: {"virtual_dot_1": ["virtual_dot_2", "barrier_12"]}.
    If None, must be generated from the machine (not yet implemented)."""
    plunger_gate_span: float = 0.1
    """Total voltage span of the plunger gate (X axis) sweep in volts."""
    plunger_gate_points: int = 201
    """Number of points along the plunger gate (X axis) sweep."""
    device_gate_span: float = 0.1
    """Total voltage span of the device gate (Y axis) sweep in volts."""
    device_gate_points: int = 201
    """Number of points along the device gate (Y axis) sweep."""
    plunger_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    device_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""
