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
