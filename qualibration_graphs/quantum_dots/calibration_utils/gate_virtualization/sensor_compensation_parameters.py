from typing import Dict, List, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class SensorCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for sensor gate vs device gate compensation scans."""

    sensor_device_mapping: Optional[Dict[str, List[str]]] = None
    """Mapping of sensor gate -> list of device gates to scan against it.
    Only local cross-talk pairs need to be specified.
    Example: {"virtual_sensor_1": ["virtual_dot_1", "virtual_dot_2"]}.
    If None, must be generated from the machine (not yet implemented)."""
