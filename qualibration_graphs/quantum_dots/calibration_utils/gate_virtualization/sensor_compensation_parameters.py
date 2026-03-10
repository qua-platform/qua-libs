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
    sensor_gate_span: float = 0.05
    """Total voltage span of the sensor gate (X axis) sweep in volts."""
    sensor_gate_points: int = 201
    """Number of points along the sensor gate (X axis) sweep."""
    device_gate_span: float = 0.1
    """Total voltage span of the device gate (Y axis) sweep in volts."""
    device_gate_points: int = 201
    """Number of points along the device gate (Y axis) sweep."""
    sensor_gate_from_qdac: bool = False
    """Whether to perform the X axis sweep using the QDAC instead of the OPX."""
    device_gate_from_qdac: bool = False
    """Whether to perform the Y axis sweep using the QDAC instead of the OPX."""
