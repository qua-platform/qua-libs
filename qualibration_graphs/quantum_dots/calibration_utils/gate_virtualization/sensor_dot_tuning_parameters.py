from typing import List, Literal, Optional

from calibration_utils.gate_virtualization.base_parameters import (
    GateVirtualizationBaseParameters,
)


class SensorDotTuningParameters(GateVirtualizationBaseParameters):
    """Parameters for 1D sensor Coulomb peak sweep and operating-point calibration."""

    sensor_gate_names: Optional[List[str]] = None
    """List of sensor gate names to sweep.  If None, all sensor dots are used."""
    sweep_span: float = 0.02
    """Voltage span of the 1D sweep (V)."""
    sweep_points: int = 201
    """Number of points in the 1D sweep."""
    sweep_center: Optional[float] = None
    """Centre of the sweep (V).  When None and sweeping via QDAC, the current
    DAC voltage is read automatically."""
    from_qdac: bool = False
    """Whether the sensor gate is driven by the QDAC."""
    operating_side: Literal["left", "right"] = "right"
    """Which side of the Lorentzian peak to place the operating point.
    'right' selects x0 + γ/(2√3), 'left' selects x0 − γ/(2√3)."""
