"""PAT and barrier-compensation parameter models.

This module intentionally isolates parameters for:
- PAT lever-arm calibration node
- Barrier-barrier compensation node
"""

from typing import Dict, List, Optional

from .parameters import GateVirtualizationBaseParameters


class PATLeverArmParameters(GateVirtualizationBaseParameters):
    """Parameters for PAT-based inter-dot lever-arm calibration."""

    barrier_dot_pair_mapping: Optional[Dict[str, str]] = None
    """Mapping of target barrier -> quantum-dot pair identifier.
    Used to expand dot-pair lever arms into barrier-keyed lever arms."""
    pat_lever_arm_mapping: Optional[Dict[str, float]] = None
    """Dot-pair lever-arm map from PAT, keyed by quantum-dot pair name.
    Example: {"qd_pair_1_2": 175.0}."""
    default_lever_arm: float = 1.0
    """Fallback lever arm when PAT data is unavailable."""


class BarrierCompensationParameters(GateVirtualizationBaseParameters):
    """Parameters for barrier-barrier compensation scans."""

    pair_names: Optional[List[str]] = None
    """Ordered list of calibration targets.

    Each name can be either:
    - a key in ``machine.quantum_dot_pairs``; or
    - a key in ``machine.qubit_pairs`` (resolved through ``qubit_pair.quantum_dot_pair``).

    Row/calibration order follows this list.
    """
    calibration_order: Optional[List[str]] = None
    """Optional explicit barrier calibration order.
    If not provided, order is derived from ``pair_names``."""
    barrier_center_overrides: Optional[Dict[str, float]] = None
    """Optional center voltages for barriers (V), keyed by barrier name."""
    detuning_center_overrides: Optional[Dict[str, float]] = None
    """Optional center voltages for detuning axes (V), keyed by target pair id."""
    slope_sweep_span_mv: float = 20.0
    """Total drive-barrier sweep span for local slope extraction (mV)."""
    slope_sweep_points: int = 7
    """Number of drive-barrier points per local slope fit."""
    detuning_min: float = -0.1
    """Minimum detuning value for per-point tunnel-coupling extraction (V)."""
    detuning_max: float = 0.1
    """Maximum detuning value for per-point tunnel-coupling extraction (V)."""
    detuning_points: int = 121
    """Number of detuning points used in tunnel-coupling extraction."""
    residual_crosstalk_target: float = 0.10
    """Target maximum residual off-diagonal crosstalk ratio."""
    max_refinement_rounds: int = 2
    """Maximum number of refinement rounds in the stepwise ``B* -> B†`` flow."""
    matrix_layer_id: Optional[str] = None
    """Optional VirtualizationLayer id to update.
    If None, the last layer of the selected VirtualGateSet is updated."""
    target_tunnel_couplings: Optional[Dict[str, float]] = None
    """Optional target tunnel couplings keyed by target barrier name.
    If provided, metadata is stored for future retuning hooks."""
    min_abs_self_slope: float = 1e-12
    """Minimum absolute value for ``dt_i/dB_i`` to accept a calibration row."""
    min_slope_snr: float = 5.0
    """Minimum slope signal-to-noise ratio ``|slope| / slope_stderr``."""
    min_tunnel_span_sigma: float = 3.0
    """Minimum tunnel span in units of median per-point ``t`` uncertainty."""
    min_pair_fit_r2: float = 0.9
    """Minimum linear-fit R² for ``t`` vs drive slope acceptance."""
    pat_lever_arm_mapping: Optional[Dict[str, float]] = None
    """Optional PAT lever-arm map keyed by quantum-dot pair name.
    Used to scale detuning-axis values for tunnel-coupling extraction."""
    barrier_lever_arm_mapping: Optional[Dict[str, float]] = None
    """Optional lever-arm map keyed by target barrier name.
    Takes precedence over ``pat_lever_arm_mapping`` when both are set."""
    default_lever_arm: float = 1.0
    """Fallback detuning scale when no lever-arm calibration is available."""
