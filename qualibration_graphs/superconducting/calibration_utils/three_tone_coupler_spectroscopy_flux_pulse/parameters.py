"""Parameters for three-tone coupler spectroscopy with a coupler flux pulse (22a)."""

from typing import Callable, ClassVar, Literal, Optional, Sequence

import numpy as np
from qualibrate import NodeParameters
from qualibrate.core.parameters import RunnableParameters
from qualibration_libs.parameters import CommonNodeParameters, QubitPairExperimentNodeParameters


class NodeSpecificParameters(RunnableParameters):
    """Three-tone spectroscopy at fixed coupler flux pulse amplitude."""

    num_shots: int = 100
    """Number of averages per frequency point."""
    control_drive_operation: str = "x180"
    """Control-qubit operation used to drive the coupler."""
    control_pulse_duration_in_ns: int = 500
    """Duration of the control drive pulse in ns."""
    control_pulse_amplitude: float = 0.1
    """Amplitude scale for the control drive pulse."""
    target_drive_operation: str = "saturation"
    """Weak probe operation on the target qubit."""
    target_pulse_amplitude: float = 0.02
    """Amplitude scale for the target probe pulse."""
    target_pulse_duration_in_ns: Optional[int] = 400
    """Target probe duration in ns; default uses the operation length from state."""
    frequency_span_in_mhz: float = 800.0
    """Total frequency span of the coupler drive sweep in MHz."""
    frequency_step_in_mhz: float = 1.0
    """Frequency step in MHz."""
    coupler_flux_in_v: float = 0.02
    """Coupler flux pulse amplitude in V (played relative to decouple_offset)."""
    coupler_flux_settle_in_ns: int = 25
    """Wait after the coupler flux pulse before the control drive, in ns."""
    coupler_band: Literal["above", "below"] = "above"
    """Where the coupler sits relative to the qubit pair (used only for the RF guess).

    ``above``: ``max(f) + min(f) / 2``. ``below``: ``min(f) - max(f) / 2``.
    Ignored when ``rf_frequency_startpoint_in_hz`` or ``coupler.RF_frequency`` is set.
    """
    rf_frequency_startpoint_in_hz: Optional[float] = None
    """Optional coupler RF sweep centre in Hz for all pairs.

    When ``None`` (default), each pair uses ``coupler.RF_frequency`` from state if set,
    otherwise an estimate from the qubit XY frequencies and ``coupler_band``.
    """
    update_state: bool = False
    """Write fitted ``coupler.RF_frequency`` into state when analysis succeeds."""


class Parameters(
    NodeParameters,
    CommonNodeParameters,
    NodeSpecificParameters,
    QubitPairExperimentNodeParameters,
):
    """Combined parameters for 22a three-tone coupler spectroscopy (flux pulse)."""

    targets_name: ClassVar[str] = "qubit_pairs"


LogCallable = Callable[[str], None]
CouplerBand = Literal["above", "below"]


def estimate_coupler_rf_from_qubits(f_control: float, f_target: float, coupler_band: CouplerBand) -> float:
    """Guess coupler RF from the two qubit XY frequencies.

    ``above``: ``max + min / 2``. ``below``: ``min - max / 2``.
    """
    f_high = max(f_control, f_target)
    f_low = min(f_control, f_target)
    if coupler_band == "below":
        return f_low - f_high / 2.0
    return f_high + f_low / 2.0


def resolve_coupler_rf_centers_by_pair(
    qubit_pairs: Sequence,
    rf_override_hz: Optional[float] = None,
    *,
    coupler_band: CouplerBand = "above",
    log_callable: Optional[LogCallable] = None,
) -> dict[str, float]:
    """Resolve per-pair coupler RF sweep centres.

    Uses ``rf_override_hz`` if set, else ``coupler.RF_frequency`` from state,
    else :func:`estimate_coupler_rf_from_qubits` with ``coupler_band``.
    """
    centers: dict[str, float] = {}
    for qp in qubit_pairs:
        if rf_override_hz is not None:
            centers[qp.name] = float(rf_override_hz)
            continue

        coupler_rf = getattr(getattr(qp, "coupler", None), "RF_frequency", None)
        if coupler_rf is not None and np.isfinite(coupler_rf) and coupler_rf > 1e9:
            centers[qp.name] = float(coupler_rf)
            continue

        f_control = float(qp.qubit_control.xy.RF_frequency)
        f_target = float(qp.qubit_target.xy.RF_frequency)
        estimate_hz = estimate_coupler_rf_from_qubits(f_control, f_target, coupler_band)
        if log_callable is not None:
            log_callable(
                f"{qp.name}: no coupler.RF_frequency in state; "
                f"estimating sweep centre {estimate_hz * 1e-9:.4f} GHz ({coupler_band}) from "
                f"qubit XY ({f_control * 1e-9:.4f}, {f_target * 1e-9:.4f} GHz)"
            )
        centers[qp.name] = estimate_hz
    return centers
