"""Shared matplotlib styling for quantum-dots calibration plots."""

from __future__ import annotations

from typing import Optional

from matplotlib.axes import Axes


def sensor_success(ds_fit, sensor_name: str) -> Optional[bool]:
    """Return per-sensor fit success from ``ds_fit`` summary coords, if present."""
    if "success" not in ds_fit.coords:
        return None
    try:
        return bool(ds_fit.sel(sensor=sensor_name).success.values)
    except Exception:
        return None


def apply_sensor_outcome_style(ax: Axes, sensor_name: str, success: Optional[bool]) -> None:
    """Style a subplot title to reflect fit success or failure."""
    if success is False:
        ax.set_title(f"{sensor_name} — FAILED", color="crimson", fontweight="bold")
    else:
        ax.set_title(sensor_name)
