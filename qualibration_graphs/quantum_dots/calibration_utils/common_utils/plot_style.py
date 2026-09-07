"""Shared matplotlib styling for quantum-dots calibration plots."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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


def qubit_success(fit_results: dict | None, qubit_name: str) -> Optional[bool]:
    """Return per-qubit fit success from ``fit_results``, if present."""
    if not fit_results or qubit_name not in fit_results:
        return None
    value = fit_results[qubit_name].get("success")
    if value is None:
        return None
    return bool(value)


def apply_qubit_outcome_style(
    ax: Axes,
    qubit_name: str,
    success: Optional[bool],
    subtitle: str = "",
) -> None:
    """Style a subplot title to reflect qubit fit success or failure."""
    base = f"{qubit_name}" + (f" — {subtitle}" if subtitle else "")
    if success is False:
        ax.set_title(f"{base} — FAILED", color="crimson", fontweight="bold")
    else:
        ax.set_title(base)


def empty_figure(message: str, *, figsize: tuple[float, float] = (8.0, 4.0)) -> Figure:
    """Return a figure with a centered message (e.g. when no qubit data is available)."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.text(
        0.5,
        0.5,
        message,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
    )
    return fig
