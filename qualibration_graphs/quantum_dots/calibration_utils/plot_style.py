"""Shared matplotlib styling for quantum-dots calibration plots."""

from __future__ import annotations

from typing import Optional

from matplotlib.axes import Axes


def node_success(ds_fit, entity_name: str) -> Optional[bool]:
    """Return per-entity fit success from ``ds_fit`` summary, if present."""
    if "success" not in ds_fit:
        return None
    for dim in ("sensor", "qubit"):
        if dim in ds_fit["success"].dims:
            try:
                return bool(ds_fit["success"].sel({dim: entity_name}).values)
            except Exception:
                continue
    return None


def apply_node_outcome_style(ax: Axes, entity_name: str, success: Optional[bool]) -> None:
    """Style a subplot title to reflect fit success or failure."""
    if success is False:
        ax.set_title(f"{entity_name} — FAILED", color="crimson", fontweight="bold")
    else:
        ax.set_title(entity_name)
