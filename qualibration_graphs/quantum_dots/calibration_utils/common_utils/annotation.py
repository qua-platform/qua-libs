"""Utilities for annotating matplotlib figures with qualibrate metadata."""

from __future__ import annotations

from typing import Any, Dict, Optional

from matplotlib.figure import Figure


def stamp_snapshot(
    fig: Figure,
    snapshot_idx: Optional[int],
    node_name: str = "",
) -> None:
    """Add a small italic snapshot label to the bottom-right of *fig*.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to annotate.
    snapshot_idx : int or None
        The qualibrate snapshot index.  If ``None`` the call is a no-op.
    node_name : str, optional
        Logical node name to include alongside the index.
    """
    if snapshot_idx is None:
        return
    label = f"#{snapshot_idx}"
    if node_name:
        label = f"#{snapshot_idx} — {node_name}"
    fig.text(
        0.99,
        0.01,
        label,
        fontsize=8,
        fontstyle="italic",
        ha="right",
        va="bottom",
        alpha=0.5,
        transform=fig.transFigure,
    )


def annotate_node_figures(node: Any) -> None:
    """Stamp every figure stored in *node.results* with the snapshot index.

    Handles the common storage patterns:

    * ``node.results["figures"]``  – ``Dict[str, Figure]``
    * ``node.results["figures"]``  – nested ``Dict[str, Dict[str, Figure]]``
    * ``node.results["figure"]``   – single ``Figure``
    * ``node.results["fig_<name>"]`` – per-target figures
    """
    snapshot_idx = getattr(node, "snapshot_idx", None)
    if snapshot_idx is None:
        return
    node_name = getattr(node, "name", "")
    results: dict = getattr(node, "results", None) or {}

    single_fig = results.get("figure")
    if isinstance(single_fig, Figure):
        stamp_snapshot(single_fig, snapshot_idx, node_name)

    figures = results.get("figures")
    if isinstance(figures, dict):
        _stamp_dict(figures, snapshot_idx, node_name)

    for key, val in results.items():
        if key.startswith("fig_") and isinstance(val, Figure):
            stamp_snapshot(val, snapshot_idx, node_name)


def _stamp_dict(
    d: Dict[str, Any],
    snapshot_idx: int,
    node_name: str,
) -> None:
    for val in d.values():
        if isinstance(val, Figure):
            stamp_snapshot(val, snapshot_idx, node_name)
        elif isinstance(val, dict):
            _stamp_dict(val, snapshot_idx, node_name)
