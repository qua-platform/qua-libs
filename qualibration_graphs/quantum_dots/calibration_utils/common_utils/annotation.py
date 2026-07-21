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
    _auto_add_heralded_n_loops_figure(node)

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


def _auto_add_heralded_n_loops_figure(node: Any) -> None:
    """Auto-add heralded n_loops figures when dataset/params support it."""
    params = getattr(node, "parameters", None)
    if not bool(getattr(params, "return_n_loops", False)):
        return

    results: dict = getattr(node, "results", None) or {}
    ds_raw = results.get("ds_raw")
    if ds_raw is None:
        return

    n_loop_vars = [name for name in getattr(ds_raw, "data_vars", {}) if str(name).startswith("n_loops_")]
    if not n_loop_vars:
        return

    figures = results.setdefault("figures", {})
    if "n_loops" in figures:
        return

    first_da = ds_raw[n_loop_vars[0]]
    candidate_item_dims = ("qubit", "qubit_pair", "quantum_dot_pair")
    item_dim = next((d for d in candidate_item_dims if d in first_da.dims), None)
    if item_dim is None and first_da.dims:
        item_dim = first_da.dims[0]
    if item_dim is None or item_dim not in ds_raw.coords:
        return

    item_names = [str(v) for v in ds_raw.coords[item_dim].values]
    sweep_dims = [d for d in first_da.dims if d != item_dim]
    if not sweep_dims:
        return

    from calibration_utils.common_utils.experiment import (  # local import to avoid cycles
        plot_heralded_n_loops,
        plot_heralded_n_loops_2d,
    )

    if len(sweep_dims) == 1:
        sweep_key = sweep_dims[0]
        fig = plot_heralded_n_loops(
            ds_raw,
            item_names,
            item_dim=item_dim,
            sweep_key=sweep_key,
            sweep_scale=1.0,
            sweep_xlabel=sweep_key,
        )
        if fig is not None:
            figures["n_loops"] = fig
        return

    x_key = sweep_dims[-1]
    y_key = sweep_dims[-2]
    fig = plot_heralded_n_loops_2d(
        ds_raw,
        item_names,
        item_dim=item_dim,
        x_sweep_key=x_key,
        y_sweep_key=y_key,
        x_sweep_scale=1.0,
        y_sweep_scale=1.0,
        x_sweep_xlabel=x_key,
        y_sweep_ylabel=y_key,
    )
    if fig is not None:
        figures["n_loops"] = fig
