"""Plotting functions for gate virtualization calibration nodes."""

from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.figure import Figure


def plot_2d_scan(
    ds: xr.Dataset,
    x_axis: str = "x_volts",
    y_axis: str = "y_volts",
    sensor_name: Optional[str] = None,
    title: Optional[str] = None,
) -> Figure:
    """Plot a 2D voltage scan as a colour map.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing amplitude (or I/Q) data.
    x_axis, y_axis : str
        Coordinate names for the axes.
    sensor_name : str, optional
        If provided, select this sensor from the dataset.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement 2D colour-map plot
    fig, ax = plt.subplots()
    ax.set_title(title or "2D Scan")
    ax.set_xlabel(x_axis)
    ax.set_ylabel(y_axis)
    return fig


def plot_compensation_fit(
    ds: xr.Dataset,
    fit_results: Dict[str, Any],
    gate_x_name: str,
    gate_y_name: str,
    title: Optional[str] = None,
) -> Figure:
    """Overlay a linear/polynomial fit on a 2D scan.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset from the 2D scan.
    fit_results : dict
        Fit results from the analysis functions.
    gate_x_name, gate_y_name : str
        Gate names for axis labels.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement fit overlay plot
    fig, ax = plt.subplots()
    ax.set_title(title or "Compensation Fit")
    ax.set_xlabel(gate_x_name)
    ax.set_ylabel(gate_y_name)
    return fig


def plot_virtual_gate_matrix(
    matrix: np.ndarray,
    gate_names: List[str],
    title: Optional[str] = None,
) -> Figure:
    """Visualise the current virtual gate compensation matrix as a heatmap.

    Parameters
    ----------
    matrix : np.ndarray
        The compensation matrix.
    gate_names : list of str
        Gate labels for rows/columns.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # TODO: implement matrix heatmap
    fig, ax = plt.subplots()
    ax.set_title(title or "Virtual Gate Matrix")
    return fig
