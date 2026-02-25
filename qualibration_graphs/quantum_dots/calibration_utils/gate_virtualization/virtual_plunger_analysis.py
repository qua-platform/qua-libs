"""Analysis functions for virtual plunger calibration (node 02).

Detects charge transition lines in plunger-plunger 2D scans and fits their
slopes to determine the virtual gate transformation that decouples the dots.

Pipeline
--------
1. BayesianCP edge detection (row + column), averaged into a single edge map.
2. ``analyze_edge_map`` from the charge-stability utilities for
   skeletonisation, branch extraction, and total-least-squares line fitting.
3. Angle clustering: segment directions are binned into a weighted angular
   histogram; the two most prominent peaks give the primary charge-transition
   angles theta1 and theta2.
4. Transformation matrix construction (paper Eqs. 23-25):
       R = rotation(theta1)
       S = shear(theta2 - theta1)
       T = S @ R
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

import jax
import jax.numpy as jnp

try:
    from ..bayesian_change_point import BayesianCP
except ImportError:
    from calibration_utils.bayesian_change_point import BayesianCP  # type: ignore[no-redef]

try:
    from ..charge_stability.edge_line_analysis import SegmentFit, analyze_edge_map
except (ImportError, SystemError):
    try:
        from calibration_utils.charge_stability.edge_line_analysis import (  # type: ignore[no-redef]
            SegmentFit,
            analyze_edge_map,
        )
    except ImportError:
        analyze_edge_map = None  # type: ignore[assignment]
        SegmentFit = None  # type: ignore[assignment,misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _edge_detect(
    amplitude_2d: np.ndarray,
    *,
    hazard: float = 1 / 200.0,
    edge_threshold: float = 0.25,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run BayesianCP edge detection on a 2D amplitude grid.

    Replicates the logic from ``charge_stability.analysis.fit_individual_raw_data``
    but decoupled from the node / sensor objects.

    Parameters
    ----------
    amplitude_2d : np.ndarray
        2D array of shape ``(ny, nx)``.
    hazard : float
        Hazard rate for BayesianCP.
    edge_threshold : float
        Threshold for binarising the edge probability map.

    Returns
    -------
    mean_cp : np.ndarray
        Averaged edge probability map (row + column BayesianCP).
    edge_base : np.ndarray
        The amplitude sub-array aligned to *mean_cp* (for plotting).
    """
    zs = np.asarray(amplitude_2d, dtype=float)
    model = BayesianCP(hazard=hazard, standardize=True)

    cp, _ = jax.vmap(model.fit)(jnp.asarray(zs))
    cp2, _ = jax.vmap(model.fit)(jnp.asarray(zs.T))

    cp = np.asarray(cp)
    cp2 = np.asarray(cp2)

    mean_cp = (cp[1:] + cp2[1:].T) / 2.0
    edge_base = zs[1:, 1:] if zs.shape[0] > 1 and zs.shape[1] > 1 else zs
    return mean_cp, edge_base


def _extract_primary_angles(
    segments: Sequence[Any],
    n_angles: int = 2,
    n_bins: int = 36,
) -> List[float]:
    """Cluster segment directions into *n_angles* primary groups.

    Each segment's angle is weighted by the number of pixels it contains so
    that longer, more reliable segments have greater influence.

    Parameters
    ----------
    segments : sequence of SegmentFit
        Line segments from ``analyze_edge_map``.
    n_angles : int
        Number of dominant angles to return (default 2).
    n_bins : int
        Number of histogram bins over [0, pi).  Default 36 gives 5° bins.

    Returns
    -------
    list of float
        The *n_angles* most prominent angles in **radians**, sorted ascending.
    """
    if not segments:
        return []

    angles = []
    weights = []
    for seg in segments:
        d = np.asarray(seg.direction, dtype=float)
        theta = np.arctan2(d[1], d[0]) % np.pi
        angles.append(theta)
        weights.append(len(seg.points))

    angles_arr = np.array(angles)
    weights_arr = np.array(weights, dtype=float)

    bin_edges = np.linspace(0, np.pi, n_bins + 1)
    hist, _ = np.histogram(angles_arr, bins=bin_edges, weights=weights_arr)

    kernel_width = max(3, n_bins // 12)
    kernel = np.ones(kernel_width) / kernel_width
    hist_smooth = np.convolve(hist, kernel, mode="same")

    # Refined peak centres: compute a weighted average within each peak's bin
    # neighbourhood rather than using the bin centre directly.
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    peak_indices = []
    for i in range(1, len(hist_smooth) - 1):
        if hist_smooth[i] >= hist_smooth[i - 1] and hist_smooth[i] > hist_smooth[i + 1]:
            peak_indices.append(i)
    # Also check first and last bins
    if len(hist_smooth) >= 2:
        if hist_smooth[0] > hist_smooth[1]:
            peak_indices.append(0)
        if hist_smooth[-1] > hist_smooth[-2]:
            peak_indices.append(len(hist_smooth) - 1)

    if not peak_indices and len(hist_smooth) > 0:
        peak_indices = [int(np.argmax(hist_smooth))]

    peak_indices = list(dict.fromkeys(peak_indices))
    peak_indices.sort(key=lambda i: hist_smooth[i], reverse=True)
    selected = peak_indices[:n_angles]

    # Refine each peak angle using weighted mean of segments in that bin range
    primary_angles = []
    half_bin = (bin_edges[1] - bin_edges[0]) * 1.5
    for idx in selected:
        centre = bin_centers[idx]
        mask = np.abs(angles_arr - centre) < half_bin
        if np.any(mask):
            refined = np.average(angles_arr[mask], weights=weights_arr[mask])
        else:
            refined = centre
        primary_angles.append(float(refined))

    return sorted(primary_angles)


def _build_transformation_matrix(
    theta1: float,
    theta2: float,
) -> np.ndarray:
    """Construct the virtual gate cross-talk correction matrix.

    Given two charge-transition angles, build a matrix *M* that maps
    virtual gate voltages to physical plunger voltages such that each
    virtual gate controls only one dot.  Returns ``T = M^{-1}``
    (physical-to-virtual), which is close to the identity when
    cross-talk is moderate.

    Parameters
    ----------
    theta1, theta2 : float
        Primary charge-transition angles (radians), where
        ``theta = arctan2(Δv_x, Δv_y) mod π``.

    Returns
    -------
    np.ndarray
        2×2 matrix *T* mapping physical plunger voltages to virtual
        gate voltages: ``v_virtual = T @ v_physical``.

    Notes
    -----
    Each transition has slope ``m = cot(θ) = dv_y / dv_x``.  The
    steeper transition (|m| > 1, more vertical) is associated with
    the plunger-x gate, and the shallower one (|m| < 1) with
    plunger-y.  The virtual-to-physical matrix is::

        M = [[   1,   1/m_steep ],
             [ m_shallow,    1   ]]

    whose columns are the physical-space directions of each virtual
    gate, normalised so the diagonal is [1, 1].
    """
    m_a = np.cos(theta1) / np.sin(theta1)
    m_b = np.cos(theta2) / np.sin(theta2)

    if abs(m_a) >= abs(m_b):
        m_steep, m_shallow = m_a, m_b
    else:
        m_steep, m_shallow = m_b, m_a

    M = np.array([
        [1.0, 1.0 / m_steep],
        [m_shallow, 1.0],
    ])
    return np.linalg.inv(M)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_virtual_plunger_coefficients(
    ds: xr.Dataset,
    plunger_x_name: str,
    plunger_y_name: str,
    *,
    signal_var: str = "amplitude",
    sensor_idx: int = 0,
    edge_threshold: float = 0.25,
    hazard: float = 1 / 200.0,
) -> Optional[Dict[str, Any]]:
    """Extract virtual plunger gate coefficients from a plunger-plunger scan.

    From a 2D charge-stability map of two plunger gates, detect the charge
    transition line slopes and construct the virtual gate transformation
    matrix T that decouples the two dots.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset from the 2D scan (should contain *signal_var*).
    plunger_x_name : str
        Coordinate name for the X plunger gate axis.
    plunger_y_name : str
        Coordinate name for the Y plunger gate axis.
    signal_var : str
        Name of the data variable to analyse (default ``"amplitude"``).
    sensor_idx : int
        Index along the ``sensors`` dimension if present.
    edge_threshold : float
        Threshold for binarising the BayesianCP edge map.
    hazard : float
        Hazard rate for the BayesianCP model.

    Returns
    -------
    dict or None
        ``{"T_matrix": ndarray, "theta1": float, "theta2": float,
        "segments": list, "mean_cp": ndarray, "fit_params": dict}``
        when analysis succeeds, or ``None`` on failure.
    """
    if analyze_edge_map is None:
        raise ImportError(
            "Line fitting requires scikit-image; install it to enable edge analysis."
        )

    data = ds[signal_var]
    if "sensors" in data.dims:
        data = data.isel(sensors=sensor_idx)

    amplitude_2d = data.values

    mean_cp, edge_base = _edge_detect(
        amplitude_2d, hazard=hazard, edge_threshold=edge_threshold,
    )

    edge_analysis = analyze_edge_map(
        np.array(mean_cp),
        threshold=edge_threshold,
        base_image=edge_base,
        show=False,
    )

    segments = edge_analysis["segments"]
    if not segments:
        return {
            "T_matrix": None,
            "theta1": None,
            "theta2": None,
            "segments": [],
            "mean_cp": mean_cp,
            "fit_params": {"success": False, "reason": "no segments detected"},
        }

    primary_angles = _extract_primary_angles(segments, n_angles=2)

    if len(primary_angles) < 2:
        return {
            "T_matrix": None,
            "theta1": primary_angles[0] if primary_angles else None,
            "theta2": None,
            "segments": segments,
            "mean_cp": mean_cp,
            "fit_params": {
                "success": False,
                "reason": f"only {len(primary_angles)} primary angle(s) found",
            },
        }

    theta1, theta2 = primary_angles[0], primary_angles[1]
    T = _build_transformation_matrix(theta1, theta2)

    return {
        "T_matrix": T,
        "theta1": theta1,
        "theta2": theta2,
        "segments": segments,
        "mean_cp": mean_cp,
        "fit_params": {"success": True},
    }
