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
4. Transformation matrix construction (Volk et al., Eq. 1):
   M = virtual-to-physical cross-talk matrix for the gate pair.
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
        import sys as _sys
        from importlib.util import spec_from_file_location, module_from_spec
        from pathlib import Path as _Path

        _ela_name = "_edge_line_analysis_direct"
        _ela_path = _Path(__file__).resolve().parent.parent / "charge_stability" / "edge_line_analysis.py"
        if not _ela_path.exists():
            raise ImportError(f"edge_line_analysis.py not found at {_ela_path}")
        _spec = spec_from_file_location(_ela_name, _ela_path)
        _mod = module_from_spec(_spec)
        _sys.modules[_ela_name] = _mod
        _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
        analyze_edge_map = _mod.analyze_edge_map  # type: ignore[no-redef]
        SegmentFit = _mod.SegmentFit  # type: ignore[no-redef,misc]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _edge_detect(
    amplitude_2d: np.ndarray,
    *,
    hazard: float = 1 / 200.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run BayesianCP edge detection on a 2D amplitude grid.

    Parameters
    ----------
    amplitude_2d : np.ndarray
        2D array of shape ``(ny, nx)``.
    hazard : float
        Hazard rate for BayesianCP.

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

    if len(primary_angles) < n_angles:
        min_sep = np.pi / max(n_bins, 1)

        def _angular_distance(a: float, b: float) -> float:
            # Angles are periodic in pi for line directions.
            return float(abs(((a - b + np.pi / 2) % np.pi) - np.pi / 2))

        for idx in np.argsort(weights_arr)[::-1]:
            candidate = float(angles_arr[idx])
            if all(_angular_distance(candidate, existing) > min_sep for existing in primary_angles):
                primary_angles.append(candidate)
            if len(primary_angles) >= n_angles:
                break

    return sorted(primary_angles[:n_angles])


def _build_transformation_matrix(
    theta1: float,
    theta2: float,
) -> np.ndarray:
    """Construct the virtual-to-physical cross-talk matrix for a gate pair.

    Given two charge-transition angles from a plunger–plunger scan,
    build the 2×2 matrix *M* whose columns are the physical-space
    directions of each virtual gate.  This is the incremental
    cross-capacitance matrix in the Volk et al. convention (Eq. 1,
    npj Quantum Information 5, 29, 2019): ``v_physical = M @ v_virtual``.

    Parameters
    ----------
    theta1, theta2 : float
        Primary charge-transition angles (radians), where
        ``theta = arctan2(Δv_x, Δv_y) mod π``.

    Returns
    -------
    np.ndarray
        2×2 matrix *M* (virtual → physical).

    Notes
    -----
    Each transition has slope ``m = cot(θ) = dv_y / dv_x``.  The
    steeper transition (|m| > 1, more vertical) is associated with
    the plunger-x gate, and the shallower one (|m| < 1) with
    plunger-y.  The matrix is::

        M = [[   1,   1/m_steep ],
             [ m_shallow,    1   ]]

    Column 0 = physical direction of virtual-gate-x (parallel to the
    *shallow* charge transition, so it doesn't cross dot-y's line).
    Column 1 = physical direction of virtual-gate-y (parallel to the
    *steep* charge transition).
    """
    m_a = np.cos(theta1) / np.sin(theta1)
    m_b = np.cos(theta2) / np.sin(theta2)

    if abs(m_a) >= abs(m_b):
        m_steep, m_shallow = m_a, m_b
    else:
        m_steep, m_shallow = m_b, m_a

    M = np.array(
        [
            [1.0, 1.0 / m_steep],
            [m_shallow, 1.0],
        ]
    )
    return M


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_single_angle_matrix(theta: float) -> np.ndarray:
    """Build a partial 2×2 M matrix from a single charge-transition angle.

    Used for asymmetric pairs (plunger vs sensor/barrier) where only one
    charge-transition direction is observed — typically a nearly-vertical
    line (θ ≈ 0 or θ ≈ π in our convention).

    Angle convention
    ~~~~~~~~~~~~~~~~
    ``θ = arctan2(d_col, d_row) % π``, so **vertical** lines in the plot
    (along the row/y axis) have θ ≈ 0 or θ ≈ π, and **horizontal** lines
    have θ ≈ π/2.

    The cross-talk coefficient is the column (plunger) shift per unit row
    (device-gate) change, i.e. ``α = d_col / d_row = sin(θ) / cos(θ) =
    tan(θ)``.  For a perfectly vertical line (θ → 0) this is ≈ 0 (no
    cross-talk), as expected.

    In the symmetric ``_build_transformation_matrix`` the steep-line
    entry is ``M[0,1] = 1/m_steep = 1/cot(θ) = tan(θ)``, consistent
    with this formula.

    Returns ``M = [[1, α], [0, 1]]``, so only M[0,1] is non-trivial.
    """
    cos_t = np.cos(theta)
    alpha = np.sin(theta) / cos_t if abs(cos_t) > 1e-12 else 0.0
    return np.array([[1.0, alpha], [0.0, 1.0]], dtype=float)


def extract_virtual_plunger_coefficients(
    ds: xr.Dataset,
    plunger_gate_name: str,
    device_gate_name: str,
    *,
    signal_var: str = "amplitude",
    sensor_idx: int = 0,
    edge_threshold: float = 0.25,
    hazard: float = 1 / 200.0,
    on_segment_tol: float = 2.5,
    asymmetric: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract virtual plunger gate coefficients from a 2D scan.

    For **symmetric** pairs (plunger–plunger), two charge-transition angles
    are required to build the full 2×2 cross-talk matrix.

    For **asymmetric** pairs (plunger–sensor, plunger–barrier), only one
    angle is needed.  The single transition gives the device-gate→plunger
    cross-talk coefficient.  Set ``asymmetric=True`` to accept a single
    angle as success.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset from the 2D scan (should contain *signal_var*).
    plunger_gate_name : str
        Name of the plunger gate on the X scan axis.
    device_gate_name : str
        Name of the device gate on the Y scan axis.
    signal_var : str
        Name of the data variable to analyse (default ``"amplitude"``).
    sensor_idx : int
        Index along the ``sensors`` dimension if present.
    edge_threshold : float
        Threshold for binarising the BayesianCP edge map.
    hazard : float
        Hazard rate for the BayesianCP model.
    on_segment_tol : float
        Pixel tolerance for accepting an intersection point as lying on
        both segments (passed to ``analyze_edge_map``).
    asymmetric : bool
        If True, accept a single detected angle and build a partial M
        matrix with only the device→plunger cross-talk entry.

    Returns
    -------
    dict or None
        ``{"T_matrix": ndarray (2×2, virtual→physical), "theta1": float,
        "theta2": float | None, "segments": list, "mean_cp": ndarray,
        "fit_params": dict}``
        when analysis succeeds, or ``None`` on failure.
    """
    data = ds[signal_var]
    if "sensors" in data.dims:
        data = data.isel(sensors=sensor_idx)

    amplitude_2d = data.values

    mean_cp, edge_base = _edge_detect(
        amplitude_2d,
        hazard=hazard,
    )

    edge_analysis = analyze_edge_map(
        np.array(mean_cp),
        threshold=edge_threshold,
        on_segment_tol=on_segment_tol,
        base_image=edge_base,
        show=False,
    )
    segments = edge_analysis["segments"]
    intersections = edge_analysis.get("intersections", [])
    if not segments:
        return {
            "T_matrix": None,
            "theta1": None,
            "theta2": None,
            "segments": [],
            "intersections": [],
            "mean_cp": mean_cp,
            "plunger_gate_name": plunger_gate_name,
            "device_gate_name": device_gate_name,
            "fit_params": {"success": False, "reason": "no segments detected"},
        }

    if asymmetric:
        # For asymmetric pairs (plunger vs barrier/sensor), we look for the
        # charge-transition cluster closest to vertical.  In our angle
        # convention θ = arctan2(d_col, d_row) % π, vertical lines have
        # θ ≈ 0 or θ ≈ π, so "distance from vertical" = min(θ, π − θ).
        # If only one cluster is found, use it directly.
        primary_angles = _extract_primary_angles(segments, n_angles=4)
        if not primary_angles:
            return {
                "T_matrix": None,
                "theta1": None,
                "theta2": None,
                "segments": segments,
                "intersections": intersections,
                "mean_cp": mean_cp,
                "plunger_gate_name": plunger_gate_name,
                "device_gate_name": device_gate_name,
                "fit_params": {"success": False, "reason": "no primary angles found"},
            }
        theta1 = min(primary_angles, key=lambda a: min(a, np.pi - a))
        T = _build_single_angle_matrix(theta1)
        return {
            "T_matrix": T,
            "theta1": theta1,
            "theta2": None,
            "segments": segments,
            "intersections": intersections,
            "mean_cp": mean_cp,
            "plunger_gate_name": plunger_gate_name,
            "device_gate_name": device_gate_name,
            "fit_params": {"success": True, "mode": "asymmetric"},
        }

    primary_angles = _extract_primary_angles(segments, n_angles=2)

    if len(primary_angles) < 2:
        return {
            "T_matrix": None,
            "theta1": primary_angles[0] if primary_angles else None,
            "theta2": None,
            "segments": segments,
            "intersections": intersections,
            "mean_cp": mean_cp,
            "plunger_gate_name": plunger_gate_name,
            "device_gate_name": device_gate_name,
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
        "intersections": intersections,
        "mean_cp": mean_cp,
        "plunger_gate_name": plunger_gate_name,
        "device_gate_name": device_gate_name,
        "fit_params": {"success": True},
    }
