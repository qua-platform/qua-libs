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

from dataclasses import dataclass
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


@dataclass
class _FallbackSegment:
    """Minimal segment representation compatible with plotting/extraction code."""

    points: np.ndarray
    start: np.ndarray
    end: np.ndarray
    centroid: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    slope: float
    intercept: float
    proj_min: float
    proj_max: float


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


def _orthogonal_fit_fallback(points: np.ndarray) -> _FallbackSegment:
    """Total-least-squares line fit used when scikit-image is unavailable."""
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    cov = centered.T @ centered / max(len(pts), 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    direction = eigvecs[:, int(np.argmax(eigvals))]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    normal = np.array([-direction[1], direction[0]])

    projections = centered @ direction
    proj_min = float(projections.min())
    proj_max = float(projections.max())

    start = centroid + proj_min * direction
    end = centroid + proj_max * direction

    slope = np.inf if abs(direction[0]) < 1e-9 else direction[1] / direction[0]
    intercept = np.nan if not np.isfinite(slope) else float(centroid[1] - slope * centroid[0])

    return _FallbackSegment(
        points=pts,
        start=start,
        end=end,
        centroid=centroid,
        direction=direction,
        normal=normal,
        slope=float(slope),
        intercept=intercept,
        proj_min=proj_min,
        proj_max=proj_max,
    )


def _extract_segments_fallback(
    mean_cp: np.ndarray,
    *,
    threshold: float,
    n_lines: int = 2,
    iterations: int = 300,
) -> List[_FallbackSegment]:
    """Estimate up to two dominant line segments via deterministic RANSAC."""
    def _angular_distance(a: float, b: float) -> float:
        return float(abs(((a - b + np.pi / 2) % np.pi) - np.pi / 2))

    binary = np.asarray(mean_cp >= threshold, dtype=bool)
    pts = np.argwhere(binary).astype(float)

    if len(pts) < 10:
        return []

    # Keep runtime stable on dense maps while preserving the dominant geometry.
    rng = np.random.default_rng(0)
    if len(pts) > 6000:
        pts = pts[rng.choice(len(pts), size=6000, replace=False)]

    remaining = np.ones(len(pts), dtype=bool)
    segments: List[_FallbackSegment] = []
    min_inliers = max(20, int(0.01 * len(pts)))
    dist_threshold = 1.8
    selected_angles: List[float] = []
    min_angle_sep = np.deg2rad(15.0)

    for line_idx in range(n_lines):
        candidates = pts[remaining]
        line_min_inliers = min_inliers if line_idx == 0 else max(10, int(0.003 * len(pts)))
        if len(candidates) < line_min_inliers:
            break

        best_inliers = None
        best_count = 0
        n = len(candidates)
        if n < 2:
            break

        for _ in range(iterations):
            i, j = rng.integers(0, n, size=2)
            if i == j:
                continue
            p1, p2 = candidates[i], candidates[j]
            d = p2 - p1
            norm = np.linalg.norm(d)
            if norm < 1e-9:
                continue
            d = d / norm
            theta = float(np.arctan2(d[1], d[0]) % np.pi)
            if selected_angles and any(_angular_distance(theta, t) < min_angle_sep for t in selected_angles):
                continue
            nvec = np.array([-d[1], d[0]])
            distances = np.abs((candidates - p1) @ nvec)
            inliers = distances <= dist_threshold
            count = int(np.count_nonzero(inliers))
            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < line_min_inliers:
            break

        seg_points = candidates[best_inliers]
        segment = _orthogonal_fit_fallback(seg_points)
        segments.append(segment)
        selected_angles.append(float(np.arctan2(segment.direction[1], segment.direction[0]) % np.pi))

        # Remove points near this line to reveal a second dominant family.
        rem_idx = np.where(remaining)[0]
        all_candidates = pts[rem_idx]
        d = segment.direction
        nvec = np.array([-d[1], d[0]])
        distances_all = np.abs((all_candidates - segment.centroid) @ nvec)
        keep_local = distances_all > (dist_threshold * 3.0)
        remaining[rem_idx[~keep_local]] = False

    return segments


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_virtual_plunger_coefficients(
    ds: xr.Dataset,
    plunger_gate_name: str,
    device_gate_name: str,
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

    Returns
    -------
    dict or None
        ``{"T_matrix": ndarray, "theta1": float, "theta2": float,
        "segments": list, "mean_cp": ndarray, "fit_params": dict}``
        when analysis succeeds, or ``None`` on failure.
    """
    data = ds[signal_var]
    if "sensors" in data.dims:
        data = data.isel(sensors=sensor_idx)

    amplitude_2d = data.values

    mean_cp, edge_base = _edge_detect(
        amplitude_2d, hazard=hazard, edge_threshold=edge_threshold,
    )

    if analyze_edge_map is None:
        segments = _extract_segments_fallback(np.array(mean_cp), threshold=edge_threshold)
    else:
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
            "plunger_gate_name": plunger_gate_name,
            "device_gate_name": device_gate_name,
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
        "mean_cp": mean_cp,
        "plunger_gate_name": plunger_gate_name,
        "device_gate_name": device_gate_name,
        "fit_params": {"success": True},
    }
