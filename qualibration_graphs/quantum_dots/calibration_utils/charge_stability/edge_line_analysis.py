"""
Line-segment fitting on an edge-detection map (charge stability diagrams).

Pipeline:
- threshold the edge map to a binary mask
- skeletonize to 1-pixel-wide traces
- build a pixel graph and extract branches between endpoints/junctions
- split branches at kinks via Ramer–Douglas–Peucker simplification
- fit each chunk with an orthogonal (total-least-squares) line
- compute intersections between non-parallel segments
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    from skimage.morphology import skeletonize
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError("scikit-image is required for skeletonization. Install with `pip install scikit-image`.") from exc


_NEIGHBORS: Tuple[Tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass
class SegmentFit:
    """Total-least-squares fit for one polyline chunk."""

    points: np.ndarray  # (N, 2) array of (row, col) pixels
    start: np.ndarray
    end: np.ndarray
    centroid: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    slope: float
    intercept: float
    proj_min: float
    proj_max: float


def threshold_edge_map(edge_map: np.ndarray, threshold: float) -> np.ndarray:
    """Convert a probability/intensity edge map to a binary mask."""
    return (edge_map >= threshold).astype(np.uint8)


def skeletonize_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Skeletonize a binary mask to 1-pixel-wide traces."""
    return skeletonize(binary_mask > 0).astype(np.uint8)


def _pixel_graph(
    skeleton: np.ndarray,
) -> Tuple[Dict[Tuple[int, int], List[Tuple[int, int]]], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Build adjacency graph from a skeleton.

    Returns:
        adjacency: dict mapping pixel -> list of neighbor pixels
        endpoints: degree-1 pixels
        junctions: degree>=3 pixels
    """
    coords = np.argwhere(skeleton > 0)
    pixel_set = {tuple(p) for p in coords}
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]] = defaultdict(list)
    degrees: Dict[Tuple[int, int], int] = {}

    for r, c in coords:
        nbrs = []
        for dr, dc in _NEIGHBORS:
            nbr = (r + dr, c + dc)
            if nbr in pixel_set:
                nbrs.append(nbr)
        adjacency[(r, c)] = nbrs
        degrees[(r, c)] = len(nbrs)

    endpoints = [p for p, deg in degrees.items() if deg == 1]
    junctions = [p for p, deg in degrees.items() if deg >= 3]
    return adjacency, endpoints, junctions


def _extract_branches(
    adjacency: Dict[Tuple[int, int], List[Tuple[int, int]]],
    endpoints: Sequence[Tuple[int, int]],
    junctions: Sequence[Tuple[int, int]],
) -> List[np.ndarray]:
    """Traverse the skeleton graph and return polylines between junctions/endpoints."""
    junction_set = set(junctions)
    starts = list(endpoints) + list(junctions)
    visited_edges = set()
    branches: List[np.ndarray] = []

    def _edge_key(a: Tuple[int, int], b: Tuple[int, int]) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return tuple(sorted([a, b]))

    for start in starts:
        for nbr in adjacency.get(start, []):
            key = _edge_key(start, nbr)
            if key in visited_edges:
                continue

            path = [start]
            prev = start
            cur = nbr
            visited_edges.add(key)

            while True:
                path.append(cur)
                neighbors = [n for n in adjacency[cur] if n != prev]

                # stop if we hit a junction/endpoint or dead-end
                if len(adjacency[cur]) != 2 or cur in junction_set or not neighbors:
                    break

                nxt = neighbors[0]
                visited_edges.add(_edge_key(cur, nxt))
                prev, cur = cur, nxt

            branches.append(np.array(path))

    return branches


def _rdp_indices(points: np.ndarray, epsilon: float) -> List[int]:
    """
    Ramer–Douglas–Peucker simplification returning indices of retained points.

    points: array of shape (N, 2) with ordered (row, col) pixels
    epsilon: max perpendicular distance before a kink is kept
    """

    def _perp_dist(pt: np.ndarray, start: np.ndarray, end: np.ndarray) -> float:
        line = end - start
        if np.allclose(line, 0):
            return float(np.linalg.norm(pt - start))
        return float(abs(np.cross(line, pt - start)) / (np.linalg.norm(line) + 1e-12))

    def _rdp(start_idx: int, end_idx: int) -> List[int]:
        if end_idx <= start_idx + 1:
            return [start_idx, end_idx]

        start = points[start_idx]
        end = points[end_idx]
        dists = np.array([_perp_dist(points[i], start, end) for i in range(start_idx + 1, end_idx)])

        if len(dists) == 0:
            return [start_idx, end_idx]

        max_idx_rel = int(np.argmax(dists))
        max_dist = dists[max_idx_rel]
        max_idx = start_idx + 1 + max_idx_rel

        if max_dist > epsilon:
            left = _rdp(start_idx, max_idx)
            right = _rdp(max_idx, end_idx)
            return left[:-1] + right
        return [start_idx, end_idx]

    keep = _rdp(0, len(points) - 1)
    # ensure monotonicity and uniqueness
    return sorted(dict.fromkeys(keep))


def _split_branch(branch: np.ndarray, epsilon: float, min_points: int) -> List[np.ndarray]:
    """Split a branch at kinks found by RDP and return the sub-polylines."""
    if len(branch) < max(3, min_points):
        return []

    idxs = _rdp_indices(branch, epsilon)
    if len(idxs) < 2:
        return []

    splits: List[np.ndarray] = []
    for a, b in zip(idxs[:-1], idxs[1:]):
        chunk = branch[a : b + 1]
        if len(chunk) >= min_points:
            splits.append(chunk)
    return splits


def _orthogonal_fit(points: np.ndarray) -> SegmentFit:
    """Total-least-squares fit to (row, col) points; returns segment endpoints."""
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

    return SegmentFit(
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


def _segment_intersection(
    seg_a: SegmentFit,
    seg_b: SegmentFit,
    *,
    parallel_tol: float = 1e-3,
    on_segment_tol: float = 2.0,
) -> Optional[np.ndarray]:
    """
    Compute intersection between two fitted lines and check it lies near both segments.

    Returns None for parallel lines or intersections outside the segment extents.
    """
    r = seg_a.direction
    s = seg_b.direction
    p = seg_a.centroid
    q = seg_b.centroid

    cross_rs = r[0] * s[1] - r[1] * s[0]
    if abs(cross_rs) < parallel_tol:
        return None

    qmp = q - p
    t = (qmp[0] * s[1] - qmp[1] * s[0]) / cross_rs
    u = (qmp[0] * r[1] - qmp[1] * r[0]) / cross_rs

    # ensure intersection is near the finite segment extents
    if (seg_a.proj_min - on_segment_tol) <= t <= (seg_a.proj_max + on_segment_tol) and (
        seg_b.proj_min - on_segment_tol
    ) <= u <= (seg_b.proj_max + on_segment_tol):
        return p + t * r
    return None


def analyze_edge_map(
    edge_map: np.ndarray,
    *,
    threshold: float = 0.25,
    rdp_epsilon: float = 2.5,
    min_branch_points: int = 5,
    min_segment_points: int = 4,
    parallel_tol: float = 1e-3,
    on_segment_tol: float = 2.5,
    base_image: Optional[np.ndarray] = None,
    show: bool = True,
) -> Dict[str, object]:
    """
    Full analysis pipeline from edge map to fitted lines and intersections.

    Args:
        edge_map: 2D array (e.g. mean_cp) containing edge probabilities.
        threshold: cutoff for binarizing the edge map.
        rdp_epsilon: kink sensitivity in pixels; lower keeps more bends.
        min_branch_points: discard very short skeleton branches.
        min_segment_points: minimum points per kink-defined chunk to fit a line.
        parallel_tol: determinant tolerance for treating two lines as parallel.
        on_segment_tol: allowable slack (pixels) when keeping intersections.
        base_image: optional background to show (e.g. raw sensor map).
        show: whether to display summary plots.

    Returns:
        dict with binary mask, skeleton, branches, segment fits, and intersections.
    """
    binary = threshold_edge_map(edge_map, threshold)
    skel = skeletonize_mask(binary)

    adjacency, endpoints, junctions = _pixel_graph(skel)
    branches = [b for b in _extract_branches(adjacency, endpoints, junctions) if len(b) >= min_branch_points]

    segment_points: List[np.ndarray] = []
    for branch in branches:
        segment_points.extend(_split_branch(branch, rdp_epsilon, min_segment_points))

    segments = [_orthogonal_fit(seg) for seg in segment_points if len(seg) >= min_segment_points]

    intersections: List[np.ndarray] = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            pt = _segment_intersection(
                segments[i], segments[j], parallel_tol=parallel_tol, on_segment_tol=on_segment_tol
            )
            if pt is not None:
                intersections.append(pt)

    if show:
        _plot_results(
            base_image if base_image is not None else edge_map,
            binary,
            skel,
            branches,
            segments,
            intersections,
        )

    return {
        "binary_mask": binary,
        "skeleton": skel,
        "branches": branches,
        "segments": segments,
        "intersections": intersections,
    }


def _plot_results(
    background: np.ndarray,
    binary: np.ndarray,
    skeleton: np.ndarray,
    branches: Sequence[np.ndarray],
    segments: Sequence[SegmentFit],
    intersections: Sequence[np.ndarray],
) -> None:
    """Plot background, skeleton, fitted segments, and intersections."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), constrained_layout=True)

    # background + binary overlay
    ax0 = axes[0]
    im0 = ax0.imshow(background, origin="lower", cmap="magma")
    fig.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04, label="sensor")
    ax0.imshow(np.ma.masked_where(binary == 0, binary), cmap="Reds", alpha=0.4, origin="lower")
    ax0.set_title("Edge threshold + skeleton")
    ax0.set_xlabel("col (V2)")
    ax0.set_ylabel("row (V1)")

    # skeleton + branches + fits
    ax1 = axes[1]
    ax1.imshow(background, origin="lower", cmap="gray", alpha=0.35)
    ax1.imshow(np.ma.masked_where(skeleton == 0, skeleton), cmap="Blues", alpha=0.5, origin="lower")

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(branches), 1)))
    for idx, branch in enumerate(branches):
        rc = branch[:, 0]
        cc = branch[:, 1]
        ax1.plot(cc, rc, ".", color=colors[idx % len(colors)], ms=2, alpha=0.8)

    for seg in segments:
        ax1.plot([seg.start[1], seg.end[1]], [seg.start[0], seg.end[0]], "-", color="orange", lw=2)

    if intersections:
        pts = np.vstack(intersections)
        ax1.scatter(pts[:, 1], pts[:, 0], marker="*", s=120, c="gold", edgecolor="k", zorder=5, label="triple points")

    ax1.set_title("Fitted segments + intersections")
    ax1.set_xlabel("col (V2)")
    ax1.set_ylabel("row (V1)")
    ax1.legend(loc="upper right")

    plt.show()
