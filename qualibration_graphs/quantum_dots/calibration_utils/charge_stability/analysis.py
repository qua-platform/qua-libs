import logging
from dataclasses import dataclass
from typing import Tuple, Dict, List, Any, Optional
import numpy as np
import xarray as xr
import jax
import jax.numpy as jnp
from jax import lax

from qualibrate import QualibrationNode

from bayesian_change_point.bayesian_cp import BayesianCP

try:
    from .edge_line_analysis import analyze_edge_map, SegmentFit

    _edge_line_import_error: Optional[Exception] = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    analyze_edge_map = None
    SegmentFit = None
    _edge_line_import_error = exc


@dataclass
class FitParameters:
    """Stores the relevant charge stability experiment fit parameters for a single sensor"""

    cp: np.ndarray
    cp2: np.ndarray
    mean_cp: np.ndarray
    edge_binary: Optional[np.ndarray] = None
    skeleton: Optional[np.ndarray] = None
    segments: Optional[List[Any]] = None
    intersections: Optional[np.ndarray] = None
    edge_threshold: float = 0.25
    success: bool = False

    def to_dict(self):
        """Convert FitParameters to a JSON-serializable dictionary."""

        def serialize_segment(seg: Any):
            """Convert SegmentFit or dict to serializable dict."""
            if isinstance(seg, dict):
                return seg
            if SegmentFit is not None and isinstance(seg, SegmentFit):
                return {
                    "start": np.asarray(seg.start).tolist(),
                    "end": np.asarray(seg.end).tolist(),
                    "centroid": np.asarray(seg.centroid).tolist(),
                    "direction": np.asarray(seg.direction).tolist(),
                    "slope": seg.slope,
                    "intercept": seg.intercept,
                }
            return {}

        return {
            "cp": np.asarray(self.cp).tolist() if self.cp is not None else [],
            "cp2": np.asarray(self.cp2).tolist() if self.cp2 is not None else [],
            "mean_cp": np.asarray(self.mean_cp).tolist() if self.mean_cp is not None else [],
            "edge_binary": np.asarray(self.edge_binary).tolist() if self.edge_binary is not None else [],
            "skeleton": np.asarray(self.skeleton).tolist() if self.skeleton is not None else [],
            "segments": [serialize_segment(s) for s in (self.segments or [])],
            "intersections": np.asarray(self.intersections).tolist() if self.intersections is not None else [],
            "edge_threshold": float(self.edge_threshold),
            "success": self.success,
        }


def peak_mask(cp, window=5, threshold=0.0):
    """
    Args:
      cp: 1D array of shape [T] (posterior at each boundary)
      window: odd int >= 3, peak must be the max in this window
      threshold: keep peaks >= this value
    Returns:
      mask: boolean array [T] where True marks a peak
    """
    x = jnp.asarray(cp)
    w = int(window)
    assert w >= 3 and w % 2 == 1
    pad = w // 2

    # local max within window
    local_max = lax.reduce_window(
        x, -jnp.inf, lax.max, window_dimensions=(w,), window_strides=(1,), padding=((pad, pad),)  # "same"
    )

    # mark points that are the unique (or leftmost in a tie) local max and over threshold
    is_local_max = (x >= threshold) & (x == local_max)

    # break plateaus: keep only where slope from left is positive
    left_pos_slope = jnp.concatenate([jnp.array([False]), x[1:] > x[:-1]])
    mask = is_local_max & left_pos_slope

    # also guard the right edge plateau case
    mask = mask.at[-1].set(is_local_max[-1] & (x[-1] > x[-2]))

    return mask


def log_fitted_results(fit_results: Dict, log_callable=None):
    """
    Logs the node-specific fitted results for all sensors from the fit results

    Parameters:
    -----------
    fit_results : dict
        Dictionary containing the fitted results for all sensors.
    log_callable : callable, optional
        Callable for logging the fitted results. If None, a default logger is used.

    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info
    for q in fit_results.keys():
        s_sensor = f"Results for sensor {q}: "
        num_segments = f"\tLine segments fitted: {len(fit_results[q].get('segments', []))}\n"
        num_intersections = f"\tIntersections found: {len(fit_results[q].get('intersections', []))}\n"
        if fit_results[q]["success"]:
            s_sensor += " SUCCESS!\n"
        else:
            s_sensor += " FAIL!\n"
        log_callable(s_sensor + num_segments + num_intersections)


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode):
    """
    Process the raw charge stability dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw I and Q quadrature data.
    node : QualibrationNode
        The calibration node containing parameters.

    Returns:
    --------
    xr.Dataset
        Processed dataset with amplitude added.
    """
    # Compute amplitude from I and Q
    amplitude = np.sqrt(ds.I**2 + ds.Q**2)
    ds = ds.assign({"amplitude": amplitude})
    ds.amplitude.attrs = {"long_name": "IQ amplitude", "units": "V"}

    # Compute phase from I and Q
    phase = np.arctan2(ds.Q, ds.I)
    ds = ds.assign({"phase": phase})
    ds.phase.attrs = {"long_name": "IQ phase", "units": "rad"}

    return ds


def fit_raw_data(ds: xr.Dataset, node: QualibrationNode) -> Tuple[xr.Dataset, dict[str, FitParameters]]:
    """
    Perform charge stability analysis for each sensor in the dataset.

    Parameters:
    -----------
    ds : xr.Dataset
        Dataset containing the raw data.
    node : QualibrationNode
        The calibration node containing parameters and sensors.

    Returns:
    --------
    Tuple[xr.Dataset, dict[str, FitParameters]]
        - Dataset containing the fit results
        - Dictionary of FitParameters for each sensor
    """
    sensors = node.namespace["sensors"]
    ds_fit = ds.copy()

    # Fit each sensor individually
    fit_results = {}
    success_list = []

    for sensor in sensors:
        sensor_data = ds.sel(sensors=sensor.id)
        fit_params = fit_individual_raw_data(sensor_data, sensor.id, node)
        fit_results[sensor.id] = fit_params
        success_list.append(fit_params.success)

    # Add success criteria to the dataset
    ds_fit = ds_fit.assign_coords(success=("sensors", success_list))

    # Set node outcomes
    node.outcomes = {sensor.id: "successful" if fit_results[sensor.id].success else "fail" for sensor in sensors}

    return ds_fit, fit_results


def fit_individual_raw_data(data: xr.Dataset, sensor_id: str, node: QualibrationNode) -> FitParameters:
    """
    Perform charge stability analysis for a single sensor.

    Parameters:
    -----------
    data : xr.Dataset
        Dataset containing the sensor's I and Q quadrature data.
    sensor_id : str
        The sensor identifier.
    node : QualibrationNode
        The calibration node containing parameters.

    Returns:
    --------
    FitParameters
        The fitted parameters including peak locations and gap shoulder results.
    """
    # Extract amplitude from I and Q
    amplitude = np.sqrt(data.I**2 + data.Q**2)
    zs = amplitude.values

    # Bayesian change point detection
    model = BayesianCP(hazard=1 / 200.0, standardize=True)
    cp, _ = jax.vmap(model.fit)(zs)  # cp has length T-1 (probability at each boundary)
    cp2, _ = jax.vmap(model.fit)(zs.T)

    mean_cp = (cp[1:] + cp2[1:].T) / 2.0
    edge_threshold = 0.25

    if analyze_edge_map is None:
        raise ImportError(
            "Line fitting requires scikit-image; install it to enable edge analysis."
        ) from _edge_line_import_error

    # Line-segment fitting on the edge map
    edge_base = zs[1:, 1:] if zs.shape[0] > 1 and zs.shape[1] > 1 else zs
    edge_analysis = analyze_edge_map(
        np.array(mean_cp),
        threshold=edge_threshold,
        base_image=edge_base,
        show=False,
    )

    intersections = np.vstack(edge_analysis["intersections"]) if edge_analysis["intersections"] else np.empty((0, 2))

    success = len(edge_analysis["segments"]) > 0

    return FitParameters(
        cp=cp,
        cp2=cp2,
        mean_cp=mean_cp,
        edge_binary=edge_analysis["binary_mask"],
        skeleton=edge_analysis["skeleton"],
        segments=edge_analysis["segments"],
        intersections=intersections,
        edge_threshold=edge_threshold,
        success=success,
    )
