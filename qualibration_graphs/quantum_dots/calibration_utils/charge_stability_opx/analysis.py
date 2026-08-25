import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import xarray as xr
from qualibrate.core import QualibrationNode


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
            """Convert a stored segment to a serializable dictionary."""
            if isinstance(seg, dict):
                return seg
            if all(hasattr(seg, attr) for attr in ("start", "end", "centroid", "direction", "slope", "intercept")):
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
            "mean_cp": (np.asarray(self.mean_cp).tolist() if self.mean_cp is not None else []),
            "edge_binary": (np.asarray(self.edge_binary).tolist() if self.edge_binary is not None else []),
            "skeleton": (np.asarray(self.skeleton).tolist() if self.skeleton is not None else []),
            "segments": [serialize_segment(s) for s in (self.segments or [])],
            "intersections": (np.asarray(self.intersections).tolist() if self.intersections is not None else []),
            "edge_threshold": float(self.edge_threshold),
            "success": self.success,
        }


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


def process_raw_dataset(ds: xr.Dataset, node: Optional[QualibrationNode] = None):
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


def analyse_raw_data(
    ds_processed: xr.Dataset,
    node: QualibrationNode,
    *,
    log_callable=None,
) -> tuple[xr.Dataset, dict, dict]:
    """Process the raw dataset and, when enabled, run edge analysis."""
    if not node.parameters.perform_edge_analysis:
        return ds_processed, {}, {}

    ds_fit, fit_results = fit_raw_data(ds_processed, node)
    fit_results_dict = {k: v.to_dict() for k, v in fit_results.items()}
    log_fitted_results(fit_results_dict, log_callable=log_callable)
    outcomes = {
        name: ("successful" if fit_result["success"] else "failed") for name, fit_result in fit_results_dict.items()
    }
    return ds_fit, fit_results_dict, outcomes


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
    # Extract amplitude from I and Q, sorted by physical voltage coordinates so
    # that change-point detection runs on spatially ordered data regardless of
    # the scan pattern used (e.g. SwitchRasterScan stores rows in interleaved order).
    amplitude = np.sqrt(data.I**2 + data.Q**2)
    amplitude = amplitude.sortby("x_volts").sortby("y_volts")
    zs = amplitude.values

    from .edge_line_analysis import analyze_sensor_edge_map

    edge_results = analyze_sensor_edge_map(
        zs,
        threshold=0.25,
        show=False,
    )

    success = len(edge_results["segments"]) > 0

    return FitParameters(
        cp=edge_results["cp"],
        cp2=edge_results["cp2"],
        mean_cp=edge_results["mean_cp"],
        edge_binary=edge_results["binary_mask"],
        skeleton=edge_results["skeleton"],
        segments=edge_results["segments"],
        intersections=edge_results["intersections"],
        edge_threshold=edge_results["edge_threshold"],
        success=success,
    )
