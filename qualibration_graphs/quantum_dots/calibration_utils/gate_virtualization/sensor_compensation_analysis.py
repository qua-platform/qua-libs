"""Analysis functions for sensor gate compensation (node 01).

Extracts the cross-talk coefficient between sensor gates and device gates
from 2D scans, enabling compensation so that sensor operating points remain
stable when device gates are swept.
"""

from typing import Any, Dict, List

import numpy as np
import xarray as xr


def extract_sensor_compensation_coefficients(
    ds: xr.Dataset,
    sensor_gate_name: str,
    device_gate_name: str,
) -> Dict[str, float]:
    """Extract the cross-talk coefficient between a sensor gate and a device gate.

    From a 2D scan of sensor_gate vs device_gate, fit the sensor signal
    response to determine how much the sensor gate must compensate when
    the device gate is changed.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset from the 2D scan (should contain ``amplitude``).
    sensor_gate_name : str
        Name of the sensor gate axis.
    device_gate_name : str
        Name of the device gate axis.

    Returns
    -------
    dict
        ``{"coefficient": float, "fit_quality": float, ...}``
    """
    # TODO: implement linear/polynomial fit of sensor response
    pass


def fit_sensor_response_per_line(
    ds: xr.Dataset,
    sweep_axis: str,
    fixed_axis: str,
) -> Dict[str, Any]:
    """Fit the sensor response along each line of the 2D scan.

    For each value of *fixed_axis*, fit the sensor signal along *sweep_axis*
    to extract a per-line slope.  The median slope gives the compensation
    coefficient while the spread indicates fit quality.

    Parameters
    ----------
    ds : xr.Dataset
        Processed dataset.
    sweep_axis : str
        Coordinate along which to fit (typically the device gate).
    fixed_axis : str
        Coordinate held constant per fit (typically the sensor gate).

    Returns
    -------
    dict
        ``{"slopes": np.ndarray, "median_slope": float, "std_slope": float}``
    """
    # TODO: implement per-line linear regression
    pass


def compute_compensation_matrix_entries(
    fit_results: Dict[str, Dict[str, float]],
) -> List[Dict[str, Any]]:
    """Convert per-pair fit results into compensation matrix entries.

    Parameters
    ----------
    fit_results : dict
        Keyed by ``"{sensor}_vs_{device}"`` with coefficient dicts.

    Returns
    -------
    list of dict
        Each entry: ``{"row": str, "col": str, "value": float}``.
    """
    # TODO: map fit coefficients to matrix row/col/value triples
    pass
