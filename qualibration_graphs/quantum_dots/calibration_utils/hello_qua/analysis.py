from __future__ import annotations

"""Analysis utilities for 00_hello_qua.

Mirrors the structure used by ``calibration_utils.sensor_dot``: the node's
``analyse_data`` action calls into this module instead of doing dataset
processing inline. ``00_hello_qua`` is a connectivity/sanity-check script
rather than a calibration, so there is no peak fitting here -- just
amplitude/phase derivation and a lightweight summary log.

Note on dataset shape: each (quantum_dot, sensor) trace is stored as its
own flat variable, ``I_{qd.name}_{i}`` / ``Q_{qd.name}_{i}`` (only ``voltage``
dimension), rather than combined into a single ``I``/``Q`` array indexed by
quantum_dot and sensor. This keeps the node's data-fetching step a one-liner
-- XarrayDataFetcher can't auto-combine these names (quantum dot names like
``virtual_dot_1`` contain digits, so its regex-based stacking doesn't apply)
-- so this module works with the flat names directly instead of reshaping.
"""

import logging
from typing import Callable, Optional

import numpy as np
import xarray as xr

from qualibrate.core import QualibrationNode

__all__ = [
    "process_raw_dataset",
    "log_processed_summary",
]


def process_raw_dataset(ds: xr.Dataset, node: QualibrationNode) -> xr.Dataset:
    """Add amplitude and (unwrapped) phase for each (quantum_dot, sensor) trace.

    Parameters
    ----------
    ds:
        Raw dataset containing, for each quantum dot ``qd`` and sensor index
        ``i``, flat variables ``I_{qd.name}_{i}`` / ``Q_{qd.name}_{i}``
        (dimension: ``voltage``).
    node:
        The calibration node; used to read the swept quantum dots/sensors
        from ``node.namespace``.

    Returns
    -------
    xr.Dataset
        ``ds`` with ``amplitude_{qd.name}_{i}`` / ``phase_{qd.name}_{i}``
        added for every trace. The original ``ds`` is left untouched.
    """
    quantum_dots = node.namespace["quantum_dots"]
    sensors = node.namespace["sensors"]

    new_vars = {}
    for qd in quantum_dots:
        for i in range(len(sensors)):
            I = ds[f"I_{qd.name}_{i}"]
            Q = ds[f"Q_{qd.name}_{i}"]

            amplitude = np.sqrt(I**2 + Q**2)
            amplitude.attrs = {"long_name": "IQ amplitude", "units": "V"}
            new_vars[f"amplitude_{qd.name}_{i}"] = amplitude

            phase = np.arctan2(Q, I)
            phase = phase.copy(data=np.unwrap(phase.values))
            phase.attrs = {"long_name": "IQ phase", "units": "rad"}
            new_vars[f"phase_{qd.name}_{i}"] = phase

    return ds.assign(new_vars)


def log_processed_summary(
    ds: xr.Dataset,
    quantum_dots,
    sensors,
    log_callable: Optional[Callable[[str], None]] = None,
) -> None:
    """Log a one-line amplitude summary per (quantum dot, sensor) trace.

    This has no notion of "success"/"failure" -- 00_hello_qua is a
    connectivity check, not a calibration -- so this simply reports the
    signal range seen for each trace, to help spot dead channels or an
    obviously mis-set voltage span at a glance.

    Parameters
    ----------
    ds:
        Processed dataset (post :func:`process_raw_dataset`).
    quantum_dots, sensors:
        Iterables of the swept dots / sensors used to build the sweep.
    log_callable:
        Callable used for logging. Defaults to the module logger.
    """
    if log_callable is None:
        log_callable = logging.getLogger(__name__).info

    for qd in quantum_dots:
        for i, sensor in enumerate(sensors):
            data = ds[f"amplitude_{qd.name}_{i}"]
            log_callable(
                f"[{qd.name} / {sensor.name}] amplitude: "
                f"min={float(data.min()):.4e} V, max={float(data.max()):.4e} V, "
                f"mean={float(data.mean()):.4e} V"
            )
