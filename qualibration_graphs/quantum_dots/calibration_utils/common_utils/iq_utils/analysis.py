"""
Shared analysis utilities for IQ readout calibration.
"""

from __future__ import annotations

from typing import Any

import xarray as xr


def process_raw_dataset(ds: xr.Dataset, *_: Any, **__: Any) -> xr.Dataset:
    """Strip tuple-wrapping coming out of fetchers.

    Some data fetchers return values as 1-tuples; downstream analysis expects
    plain numeric arrays. This helper is shared across IQ readout nodes.
    """

    def extract_value(element: Any):
        if isinstance(element, tuple):
            return element[0]
        return element

    return xr.apply_ufunc(
        extract_value,
        ds,
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )


__all__ = ["process_raw_dataset"]
