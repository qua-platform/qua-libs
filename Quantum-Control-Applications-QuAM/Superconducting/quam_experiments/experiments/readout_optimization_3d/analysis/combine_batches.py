from typing import List

import xarray as xr
import numpy as np


def combine_batches(datasets: List[xr.Dataset]):
    for i in range(len(datasets) - 1):
        datasets[i + 1] = datasets[i].combine_first(datasets[i + 1])

    combined_ds = datasets[-1]
    combined_ds = combined_ds.map(lambda da: _shift_left(da, dim="run"))
    combined_ds = combined_ds.where(~np.isnan(combined_ds), drop=True)

    return combined_ds


def _shift_left(da: xr.DataArray, dim: str):
    """Shifts all non-NaN values left along the specified dimension."""
    axis = da.get_axis_num(dim)
    mask = ~np.isnan(da)
    sorted_indices = np.argsort(~mask, axis=axis)  # Indices that push NaNs to the right

    # Reorder values along the given dimension
    shifted = xr.apply_ufunc(np.take_along_axis, da, sorted_indices, axis, dask="allowed")

    return shifted
