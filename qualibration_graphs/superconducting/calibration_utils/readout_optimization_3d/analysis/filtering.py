import numpy as np
import xarray as xr

from scipy.ndimage import gaussian_filter

from calibration_utils.readout_optimization_3d.parameters import (
    ReadoutOptimization3dParameters,
)


def filter_readout_fidelity(ds: xr.Dataset, node_parameters: ReadoutOptimization3dParameters):
    """
    Apply Gaussian filter to smooth the data across the frequency and amplitude dimensions.
    Doesn't apply any smoothing across the duration dimension because it is usually sparse.
    """
    sigma = node_parameters.fidelity_smoothing_intensity

    fidelity = ds.raw_fidelity.copy()
    for qubit in ds.qubit:
        for duration in ds.duration:
            fidelity.loc[{"duration": duration, "qubit": qubit}] = gaussian_filter(
                fidelity.sel(duration=duration, qubit=qubit).values,
                sigma=(sigma, sigma),
            )

    # Convert the smoothed data back to an xarray.DataArray
    # smoothed_fidelity = xr.DataArray(smoothed_data, coords=ds.raw_fidelity.coords, dims=ds.raw_fidelity.dims)

    return fidelity
