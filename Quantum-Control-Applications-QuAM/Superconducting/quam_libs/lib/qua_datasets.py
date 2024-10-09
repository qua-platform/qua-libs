from typing import Optional

import numpy as np
import xarray as xr


def to_long_dataframe(ds: xr.Dataset):
    """
    convert a dataset to a pandas dataframe in a long format. This is primarily used when plotting with plotly
    """
    return ds.to_dataframe().reset_index().melt(id_vars=[k for k in ds.coords.keys()],
                                                value_vars=[k for k in ds.data_vars.keys()])


def extract_dict(ds: xr.Dataset, data_var) -> dict:
    """
    from a dataset, extract a dictionary where the keys are the values of data_var and the values is the
    dataframe for this value
    """
    return ds[[data_var]].to_dataframe().to_dict()[data_var]


def apply_angle(da: xr.DataArray, dim: str, unwrap=True) -> xr.DataArray:
    """
    return data array with the angle of complex data calculated along dimension `dim`.
    Useful when we have IQ data and we want to get the phase of it, usually together with `subtract_slope`.

    :param da: the array on which to calculate angle. Assumed to have complex data
    :type da: xr.DataArray
    :param dim: the dimesnsion along which to subtract
    :type dim: str
    :param unwrap: whether to unwrap the phase 
    :type unwrap: bool, optional
    :return: a dataarray with the same dimensions and coordinates, with a linear slope subtrcated along dim
    :rtype: xr.DataArray
    """
    if unwrap:
        def f(x): return np.unwrap(np.angle(x))
    else:
        def f(x): return np.angle(x)

    return xr.apply_ufunc(f, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def subtract_slope(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    subtract a linear slope from a data array along dimension `dim`.
    This is usually useful when looking at phase data and when we need to subtract the background phase shift due to cable delay

    :param da: the array from which to subtract
    :type da: xr.DataArray
    :param dim: the dimesnsion along which to subtract
    :type dim: str
    :return: a dataarray with the same dimensions and coordinates, with a linear slope subtrcated along dim
    :rtype: xr.DataArray
    """
    x = da[dim]

    def sub_slope(arr):
        def fit_func(a): return np.polyfit(x, a, deg=1)
        def eval_func(c): return np.polyval(c, x)
        pf = np.apply_along_axis(fit_func, -1, arr)
        evaled = np.apply_along_axis(eval_func, -1, pf)
        return arr - evaled

    return xr.apply_ufunc(sub_slope, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def unrotate_phase(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    unrotate the linear slope of a complex IQ plane data array along dimension `dim`.
    This is usually useful when we want to plot resonator circles in the IQ plane without the overall phase shift

    This evaluates a linear slope of a phase and then applies to the array *exp(-i*phase).

    :param da: the array from which to unrotate
    :type da: xr.DataArray
    :param dim: the dimesnsion along which to unrotate
    :type dim: str
    :return: a dataarray with the same dimensions and coordinates, with phase unrotated along dim  
    :rtype: xr.DataArray
    """
    x = da[dim]
    angle = apply_angle(da, dim).values

    def unrotate(arr):
        def fit_func(a): return np.polyfit(x, a, deg=1)
        def eval_func(c): return np.polyval(c, x)
        pf = np.apply_along_axis(fit_func, -1, angle)
        evaled_angle = np.apply_along_axis(eval_func, -1, pf)
        return arr*np.exp(-1j*evaled_angle)

    return xr.apply_ufunc(unrotate, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def integer_histogram(da: xr.DataArray, dim: str, minlength: Optional[int] = None) -> xr.DataArray:
    """Calculate histogram of non-negative integer values of data array along
    a dimension.

    :param da: data array containing non-negative integer values only
    :type da: xr.DataArray
    :param dim: dimension along which to calculate histogram. This dimension will be converted into a bins dimension.
    :type dim: str
    :param minlength: if included, bins will include values from 0 to minlength-1
    :type minlength: int
    :return: a data array with counts on integers, all dimensions identical to `da`, except along `dim`.
    :rtype: xr.DataArray
    """

    if da.name is None:
        raise ValueError("da must have a name")

    def count_bins(vec):
        return np.bincount(vec, minlength=minlength)
    da2 = xr.apply_ufunc(count_bins, da, input_core_dims=[
                         [dim]], output_core_dims=[[da.name]], vectorize=True)
    return da2.assign_coords({da.name: np.arange(len(da2[da.name]))})






