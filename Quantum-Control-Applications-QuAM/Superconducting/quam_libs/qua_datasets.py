from typing import Optional
from quam_builder.architecture.superconducting.qubit import AnyTransmon
import numpy as np
import xarray as xr


def to_long_dataframe(ds: xr.Dataset):
    """
    convert a dataset to a pandas dataframe in a long format. This is primarily used when plotting with plotly
    """
    return (
        ds.to_dataframe()
        .reset_index()
        .melt(id_vars=[k for k in ds.coords.keys()], value_vars=[k for k in ds.data_vars.keys()])
    )


def extract_dict(ds: xr.Dataset, data_var) -> dict:
    """
    from a dataset, extract a dictionary where the keys are the values of data_var and the values is the
    dataframe for this value
    """
    return ds[[data_var]].to_dataframe().to_dict()[data_var]


def convert_IQ_to_V(
    da: xr.DataArray, qubits: list[AnyTransmon], IQ_list: list[str] = ("I", "Q"), single_demod: bool = False
) -> xr.DataArray:
    # TODO: this is Transmon specific so it should probably not belong here
    """
    return data array with the 'I' and 'Q' quadratures converted to Volts.

    :param da: the array on which to calculate angle. Assumed to have complex data
    :type da: xr.DataArray
    :param qubits: the list of qubits components.
    :type qubits: list[AnyTransmon]
    :param IQ_list: the list of data to convert to V e.g. ["I", "Q"].
    :type IQ_list: list[str]
    :param single_demod: Flag to add the additional factor of 2 needed for single demod.
    :type single_demod: bool
    :return: a data array with the same dimensions and coordinates, with a 'I' and 'Q' in Volts
    :type: xr.DataArray

    """
    # Create a xarray with a coordinate 'qubit' and the value is q.resonator.operations["readout"].length
    readout_lengths = xr.DataArray(
        [q.resonator.operations["readout"].length for q in qubits], coords=[("qubit", [q.name for q in qubits])]
    )
    demod_factor = 2 if single_demod else 1
    return da.assign({key: da[key] * demod_factor * 2**12 / readout_lengths for key in IQ_list})


def apply_angle(da: xr.DataArray, dim: str, unwrap=True) -> xr.DataArray:
    """
    return data array with the angle of complex data calculated along dimension `dim`.
    Useful when we have IQ data, and we want to get the phase of it, usually together with `subtract_slope`.

    :param da: the array on which to calculate angle. Assumed to have complex data
    :type da: xr.DataArray
    :param dim: the dimension along which to subtract
    :type dim: str
    :param unwrap: whether to unwrap the phase
    :type unwrap: bool, optional
    :return: a data-array with the same dimensions and coordinates, with a linear slope subtracted along dim
    :type: xr.DataArray
    """
    if unwrap:

        def f(x):
            return np.unwrap(np.angle(x))

    else:

        def f(x):
            return np.angle(x)

    return xr.apply_ufunc(f, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def subtract_slope(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    subtract a linear slope from a data array along dimension `dim`.
    This is usually useful when looking at phase data and when we need to subtract the background phase shift due to cable delay

    :param da: the array from which to subtract
    :type da: xr.DataArray
    :param dim: the dimension along which to subtract
    :type dim: str
    :return: a data-array with the same dimensions and coordinates, with a linear slope subtracted along dim
    :rtype: xr.DataArray
    """
    x = da[dim]

    def sub_slope(arr):
        def fit_func(a):
            return np.polyfit(x, a, deg=1)

        def eval_func(c):
            return np.polyval(c, x)

        pf = np.apply_along_axis(fit_func, -1, arr)
        evaluated = np.apply_along_axis(eval_func, -1, pf)
        return arr - evaluated

    return xr.apply_ufunc(sub_slope, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def unrotate_phase(da: xr.DataArray, dim: str) -> xr.DataArray:
    """
    unrotate the linear slope of a complex IQ plane data array along dimension `dim`.
    This is usually useful when we want to plot resonator circles in the IQ plane without the overall phase shift

    This evaluates a linear slope of a phase and then applies to the array *exp(-i*phase).

    :param da: the array from which to unrotate
    :type da: xr.DataArray
    :param dim: the dimension along which to unrotate
    :type dim: str
    :return: a data-array with the same dimensions and coordinates, with phase unrotated along dim
    :rtype: xr.DataArray
    """
    x = da[dim]
    angle = apply_angle(da, dim).values

    def unrotate(arr):
        def fit_func(a):
            return np.polyfit(x, a, deg=1)

        def eval_func(c):
            return np.polyval(c, x)

        pf = np.apply_along_axis(fit_func, -1, angle)
        evaluated_angle = np.apply_along_axis(eval_func, -1, pf)
        return arr * np.exp(-1j * evaluated_angle)

    return xr.apply_ufunc(unrotate, da, input_core_dims=[[dim]], output_core_dims=[[dim]])


def integer_histogram(da: xr.DataArray, dim: str, min_length: Optional[int] = None) -> xr.DataArray:
    """Calculate histogram of non-negative integer values of data array along
    a dimension.

    :param da: data array containing non-negative integer values only
    :type da: xr.DataArray
    :param dim: dimension along which to calculate histogram. This dimension will be converted into a bins dimension.
    :type dim: str
    :param min_length: if included, bins will include values from 0 to min_length-1
    :type min_length: int
    :return: a data array with counts on integers, all dimensions identical to `da`, except along `dim`.
    :rtype: xr.DataArray
    """

    if da.name is None:
        raise ValueError("da must have a name")

    def count_bins(vec):
        return np.bincount(vec, minlength=min_length)

    da2 = xr.apply_ufunc(count_bins, da, input_core_dims=[[dim]], output_core_dims=[[da.name]], vectorize=True)
    return da2.assign_coords({da.name: np.arange(len(da2[da.name]))})
