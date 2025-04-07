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
        .melt(
            id_vars=[k for k in ds.coords.keys()],
            value_vars=[k for k in ds.data_vars.keys()],
        )
    )


def extract_dict(ds: xr.Dataset, data_var) -> dict:
    """
    from a dataset, extract a dictionary where the keys are the values of data_var and the values is the
    dataframe for this value
    """
    return ds[[data_var]].to_dataframe().to_dict()[data_var]


def convert_IQ_to_V(
    da: xr.Dataset,
    qubits: list[AnyTransmon],
    IQ_list: list[str] = ("I", "Q"),
    single_demod: bool = False,
) -> xr.Dataset:
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
        [q.resonator.operations["readout"].length for q in qubits],
        coords=[("qubit", [q.name for q in qubits])],
    )
    demod_factor = 2 if single_demod else 1
    return da.assign({key: da[key] * demod_factor * 2**12 / readout_lengths for key in IQ_list})


def add_amplitude_and_phase(
    ds: xr.Dataset,
    dim: str,
    unwrap_flag: bool = True,
    subtract_slope_flag: bool = False,
) -> xr.Dataset:
    """
    Adds the amplitude 'IQ_abs' and phase 'phase' data to the specified xarray dataset containing the quadratures 'I' and 'Q'.

    Parameters
    ----------
    ds : xr.Dataset
        The input xarray dataset containing the quadratures 'I' and 'Q'.
    dim : str
        The dimension along which to calculate the angle.
    unwrap_flag : bool, optional
        Flag indicating whether to unwrap the phase (default is True).
    subtract_slope_flag : bool, optional
        Flag indicating whether to subtract the slope from the 'detuning' coordinate (default is False).

    Returns
    -------
    xr.Dataset
        The modified xarray dataset with added 'IQ_abs' and 'phase' variables.

    Notes
    -----
    - The amplitude 'IQ_abs' is calculated as the absolute value of the complex signal formed by 'I' and 'Q'.
    - The phase 'phase' is calculated as the unwrapped angle of the complex signal formed by 'I' and 'Q'.
    - If `subtract_slope_flag` is True, the slope is subtracted from the 'detuning' coordinate using the `subtract_slope` function.
    """
    s = ds["I"] + 1j * ds["Q"]
    ds["IQ_abs"] = apply_modulus(s)
    ds["phase"] = apply_angle(s, dim, unwrap_flag)
    if subtract_slope_flag:
        ds["phase"] = subtract_slope(ds.phase, dim)
    return ds


def apply_modulus(da: xr.DataArray) -> xr.DataArray:
    """
    Applies the modulus (absolute value) function to each element in the input xarray DataArray.

    Parameters
    ----------
    da : xr.DataArray
        The input xarray DataArray to which the modulus function will be applied.

    Returns
    -------
    xr.DataArray
        The modified xarray DataArray with the modulus applied to each element.

    Notes
    -----
    - The function `np.abs` is used to compute the absolute value of each element in the DataArray.
    - The `xr.apply_ufunc` function is used to apply the modulus function element-wise across the DataArray.
    """

    def f(x):
        return np.abs(x)

    return xr.apply_ufunc(f, da)


def apply_angle(da: xr.DataArray, dim: str, unwrap=True) -> xr.DataArray:
    """
    Calculates the angle of complex data along the specified dimension.

    Useful for IQ data to obtain the phase, often used together with `subtract_slope`.

    Parameters
    ----------
    da : xr.DataArray
        The array on which to calculate the angle. Assumed to have complex data.
    dim : str
        The dimension along which to calculate the angle.
    unwrap : bool, optional
        Whether to unwrap the phase (default is False).

    Returns
    -------
    xr.DataArray
        A data array with the same dimensions and coordinates, with the angle calculated along the specified dimension.

    Notes
    -----
    - The function computes the angle (phase) of the complex data in the input DataArray.
    - If `unwrap` is True, the phase is unwrapped to avoid discontinuities.
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
    Subtracts a linear slope from a data array along the specified dimension.

    Parameters
    ----------
    da : xr.DataArray
        The array from which to subtract the linear slope.
    dim : str
        The dimension along which to subtract the linear slope.

    Returns
    -------
    xr.DataArray
        A data array with the same dimensions and coordinates, with a linear slope subtracted along the specified dimension.

    Notes
    -----
    - The function is useful for phase data to remove background phase shifts caused by electrical delays.
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

    da2 = xr.apply_ufunc(
        count_bins,
        da,
        input_core_dims=[[dim]],
        output_core_dims=[[da.name]],
        vectorize=True,
    )
    return da2.assign_coords({da.name: np.arange(len(da2[da.name]))})
