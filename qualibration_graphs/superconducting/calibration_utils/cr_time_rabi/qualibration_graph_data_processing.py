import numpy as np
import xarray as xr

from quam_builder.architecture.superconducting.qubit import AnyTransmon
from quam_builder.architecture.superconducting.qubit_pair import AnyTransmonPair


__all__ = ["convert_IQ_to_V", "add_amplitude_and_phase", "subtract_slope"]


def convert_IQ_to_V(
    da: xr.Dataset,
    qubits: list[AnyTransmon] = None,
    qubit_pairs: list[AnyTransmonPair] = None,
    IQ_list: list[str] = ("I", "Q"),
    single_demod: bool = False,
) -> xr.Dataset:
    """
    return data array with the 'I' and 'Q' quadratures converted to Volts.

    :param da: the array on which to calculate angle. Assumed to have complex data
    :type da: xr.DataArray
    :param qubits: the list of qubits components.
    :type qubits: list[AnyTransmon]
    :param qubit_pairs: the list of qubit pair components.
    :type qubit_pairs: list[AnyTransmonPair]
    :param IQ_list: the list of data to convert to V e.g. ["I", "Q"].
    :type IQ_list: list[str]
    :param single_demod: Flag to add the additional factor of 2 needed for single demod.
    :type single_demod: bool
    :return: a data array with the same dimensions and coordinates, with a 'I' and 'Q' in Volts
    :type: xr.DataArray

    """
    # Create a xarray with a coordinate 'qubit' and the value is q.resonator.operations["readout"].length
    if qubits is not None:
        readout_lengths = xr.DataArray(
            [q.resonator.operations["readout"].length for q in qubits],
            coords=[("qubit", [q.name for q in qubits])],
        )
    elif qubit_pairs is not None:
        control_target = ["c", "t"]
        readout_lengths = xr.DataArray(
            data=[
                [
                    qp.qubit_control.resonator.operations["readout"].length,
                    qp.qubit_target.resonator.operations["readout"].length
                ]
                for qp in qubit_pairs
            ],
            dims=["qubit_pair", "control_target"],
            coords={
                "qubit_pair": [qp.name for qp in qubit_pairs],
                "control_target": control_target,
            },
        )
    else:
        raise ValueError("Either qubits or qubit_pairs must be provided!")

    demod_factor = 2 if single_demod else 1
    return da.assign(
        {key: da[key] * demod_factor * 2**12 / readout_lengths for key in IQ_list}
    )


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
    ds["IQ_abs"] = _apply_modulus(s)
    ds["phase"] = _apply_angle(s, dim, unwrap_flag)
    if subtract_slope_flag:
        ds["phase"] = subtract_slope(ds.phase, dim)
    return ds


def _apply_modulus(da: xr.DataArray) -> xr.DataArray:
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


def _apply_angle(da: xr.DataArray, dim: str, unwrap=True) -> xr.DataArray:
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

    return xr.apply_ufunc(
        sub_slope, da, input_core_dims=[[dim]], output_core_dims=[[dim]]
    )
