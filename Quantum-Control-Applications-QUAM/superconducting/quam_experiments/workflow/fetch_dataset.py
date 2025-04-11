import xarray as xr
from typing import Tuple
from quam_experiments.experiments.T1.parameters import Parameters
from qualibration_libs.qua_datasets import convert_IQ_to_V
from qualibration_libs.save_utils import fetch_results_as_xarray


def fetch_dataset(job, qubits, node_parameters: Parameters, sweep_axes: dict) -> xr.Dataset:
    """
    Fetches and processes a dataset from a quantum job.

    This function retrieves the results of a quantum job, processes the data into an xarray Dataset,
    and applies necessary transformations based on the provided parameters.

    Parameters:
    -----------
    job : Job
        The QM job containing the result handles.
    qubits : list
        A list of qubits involved in the program.
    node_parameters : Parameters
        Parameters related to the node, including whether state discrimination is used.
    sweep_axes : dict
        A dictionary of sweep axes, where keys are axis names and values are dictionaries
        containing 'data' and 'attrs'. The optional 'attrs' dictionary should include the standard xarray coordinate attributes.
        Example:
        {
            "idle_time": {
                "data": 4 * idle_times,
                "attrs": {"long_name": "idle time", "units": "ns"},
            },
        }

    Returns:
    --------
    xr.Dataset
        An xarray Dataset containing the processed measurement data.

    Notes:
    ------
    - The function first flattens the sweep parameters to obtain measurement axes and attributes.
    - It then fetches the results as an xarray Dataset.
    - If state discrimination is not used, the function converts IQ data to voltage (V) and sets appropriate attributes.
    - Finally, it assigns attributes to the coordinates based on the sweep parameters.

    Example:
    --------
        >>> ds = fetch_dataset(job, qubits, node_parameters, sweep_axes)
        >>> print(ds)
        Dimensions:    (qubit: 2, idle_time: 151)
        Coordinates:
          * qubit      (qubit) <U2 16B 'q1' 'q2'
          * idle_time  (idle_time)
        Data variables:
            Q          (qubit, idle_time)
            I          (qubit, idle_time)
    """

    measurement_axis, measurement_attributes = _flatten_sweep_parameters(sweep_axes)
    ds = fetch_results_as_xarray(job.result_handles, qubits, measurement_axis)
    if getattr(node_parameters, "use_state_discrimination"):
        if not node_parameters.use_state_discrimination:
            ds = convert_IQ_to_V(ds, qubits)
            ds.variables["I"].attrs = {"long_name": "I quadrature", "units": "V"}
            ds.variables["Q"].attrs = {"long_name": "Q quadrature", "units": "V"}
    for param in sweep_axes:
        ds.coords[param].attrs = measurement_attributes[param]
    return ds


def _flatten_sweep_parameters(params) -> Tuple[dict, dict]:
    meas_axis = {}
    meas_attr = {}
    for param in params.keys():
        if "data" in params[param]:
            meas_axis[param] = params[param]["data"]
        else:
            raise RuntimeError(
                f"The registered sweep parameter '{param}' doesn't have data."
                + "sweep_parameters = {param: {'data': data, 'attrs': {**attrs}}}"
            )
        if "attrs" in params[param]:
            meas_attr[param] = params[param]["attrs"]
        else:
            meas_attr[param] = {}
    return meas_axis, meas_attr
