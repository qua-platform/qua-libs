import xarray as xr

from quam_experiments.experiments.T1.parameters import Parameters
from quam_experiments.parameters.sweep_parameters import get_idle_times_in_clock_cycles
from quam_libs.qua_datasets import convert_IQ_to_V
from quam_libs.save_utils import fetch_results_as_xarray


def fetch_dataset(job, qubits, node_parameters: Parameters) -> xr.Dataset:
    """
    Fetches the measurement results for the T1 calibration node.

    Returns:
    - xr.Dataset: An xarray dataset containing the fetched measurement results with the following structure:

      **Coordinates:**
      - `idle_time` : A dimension representing idle times in nanoseconds.
      - `qubit` : A dimension listing the names of the qubits involved in the measurement.

      **Data Variables:**
      - Measurement variables, e.g., `I` and `Q`, or `state`, based on the stream handles fetched from the job.
        - Each variable has dimensions corresponding to the coordinates (`time`, `qubit`).

    """

    idle_times = get_idle_times_in_clock_cycles(node_parameters)
    ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times})
    if not node_parameters.use_state_discrimination:
        ds = convert_IQ_to_V(ds, qubits)
    ds = ds.assign_coords(idle_time=4 * ds.idle_time)
    ds.idle_time.attrs = {"long_name": "idle time", "units": "ns"}
    return ds
