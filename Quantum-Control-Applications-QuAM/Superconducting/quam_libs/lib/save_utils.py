from qualibrate_app.config import get_config_path, get_settings
from quam_libs.components import QuAM
import os
from pathlib import Path
import xarray as xr


def extract_string(input_string):
    # Find the index of the first occurrence of a digit in the input string
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)

    if index is not None:
        # Extract the substring from the start of the input string to the index
        extracted_string = input_string[:index]
        return extracted_string
    else:
        return None


def fetch_results_as_xarray(handles, qubits, measurement_axis):
    """
    Fetches measurement results as an xarray dataset.
    Parameters:
    - handles : A dictionary containing stream handles, obtained through handles = job.result_handles after the execution of the program.
    - qubits (list): A list of qubits.
    - measurement_axis (dict): A dictionary containing measurement axis information, e.g. {"frequency" : freqs, "flux",}.
    Returns:
    - ds (xarray.Dataset): An xarray dataset containing the fetched measurement results.
    """

    stream_handles = handles.keys()
    meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    values = [
        [handles.get(f"{meas_var}{i + 1}").fetch_all() for i, qubit in enumerate(qubits)] for meas_var in meas_vars
    ]
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}

    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )

    return ds


def get_storage_path():
    settings = get_settings(get_config_path())
    storage_location = settings.qualibrate.storage.location
    return Path(storage_location)


def find_folder(path, id):
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            if f"#{id}" in dir:
                return os.path.join(root, dir)
    return None


def load_dataset(serial_number):
    """
    Loads a dataset from a file based on the serial number.

    Args:
        serial_number: The serial number to search for.
        base_folder: The base directory to search in.

    Returns:
        An xarray Dataset if found, None otherwise.
    """
    if type(serial_number) == int:
        serial_number = str(serial_number)

    base_folder = find_folder(get_storage_path(), serial_number)
    # Look for .nc files in the subfolder
    nc_files = [f for f in os.listdir(base_folder) if f.endswith(".h5")]

    if nc_files:
        # Assuming there's only one .nc file per folder
        file_path = os.path.join(base_folder, nc_files[0])

        # Open the dataset
        ds = xr.open_dataset(file_path)
        try:
            machine = QuAM.load(base_folder)
        except Exception as e:
            print(f"Error loading machine: {e}")
            machine = None
        return ds, machine
    else:
        print(f"No .nc file found in folder: {base_folder}")
        return None
