from qualibrate.config.resolvers import get_quam_state_path
from qualibrate.storage.local_storage_manager import LocalStorageManager
from qualibrate_config.resolvers import get_qualibrate_config_path, get_qualibrate_config
from qualibrate_app.config import get_config_path, get_settings
from qualibrate import QualibrationNode
from qualibrate.utils.node.path_solver import get_node_dir_path
from quam_libs.components import QuAM
import os
from pathlib import Path
import xarray as xr
import json
import numpy as np
import logging

# ANSI color codes
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Custom formatter for magenta colored logs
class MagentaFormatter(logging.Formatter):
    def format(self, record):
        record.msg = f"{MAGENTA}{record.msg}{RESET}"
        return super().format(record)

# Configure logging with magenta color
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(MagentaFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

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
    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)
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


def find_numbered_folder(base_path, number):
    """
    Find folder that starts with '#number_'
    Will match '#number_something' but not '#number' alone
    """
    search_prefix = f"#{number}_"
    
    # Manual search for folder starting with #number_ and having something after
    for root, dirs, _ in os.walk(base_path):
        matching_dirs = [d for d in dirs if d.startswith(search_prefix) and len(d) > len(search_prefix)]
        if matching_dirs:
            return os.path.join(root, matching_dirs[0])
    
    return None



def load_dataset(serial_number, target_filename = "ds", parameters = None):
    """
    Loads a dataset from a file based on the serial number.
    
    Args:
        serial_number: The serial number to search for.
        base_folder: The base directory to search in.
    
    Returns:
        An xarray Dataset if found, None otherwise.
    """
    if not isinstance(serial_number, int):
        raise ValueError("serial_number must be an integer")
        
    base_folder = find_numbered_folder(get_storage_path(),serial_number)
    # Look for .nc files in the subfolder
    nc_files = [f for f in os.listdir(base_folder) if f.endswith('.h5')]
    
    # look for filename.h5
    is_present = target_filename in [file.split('.')[0] for file in nc_files]
    filename = [file for file in nc_files if target_filename == file.split('.')[0]][0] if is_present else None
    json_filename = "data.json"
    
    if nc_files:
        # Assuming there's only one .nc file per folder
        file_path = os.path.join(base_folder, filename)
        json_path = os.path.join(base_folder, json_filename)
        # Open the dataset
        ds = xr.open_dataset(file_path)
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        try:
            machine = QuAM.load(base_folder + "//quam_state.json")
        except Exception as e:
            print(f"Error loading machine: {e}")
            machine = None
        qubits = [machine.qubits[qname] for qname in ds.qubit.values]    
        if parameters is not None:
            for param_name, param_value in parameters:
                if param_name != "load_data_id":
                    if param_name in json_data["initial_parameters"]:
                        setattr(parameters, param_name, json_data["initial_parameters"][param_name])
            return ds, machine, json_data, qubits,parameters
        else:
            return ds, machine, json_data, qubits
    else:
        print(f"No .nc file found in folder: {base_folder}")
        return None

def get_node_id() -> int:
    
    q_config_path = get_qualibrate_config_path()
    qs = get_qualibrate_config(q_config_path)
    state_path = get_quam_state_path(qs)
    storage_manager = LocalStorageManager(
                root_data_folder=qs.storage.location,
                active_machine_path=state_path,
            )
    
    return storage_manager.data_handler.generate_node_contents()['id']

def save_node(node : QualibrationNode):
    """
    Save a QualibrationNode both locally and to cloud if possible.
    
    This function first saves the node locally, then attempts to upload to cloud
    if the necessary cloud dependencies are available and the user has proper access rights.
    The cloud upload is optional and will be skipped if:
    1. Cloud dependencies (IQCC_Cloud and QualibrateCloudHandler) are not available
    2. No quantum computer backend is specified
    3. User doesn't have proper IQCC project access rights
    """
    logger.info(f"Saving node with snapshot index {node.snapshot_idx}")
    
    node.save()
    logger.info("Node saved locally")
    
    # Check if cloud dependencies are available
    try:
        from iqcc_cloud_client import IQCC_Cloud
        from cloud_qualibrate_link.qualibrate_cloud_handler import QualibrateCloudHandler
        cloud_deps_available = True
    except ImportError:
        logger.info("Cloud dependencies not available - skipping cloud upload")
        cloud_deps_available = False
    
    # Only proceed with cloud upload if dependencies are available
    if cloud_deps_available:
        quantum_computer_backend = node.machine.network.get("quantum_computer_backend", None)
        if quantum_computer_backend is not None:
            logger.info(f"Found quantum computer backend: {quantum_computer_backend}")
            qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
            if qc.access_rights['projects'] == ['iqcc']:
                logger.info("IQCC project access confirmed, proceeding with cloud upload")
                q_config_path = get_qualibrate_config_path()
                qs = get_qualibrate_config(q_config_path)
                base_path = qs.storage.location
                node_id = node.snapshot_idx
                node_dir = get_node_dir_path(node_id, base_path)
                handler = QualibrateCloudHandler(str(node_dir))
                handler.upload_to_cloud(quantum_computer_backend)
                logger.info("Node successfully uploaded to cloud")
            else:
                logger.info("No IQCC project access - skipping cloud upload")
        else:
            logger.info("No quantum computer backend specified - skipping cloud upload")