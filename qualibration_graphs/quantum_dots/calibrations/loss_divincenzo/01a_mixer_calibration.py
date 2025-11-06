# %% {Imports}
from dataclasses import asdict
import matplotlib.pyplot as plt
from qualang_tools.multi_user import qm_session
from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_sensors, get_qubits
from calibration_utils.mixer_calibration import (
    Parameters,
    extract_relevant_fit_parameters,
    log_fitted_results,
    plot_raw_data_with_fit,
)

description = """
    A simple program to calibrate Octave mixers for all qubits and resonators
"""


node = QualibrationNode[Parameters, Quam](
    name="01a_mixer_calibration", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load("/Users/kalidu_laptop/.qualibrate/quam_state")


# %% {Execute_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Get the active sensors from the node and organize them by batches
    node.namespace["sensors"] = sensors = get_sensors(node)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)

    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["calibration_results"] = {}
        for sensor in sensors:
            calibration_results = sensor.calibrate_octave(
                qm,
                calibrate_resonator=node.parameters.calibrate_resonator,
            )
            node.namespace["calibration_results"][sensor.name] = {
                "resonator": calibration_results,
            }
        for qubit in qubits: 
            calibration_results = qubit.calibrate_octave(
                qm, 
                calibrate_drive = node.parameters.calibrate_drive,
            )
            node.namespace["calibration_results"][qubit.name] = {
                "xy_drive": calibration_results,
            }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results in the fit_results class."""
    fit_results = extract_relevant_fit_parameters(node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit.grid_location."""
    figs = plot_raw_data_with_fit(node)
    plt.show()
    # Store the generated figures
    node.results["figures"] = figs


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
