# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt

from qualang_tools.multi_user import qm_session

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.mixer_calibration import (
    Parameters,
    extract_relevant_fit_parameters,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from qualibration_libs.parameters import get_qubits

# %% {Node initialisation}
description = """
        MIXER CALIBRATION - Octave
 
Calibrates Octave IQ mixers for active sensors (resonator readout) and qubits (XY drive)
via the Octave calibration API. LO leakage suppression and image rejection are extracted
from the calibration results for logging and plotting.
 
Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Octave hardware connected and configured for the targeted elements.
 
Datasets:
    - Calibration payloads are stored in ``node.namespace["calibration_results"]``
      (per element: ``resonator`` and/or ``xy_drive``), not as an xarray ``ds_raw``.
 
Results:
    - ``fit_results[element].resonator``: LO leakage [dB] and image rejection [dB] when calibrated.
    - ``fit_results[element].xy_drive``: LO leakage [dB] and image rejection [dB] when calibrated.
    - ``fit_results[element].success``: whether extraction completed successfully.
 
Figures:
    - Per-element LO-leakage and image-rejection calibration plots (keyed by element name).
 
State update:
    - Octave mixer corrections are written by ``calibrate_octave`` during execution
      (no separate ``update_state`` action).
"""


node = QualibrationNode[Parameters, Quam](
    name="01a_mixer_calibration", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and run Octave mixer calibration for active sensors and qubits."""
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
                calibrate_drive=node.parameters.calibrate_drive,
            )
            node.namespace["calibration_results"][qubit.name] = {
                "xy_drive": calibration_results,
            }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Extract LO leakage and image rejection into fit_results and set per-element outcomes."""
    fit_results = extract_relevant_fit_parameters(node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        element_name: ("successful" if fit_result["success"] else "failed")
        for element_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot Octave LO-leakage and image-rejection calibration figures per element."""
    figs = plot_raw_data_with_fit(node)
    plt.show()
    # Store the generated figures
    node.results["figures"] = figs


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
