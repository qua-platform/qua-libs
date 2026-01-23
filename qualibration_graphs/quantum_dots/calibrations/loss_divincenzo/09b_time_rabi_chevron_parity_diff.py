# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.time_rabi_chevron_parity_diff import Parameters
from calibration_utils.common_utils.experiment import get_sensors, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        TIME RABI CHEVRON PARITY DIFFERENCE
This sequence performs a 2D chevron measurement with parity difference to characterize qubit coherence and 
coupling as a function of both pulse duration and frequency detuning. The measurement involves sweeping both 
the duration of a qubit control pulse (typically an X180 pulse) and the frequency detuning while measuring 
the parity state before and after the pulse using charge sensing via RF reflectometry or DC current sensing.

The sequence uses voltage gate sequences to navigate through a triangle in voltage space (empty - 
initialization - measurement) using OPX channels on the fast lines of the bias-tees. At each combination 
of pulse duration and frequency detuning, the parity is measured before (P1) and after (P2) the qubit pulse, 
and the parity difference (P_diff) is calculated. When P1 == P2, P_diff = 0; otherwise P_diff = 1.

The 2D chevron pattern in the parity difference signal reveals the qubit coupling strength as a function 
of both time and frequency, creating a characteristic chevron shape. This measurement is particularly useful 
for characterizing two-qubit gates, understanding the dynamics of coupled quantum dots, and identifying 
optimal operating points for qubit control.

Prerequisites:
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the voltage points (empty - initialization - measurement).
    - Qubit pulse calibration (X180 pulse amplitude and frequency).

State update:
    - The qubit x180 operation duration and frequency.
"""


node = QualibrationNode[Parameters, Quam](
    name="09b_time_rabi_chevron_parity_diff", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]
    # node.parameters.num_shots = 10
    # node.parameters.tau_min = 16
    # node.parameters.tau_max = 10000
    # node.parameters.tau_step = 52
    # node.parameters.frequency_min_in_mhz = -0.5
    # node.parameters.frequency_max_in_mhz = 0.525
    # node.parameters.frequency_step_in_mhz = 0.025
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active sensors and qubits from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue

            fit_result = node.results["fit_results"][qubit.name]
            qubit.xy.operations[node.parameters.operation].length = fit_result["optimal_duration"]
            # Update qubit frequency if optimal frequency is extracted
            # qubit.xy.intermediate_frequency = fit_result["optimal_frequency"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
