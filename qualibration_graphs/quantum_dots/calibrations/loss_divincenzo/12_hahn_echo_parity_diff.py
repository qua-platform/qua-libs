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
from calibration_utils.common_utils.experiment import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        HAHN ECHO (SPIN ECHO) T2 MEASUREMENT - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to measure the spin-spin relaxation time T2 using the Hahn echo (spin echo) technique.
Unlike the Ramsey experiment which measures T2* (sensitive to low-frequency noise and inhomogeneous broadening),
the Hahn echo refocuses static dephasing, yielding the intrinsic T2 coherence time which is always >= T2*.

The QUA program is divided into three sections:
    1) step between the initialization point and the operation point using sticky elements.
    2) apply the Hahn echo pulse sequence: pi/2 - tau - pi - tau - pi/2.
    3) measure the state of the qubit using RF reflectometry via parity readout.

The Hahn echo sequence works by:
    - First pi/2 pulse (x90): Creates superposition, placing qubit on Bloch sphere equator.
    - First wait period (tau): Qubit dephases due to noise and field inhomogeneities.
    - Pi pulse (y180): Flips the qubit state, reversing the accumulated phase.
    - Second wait period (tau): Previously accumulated phase is undone (refocused).
    - Final pi/2 pulse (x90): Projects the refocused state for measurement.

The echo amplitude decays as exp(-2*tau/T2), where T2 reflects irreversible dephasing from
high-frequency noise that cannot be refocused. This is the simplest dynamical decoupling sequence
and forms the basis for more advanced sequences (CPMG, XY-n) that extend coherence further.

Prerequisites:
    - Having run the Ramsey node to calibrate the qubit frequency and T2*, and the corresponding prerequisites.
    - Having calibrated pi and pi/2 pulse parameters from Rabi measurements.

Before proceeding to the next node:
    - Extract T2 from exponential fit of the echo decay curve.
    - Compare T2 to T2* to assess the contribution of low-frequency noise.
    - Consider dynamical decoupling sequences if longer coherence is needed.

State update:
    - T2echo
"""


node = QualibrationNode[Parameters, Quam](
    name="12_hahn_echo_parity_diff", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["q1"]

    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    pass


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
    # Get the active qubits from the loaded node parameters
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    pass


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""

    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            if not node.results["fit_results"][qubit.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit.name]
            qubit.T2echo = fit_result["T2_echo"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
