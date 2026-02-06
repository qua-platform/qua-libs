# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.geometric_cz.parameters import Parameters
from calibration_utils.common_utils.experiment import get_sensors, get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        GEOMETRIC CZ GATE CALIBRATION - using standard QUA (pulse > 16ns and 4ns granularity)
The goal of this script is to calibrate a geometric controlled-Z (CZ) gate by finding the exchange pulse
amplitude and duration where the conditional phase equals pi and the SWAP oscillation completes a full 2pi
rotation (returning to the initial state with no population exchange).

A geometric CZ gate leverages the exchange interaction between two spin qubits. When the exchange coupling J
is pulsed, two effects occur simultaneously:
    1) SWAP oscillations: Population exchange between |up,down> and |down,up> states at frequency J.
    2) Conditional phase accumulation: The |up,up> and |down,down> states acquire phase relative to the
       antiparallel states.

For a perfect CZ gate (diag(1, 1, 1, -1)), we need:
    - Conditional phase = pi: The target qubit accumulates a pi phase shift when the control qubit is |up>
      relative to when it is |down>.
    - SWAP angle = 2*pi*n (integer n): The SWAP oscillation completes a full cycle, returning the population
      to its initial state with no net exchange.

This measurement performs a 2D sweep of exchange pulse amplitude vs duration while measuring both:
    1) Phase oscillations: Prepare target qubit in superposition (X90), apply exchange pulse, project phase
       back to Z-axis (X90). Repeat for control qubit in |down> vs |up> to extract conditional phase.
    2) SWAP oscillations: Prepare |up,down> state, apply exchange pulse, measure population oscillations.

The QUA program sequence for phase measurement:
    1) Initialize both qubits (load electrons).
    2) Prepare control qubit in |down> or |up> state.
    3) Apply X90 to target qubit (create superposition).
    4) Step to exchange point and apply exchange pulse with swept amplitude/duration.
    5) Step back to operation point.
    6) Apply X90 (or -X90) to target qubit to project phase to Z-axis.
    7) Measure both qubit states via parity readout.

The QUA program sequence for SWAP measurement:
    1) Initialize both qubits.
    2) Prepare |up,down> state (flip one qubit with X180).
    3) Step to exchange point and apply exchange pulse with swept amplitude/duration.
    4) Step back to operation point.
    5) Measure both qubit states via parity readout.

Analysis:
    - Phase oscillations: Extract conditional phase phi(V, t) from Ramsey-like fringes.
    - SWAP oscillations: Extract SWAP angle theta(V, t) from population oscillations.
    - CZ condition: Find contours where phi = pi and theta = 2*pi*n, their intersection gives CZ parameters.
    - The "geometric" nature means the gate is more robust to pulse shape imperfections.

Prerequisites:
    - Having calibrated single-qubit gates (X90, X180) for both qubits.
    - Having calibrated the readout for the qubit pair (parity readout).
    - Having characterized the exchange coupling vs barrier voltage (from CROT spectroscopy).
    - Having set appropriate voltage points for initialization, operation, and exchange.

Before proceeding to the next node:
    - Identify the optimal (amplitude, duration) point where phi = pi and theta = 2*pi.
    - Verify gate quality by checking that phi and theta contours intersect cleanly.
    - Consider pulse shaping (e.g., cosine envelope) for adiabatic operation if needed.

State update:
    - cz_exchange_amplitude
    - cz_exchange_duration
    - cz_conditional_phase
    - cz_swap_angle
"""


node = QualibrationNode[Parameters, Quam](
    name="15a_geometric_cz_calibration", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
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
    # Get the active sensors and qubit pairs from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data to extract conditional phase and SWAP angle as functions of amplitude and duration."""
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the phase oscillations, SWAP oscillations, and overlay the CZ condition contours."""
    pass


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the CZ calibration was successful."""

    with node.record_state_updates():
        for qubit_pair in node.namespace["qubit_pairs"]:
            if not node.results["fit_results"][qubit_pair.name]["success"]:
                continue
            fit_result = node.results["fit_results"][qubit_pair.name]
            # Update CZ gate parameters in the qubit pair
            qubit_pair.macros["cz"].amplitude = fit_result["optimal_amplitude"]
            qubit_pair.macros["cz"].duration = fit_result["optimal_duration"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()