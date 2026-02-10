"""Single-qubit standard gate set tomography calibration node.

This module implements a calibration node for performing standard gate set tomography on a single qubit
to measure the performance of the gate set as well as the initialization and measurement processes.
"""

# pylint: disable=duplicate-code

# %% {Imports}
from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from more_itertools import flatten
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode
from qualibrate.parameters import NodeParameters
from qualibration_libs.data import XarrayDataFetcher
from quam_config import Quam

from calibration_utils.gate_set_tomography.parameters import Parameters


# %% {Initialisation}

description = """
SINGLE-QUBIT GATE SET TOMOGRAPHY

Gate set tomography (GST) is a self-consistent protocol that simultaneously 
characterizes the prepared state, the measurement, and the quantum gates themselves.
The program performs GST on a single qubit by executing
carefully designed sequences of quantum gates and measuring the qubit state
after each sequence. 

The gate sequences used in the experiment can be divided into three parts:
    - Preparation fiducials
    - Germ sequences repeated at increasing lengths
    - Measurement fiducials

The goal of the preparation and measurement fiducials is to provide a sufficiently varied
set of input states and measurements to thoroughly probe the effect of the germ
studied. The germs are operations of interest such as the gates
themselves, as well as short sequences of gates that are specifically chosen to amplify
errors such as over- or under-rotations in the individual gates and tilt errors on
their rotation axes. Each sequence is constructed by concatenating a preparation fiducial, a germ
repeated a variable number of times (specified as an input), and a measurement
fiducial. (prep fiducial) + (germ)^L + (meas fiducial)

The gate sequences are generated offline and expressed in terms of the native
single-qubit basis gate set (e.g. ['rz', 'sx', 'x']). This basis gate set must be 
tomographically complete, requiring at least two non-commuting single-qubit gates 
that rotate about different axes.
The circuits are then executed using a switch_case block structure to efficiently play back the
different gate sequences on the hardware.

Each circuit is repeated multiple times to collect statistics, and measurements
are performed in the computational basis. The resulting measurement outcomes
are post-processed using a GST analysis package to reconstruct the process
matrices of the gates, as well as the state preparation and measurement (SPAM)
operations.

Key Features:
    - use_input_stream: When enabled, the gate sequences are streamed to the OPX
      using the input stream feature, enabling dynamic circuit execution and
      reducing memory usage.

Prerequisites:
    - Having calibrated the single-qubit gates used in the GST gate set.
    - Having calibrated the qubit readout (readout frequency, amplitude,
      duration, and IQ blobs).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18b_gate_set_tomography_1q",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/gate_set_tomography/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1"]


if node.parameters.use_input_stream:
    raise NotImplementedError("Input streams is not supported yet.")

# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    pass


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset
    called "ds_raw".
    """
    # # Connect to the QOP
    # qmm = node.machine.connect()
    # # Get the config from the machine
    # config = node.machine.generate_config()
    # # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    # with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    #     # The job is stored in the node namespace to be reused in the fetching_data run_action
    #     node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
    #     # Display the progress bar
    #     data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
    #     for dataset in data_fetcher:
    #         progress_counter(
    #             data_fetcher["n"],
    #             node.parameters.num_shots,
    #             start_time=data_fetcher.t_start,
    #         )
    #     # Display the execution report to expose possible runtime errors
    #     node.log(job.execution_report())
    # # Register the raw dataset
    # node.results["ds_raw"] = dataset


# %% {Load_data}
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
    """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results."""
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    pass


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    pass

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()