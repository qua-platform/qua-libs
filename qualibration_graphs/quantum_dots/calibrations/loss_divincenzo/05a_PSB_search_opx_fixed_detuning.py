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
from calibration_utils.psb_search_fixed_detuning import Parameters
from calibration_utils.common_utils.experiment import get_sensors, _make_batchable_list_from_multiplexed
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates


# %% {Node initialisation}
description = """
        PAULI SPIN BLOCKADE SEARCH - Fixed Detuning
The goal of this sequence is to find the Pauli Spin Blockade (PSB) region.
To do so, the following triangle in voltage space (empty - random initialization - measurement) is applied using OPX
channels on the fast lines of the bias-tees.

The OPX measures the response via RF reflectometry or DC current sensing during the readout window
(last segment of the triangle). The sequence is repeated several time in order to build up histograms.

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage on average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time, unless a compensation pulse is played.

Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.
    - Having calibrated the "empty" and "initialization" voltage points, and having defined the detuning axis.

State update:
    TODO: It seems to me that this node is just to quickly check the readout region and nothing should be updated,
    unless we use it to rotate the IQ blobs and define a threshold for state discrimination.
"""


node = QualibrationNode[Parameters, Quam](
    name="05a_PSB_search_opx_fixed_detuning", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.quantum_dot_pair_names = ["virtual_dot_1_virtual_dot_2_pair"]
    # node.parameters.num_shots = 10
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    dot_pair_objects = [node.machine.quantum_dot_pairs[name] for name in node.parameters.quantum_dot_pair_names]

    node.namespace["dot_pairs"] = dot_pair_objects

    multiplexed_sensors_by_dot_pair = {
        pair.name: _make_batchable_list_from_multiplexed(pair.sensor_dots, multiplexed = node.parameters.multiplexed) for pair in dot_pair_objects
    }

    # The swept 'axes'. Do we do a full 2D scan here? 
    node.namespace["sweep_axes"] = {
        "quantum_dot_pair": xr.DataArray([pair.name for pair in dot_pair_objects]),
    }
    with program() as prog: 
        n = declare(int)
        n_st = declare_stream()

        I_st = {pair.name: [declare_stream() for _ in range(len(multiplexed_sensors_by_dot_pair[pair.name].batch()))] for pair in dot_pair_objects}
        Q_st = {pair.name: [declare_stream() for _ in range(len(multiplexed_sensors_by_dot_pair[pair.name].batch()))] for pair in dot_pair_objects}
        I = {pair.name: [declare(fixed) for _ in range(len(multiplexed_sensors_by_dot_pair[pair.name].batch()))] for pair in dot_pair_objects}
        Q = {pair.name: [declare(fixed) for _ in range(len(multiplexed_sensors_by_dot_pair[pair.name].batch()))] for pair in dot_pair_objects}

        with for_(n, 0, n<node.parameters.num_shots, n+1): 
            save(n, n_st)

            # Perform them all sequentially for now. Can add footprint batching later
            for dot_pair in dot_pair_objects: 
                
                # ---------------------------------------------------------
                # Step 1a: Empty - step to empty point (fixed duration)
                # ---------------------------------------------------------
                dot_pair.empty()
                # Requires the dot pair object to have the empty macro, in addition to the qubits
                # Equivalet step to the lvl_init

                # ---------------------------------------------------------
                # Step 2: Initialize - load electron into dots (fixed duration)
                # ---------------------------------------------------------
                dot_pair.initialize()
                # Requires the dot pair object to have the initialize macro, in addition to the qubits

                # ---------------------------------------------------------
                # Step 3: Measure
                # ---------------------------------------------------------
                # No macro used here, since the user likely has no measure macros defined (the point of this node)
                
                # First ramp to the fixed detuning point
                dot_pair.ramp_to_detuning(
                    node.parameters.detuning, 
                    ramp_duration = node.parameters.ramp_duration
                )

                align()
                
                # And then explicitly measure. 
                # The measuring will be in batches of multiplexed sensors. Each dot_pair will have a list of SensorDot objects. 
                # For each multiplexable batch, we have a single measurement saved to a single stream. 

                for i, batch in enumerate(multiplexed_sensors_by_dot_pair[dot_pair.name].batch()):
                    for sensor in batch.values():
                        # Select the resonator tied to the sensor
                        rr = sensor.readout_resonator
                        # Measure using said resonator
                        rr.measure("readout", qua_vars=(I[dot_pair.name][i], Q[dot_pair.name][i]))
                        # Post-measurement wait (Optional)
                        rr.wait(500)

                    # Save data
                    save(I[dot_pair.name][i], I_st[dot_pair.name][i])
                    save(Q[dot_pair.name][i], Q_st[dot_pair.name][i])
                
                align()
                # Apply the compensation pulse via the voltage sequence
                dot_pair.voltage_sequence.apply_compensation_pulse()

        with stream_processing():
            n_st.save("n")

            for pair in dot_pair_objects:
                for i in range(len(multiplexed_sensors_by_dot_pair[pair.name].batch())):
                    I_st[pair.name][i].average().save(f"I_{pair.name}_sensor_{i}")
                    Q_st[pair.name][i].average().save(f"Q_{pair.name}_sensor_{i}")




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
    # Get the active sensors from the loaded node parameters
    node.namespace["sensors"] = get_sensors(node)


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
    """Update the relevant parameters if the sensor data analysis was successful."""

    with node.record_state_updates():
        # This is a characterization measurement and typically does not update state parameters.
        # If needed in the future, the identified PSB region coordinates could be stored.
        # Example of potential state update (commented out):
        # for qubit_pair in node.namespace["qubit_pairs"]:
        #     if not node.results["fit_results"][qubit_pair.name]["success"]:
        #         continue
        #     # Update PSB region coordinates if needed
        # TODO: update threshold and rotation angle here?
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
