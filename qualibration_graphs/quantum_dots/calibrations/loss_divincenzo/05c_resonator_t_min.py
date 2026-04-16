# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
from itertools import combinations

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.resonator_t_min import (
    Parameters,
    # process_raw_dataset,
    # fit_raw_data,
    # log_fitted_results,
    # plot_raw_amplitude_with_fit,
    # plot_raw_phase,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        RESONATOR t_min CHARACTERISATION
This measurement aims to characterise the minimum integration time necessary to achieve SNR = 1 for readout. In this node,
a double-quantum-dot is ramped from charge configuration (1,1) to state (0,2), in order to characterise the charge state readout
fidelity. The aim is to characterise the integration time necessary to reach SNR = 1, for use with PSB readout. The measured IQ
blobs in the IQ state distribution map is analysed, and the SNR is extracted through the relevant axis.


Prerequisites:
    - Having calibrated the resonator to the most sensitive frequency.
    - Having calibrated the relevant sensor dots.
    - Having identified the (1,1) (operation) and (0,2) (readout) points on your charge stability diagram.

State update:
    - The integration time of the measure macro.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="05c_resonator_spectroscopy",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.sensors = ["q1", "q2"]
    node.parameters.sensor_names = ["virtual_sensor_1"]
    node.parameters.quantum_dots = ["virtual_dot_1", "virtual_dot_2"]

    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    # Get the relevant sensor dots rom the node
    node.namespace["sensors"] = sensors = get_sensors(node)
    node.namespace["quantum_dots"] = quantum_dots = [
        node.machine.get_component(k) for k in node.parameters.quantum_dots
    ]  # Find the quantum_dot components

    if len(quantum_dots) < 2:
        raise ValueError(f"At least 2 Quantum Dots required. Received {len(quantum_dots)}")

    quantum_dot_pair_names = [
        pair
        for dot1, dot2 in combinations(node.parameters.quantum_dots, 2)
        if (pair := node.machine.find_quantum_dot_pair(dot1, dot2)) is not None
    ]
    node.namespace["quantum_dot_pairs"] = quantum_dot_pairs = [
        node.machine.get_component(k) for k in quantum_dot_pair_names
    ]

    for dp in quantum_dot_pairs:
        dp.add_point(point_name="empty", voltages={}, duration=1000)
        dp.add_point(point_name="initialize", voltages={}, duration=1000)
        dp.add_point(point_name="measure", voltages={}, duration=1000)

    num_sensors = len(sensors)
    num_pairs = len(quantum_dot_pairs)

    # Extract the sweep parameters and axes from the node parameters
    n_reps = node.parameters.num_shots

    integrations_times = np.arange(
        node.parameters.integration_time_start,
        node.parameters.integration_time_stop,
        node.parameters.integration_time_step,
    )
    samples_per_chunk = node.parameters.integration_time_step // 4
    array_size = len(integrations_times)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        # "sensors": xr.DataArray(sensors.get_names()),
        "quantum_dot_pairs": xr.DataArray([qdp.name for qdp in quantum_dot_pairs]),
        "repetition": xr.DataArray(np.arange(n_reps)),
        "integration_time": xr.DataArray(
            np.arange(1, array_size + 1) * samples_per_chunk * 4,
            attrs={"long_name": "integration time", "units": "ns"},
        ),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:

        I_st_11 = {sensor.name: {dp.name: declare_stream() for dp in quantum_dot_pairs} for sensor in sensors}
        Q_st_11 = {sensor.name: {dp.name: declare_stream() for dp in quantum_dot_pairs} for sensor in sensors}
        I_st_02 = {sensor.name: {dp.name: declare_stream() for dp in quantum_dot_pairs} for sensor in sensors}
        Q_st_02 = {sensor.name: {dp.name: declare_stream() for dp in quantum_dot_pairs} for sensor in sensors}

        n = declare(int)
        n_st = declare_stream()

        idx = declare(int)

        for dot_pair in quantum_dot_pairs:
            align()
            with for_(n, 0, n < n_reps, n + 1):
                save(n, n_st)

                # ---------------------------------------------------------
                # Step 1a: Empty - step to empty point (fixed duration)
                # ---------------------------------------------------------
                dot_pair.empty()
                # Requires the dot pair object to have the empty macro, in addition to the qubits
                # Equivalet step to the lvl_init

                # ---------------------------------------------------------
                # Step 2a: Initialize - load electron into dots (fixed duration)
                # ---------------------------------------------------------
                dot_pair.initialize()
                # Requires the dot pair object to have the initialize macro, in addition to the qubits

                for batch in sensors.batch():
                    for batch_idx, s in batch.items():
                        I_11, Q_11 = s.readout_resonator.measure_accumulated(
                            "readout",
                            segment_length=samples_per_chunk,
                        )
                        with for_(idx, 0, idx < array_size, idx + 1):
                            save(I_11[idx], I_st_11[s.name][dot_pair.name])
                            save(Q_11[idx], Q_st_11[s.name][dot_pair.name])

                # ---------------------------------------------------------
                # Step 2b: Wait - ensure that it is a singlet state.
                # ---------------------------------------------------------
                dot_pair.voltage_sequence.step_to_voltages(voltages={}, duration=node.parameters.wait_time)

                # ---------------------------------------------------------
                # Step 3: Measure
                # ---------------------------------------------------------

                dot_pair.voltage_sequence.step_to_point(f"{dot_pair.name}_measure")

                for batch in sensors.batch():
                    for batch_idx, s in batch.items():
                        I_02, Q_02 = s.readout_resonator.measure_accumulated(
                            "readout",
                            segment_length=samples_per_chunk,
                        )
                        with for_(idx, 0, idx < array_size, idx + 1):
                            save(I_02[idx], I_st_02[s.name][dot_pair.name])
                            save(Q_02[idx], Q_st_02[s.name][dot_pair.name])

        with stream_processing():
            n_st.save("n")
            for batch in sensors.batch():
                for idx, s in batch.items():
                    for dp in quantum_dot_pairs:
                        I_st_11[s.name][dp.name].buffer(array_size).buffer(n_reps).save(f"I_11_{dp.name}_{s.name}")
                        Q_st_11[s.name][dp.name].buffer(array_size).buffer(n_reps).save(f"Q_11_{dp.name}_{s.name}")
                        I_st_02[s.name][dp.name].buffer(array_size).buffer(n_reps).save(f"I_02_{dp.name}_{s.name}")
                        Q_st_02[s.name][dp.name].buffer(array_size).buffer(n_reps).save(f"Q_02_{dp.name}_{s.name}")


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
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["sensors"] = get_sensors(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    pass


# %% {Plot_data}
# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    pass


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor_name data analysis was successful."""
    with node.record_state_updates():
        pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
