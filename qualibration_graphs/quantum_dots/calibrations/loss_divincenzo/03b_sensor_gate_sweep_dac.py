# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict
import time

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.sensor_dot import VirtualDCSetParameters as Parameters
from calibration_utils.sensor_dot import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    generate_simulated_dataset,
    plot_all,
)
from calibration_utils.common_utils.experiment import get_sensors
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        CHARGE SENSOR GATE SWEEP with the DAC using the VirtualDCSet

The measurement performs a voltage sweep across a specified range with configurable step size. The voltage is stepped by your 
external DAC, which must be configured in the execute_qua_program run_action first. You can optionally step to the measure point
of the associated qubit_pair upon measurement, so that the un-virtualized sensor gate DAC sweep will be at the most sensitive
point at the qubit_pair's measure point. 

At each voltage point, a readout pulse is sent to the resonator coupled to the sensor dot, and the reflected signal is 
demodulated and recorded. A global average is performed (averaging on the most outer loop) and the data is extracted while 
the program is running to display the sensor response with increasing SNR.

Prerequisites:
    - Connect the AC line of the bias-tee connected to the sensor dot to one OPX channel.
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.

State update:
    - Update the optimal voltage bias of each sensor dot.
"""


node = QualibrationNode[Parameters, Quam](
    name="03b_sensor_gate_sweep_dac", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.use_simulated_data
)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Get the relevant sensor dots rom the node
    node.namespace["sensors"] = sensors = get_sensors(node)

    num_sensors = len(sensors)
    
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots

    # The voltage offset sweep
    bias_offsets = np.arange(
        node.parameters.offset_min,
        node.parameters.offset_max,
        node.parameters.offset_step,
    )
    node.namespace["sensor_axis_values"] = bias_offsets

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(
            bias_offsets, attrs={"long_name": "Sensor bias offset", "units": "V"}
        ),
    }
    if node.parameters.qubit_pair_to_step is not None:
        dot_pair = [node.machine.get_component(qp).quantum_dot_pair for qp in node.parameters.qubit_pair_to_step][0]

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_sensors
        )

        sensor_idx = declare(int)
        for multiplexed_sensors in sensors.batch():
            sequences_in_batch = {
                sensor.voltage_sequence.gate_set.id: sensor.voltage_sequence
                for sensor in multiplexed_sensors.values()
            }

            with for_(sensor_idx, 0, sensor_idx < len(bias_offsets), sensor_idx + 1): 
                pause()
                # During pause, will step the DAC

                wait(node.parameters.duration_after_step)

                align()
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    align()
                    if node.parameters.qubit_pair_to_step is not None: 
                        dot_pair.voltage_sequence.step_to_point(f"{dot_pair.name}_measure")

                        # Track the sticky duration through the maximum readout pulse in the multiplexed batch
                        dot_pair.voltage_sequence.track_sticky_duration(
                            int(max(k.readout_resonator.operations["readout"].length for k in multiplexed_sensors.values()))
                        )

                    for i, sensor in multiplexed_sensors.items():
                        align()
                        sensor.readout_resonator.measure(
                            "readout", qua_vars=(I[i], Q[i])
                        )
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()
                    if node.parameters.qubit_pair_to_step is not None: 
                        for seq in sequences_in_batch.values():
                            seq.apply_compensation_pulse(max_voltage = node.parameters.max_compensation_voltage, go_to_zero = True, return_to_zero = True)


        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(len(bias_offsets)).save(f"I{i + 1}")
                Q_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(len(bias_offsets)).save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()

    # Get the config from the machine
    config = node.machine.generate_config()

    # Time to step the external DAC. This will execute when the program is paused.
    # Use the same sensor set as the compiled QUA program.
    sensor_names = node.namespace["sensors"].get_names()
    for s in sensor_names:
        gate_set_id = node.machine.sensor_dots[s].voltage_sequence.gate_set.name
        node.namespace[f"{s}_dac_offset"] = node.machine.virtual_dc_sets[
            gate_set_id
        ].get_voltage(s, requery=True)

    try:
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

            for multiplexed_sensors in node.namespace["sensors"].batch():
                axis_values = node.namespace["sensor_axis_values"]
                for i, y_value in enumerate(axis_values):
                    while not job.is_paused():
                        time.sleep(0.1)

                    voltages_by_gate_set = {}
                    for _sensor_idx, sensor in multiplexed_sensors.items():
                        gate_set_id = (
                            node.machine.sensor_dots[sensor.name]
                            .voltage_sequence.gate_set.name
                        )
                        value_to_play = node.namespace[
                            f"{sensor.name}_dac_offset"
                        ] + y_value
                        voltages_by_gate_set.setdefault(gate_set_id, {})[
                            sensor.name
                        ] = value_to_play

                        pct = 100 * i / len(axis_values)
                        node.log(
                            f"Applying {value_to_play: .4f} to the channel {sensor.name}: ({pct: .1f} %)"
                        )

                    for gate_set_id, voltages_dict in voltages_by_gate_set.items():
                        node.machine.virtual_dc_sets[gate_set_id].set_voltages(
                            voltages_dict
                        )

                    time.sleep(node.parameters.dac_settling_time_s)
                    job.resume()

            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )

            node.log(job.execution_report())
            node.results["ds_raw"] = dataset
    finally:
        node.log("Re-applying initial offsets.")
        for s in sensor_names:
            gate_set_id = node.machine.sensor_dots[s].voltage_sequence.gate_set.name
            node.machine.virtual_dc_sets[gate_set_id].set_voltages(
                {s: node.namespace[f"{s}_dac_offset"]}
            )
        
# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated Coulomb peak data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


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

#
# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Process the raw dataset into derived signals for analysis and plotting."""
    node.results["ds_processed"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    ds_processed = node.results.get("ds_processed")
    if ds_processed is None:
        ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        sensor_name: ("successful" if fit_result["success"] else "failed")
        for sensor_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"], node.namespace["sensors"]
    )
    if not node.modes.external:
        plt.show()
    # ### Annotations can come later, once calibration_utils is done
    # annotate_node_figures(node)

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""

    with node.record_state_updates():
        for sensor in node.namespace["sensors"]:
            if node.outcomes.get(sensor.name) != "successful":
                continue

            optimal_offset = node.results["fit_results"][sensor.name]["optimal_bias"]
            dac_optimal_value = optimal_offset + node.namespace[f"{sensor.name}_dac_offset"]

            node.log(f"Optimal offset is {dac_optimal_value}. Setting this now.")

            gate_set_id = node.machine.sensor_dots[sensor.name].voltage_sequence.gate_set.name
            node.machine.virtual_dc_sets[gate_set_id].set_voltages({sensor.name: dac_optimal_value})

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
