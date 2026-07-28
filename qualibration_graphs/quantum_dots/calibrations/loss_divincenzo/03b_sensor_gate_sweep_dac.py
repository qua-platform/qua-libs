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
        SENSOR DOT GATE SWEEP (DAC / VirtualDCSet)

This sequence performs a 1D sweep of the sensor-dot gate bias using an external DAC (via the VirtualDCSet). The OPX runs the
readout and pauses between points while Python steps the DAC to the next bias value, then resumes the QUA program.
Optionally, the associated qubit-pair can be stepped to its ``measure`` point at each DAC step to calibrate the sensor response
at a physically relevant operating point.

At each bias value, a readout pulse is sent to the sensor resonator and the reflected signal is demodulated into the 'I' and 'Q'
quadratures. The sweep is averaged to improve SNR and post-processed to extract a recommended operating point
(maximum-sensitivity bias).

Prerequisites:
    - Connect the AC line of the bias-tee connected to the sensor dot to one OPX channel.
    - External DAC configured and reachable through the VirtualDCSet in QUAM.
    - QUAM initialised (e.g. ``quam_config/populate_quam_state_*.py``).
    - SensorDot readout resonators calibrated (time-of-flight/offsets/gains + readout frequency, amplitude, duration).

Datasets:
    - ``ds_raw``: untouched I/Q fetched from the OPX (never modified after acquisition).
    - ``ds_processed``: ``ds_raw`` plus derived amplitude/phase (used by fitting and plotting).
    - ``ds_fit``: processed sweeps plus analysis outputs (derived fields and per-sensor summary coordinates). Used by
      ``plot_data``.
    - ``fit_results``: compact per-sensor calibration dict (``FitParameters`` serialized with ``asdict``). Used by
      logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][<sensor>]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``optimal_bias`` [V]: recommended operating bias (Lorentzian inflection point; side set by ``peak_fit_side``).
    - ``peak_position`` [V]: detected Coulomb-peak position (feature detection).
    - ``lorentzian_gamma`` [V]: Lorentzian FWHM (linewidth) of the fitted peak.
    - ``max_gradient_bias`` [V]: bias at maximum slope (closest sampled point to ``optimal_bias``).

Figures (``node.results["figures"]``):
    - ``"phase"``: phase vs bias offset for each sensor.
    - ``"amplitude_gradient"``: amplitude with Lorentzian fit overlay and a marker at the max-gradient point.

State update:
    - For each successful sensor, sets the corresponding VirtualDCSet channel to the recommended gate voltage
      (initial DAC offset + ``optimal_bias``).
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
node.machine = Quam.load("/Users/kalidu_laptop/merge_libs/quam_state")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D sensor sweep and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Get the relevant sensor dots rom the node
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots  # number of repetitions averaged at each sensor plunger voltage

    # The voltage bias offset - set of voltages to apply on the sensor's plunger gate
    # E.g. offset_min=0 & offset_max=0.1 → sweep from Vg=0V to Vg=+0.1V
    bias_offsets = np.arange(
        node.parameters.offset_min,
        node.parameters.offset_max,
        node.parameters.offset_step,
    )
    # Store the values in a simple namespace, so they are accessible later
    node.namespace["sensor_axis_values"] = bias_offsets

    # Metadata for data fetching: labels the saved I/Q arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(bias_offsets, attrs={"long_name": "Sensor bias offset", "units": "V"}),
    }

    # In-case you want to step along the detuning axis at each sensor dot point.
    # This is useful if you want to calibrate the sensor dot peak relative to the actual measure point
    if node.parameters.qubit_pair_to_step is not None:
        dot_pair = [node.machine.get_component(qp).quantum_dot_pair for qp in node.parameters.qubit_pair_to_step][0]

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I[i], Q[i]   : demodulated quadratures for sensor i
        #   I_st[i], Q_st[i] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)

        # Real-time variable to indicate the index along the sensor bias array
        sensor_idx = declare(int)

        # If several sensors share the same AWG resources, they are grouped into batches
        for multiplexed_sensors in sensors.batch():
            align()  # sync all channels in this batch before starting

            # Extract the VoltageSequence objects in this batch
            sequences_in_batch = {
                sensor.voltage_sequence.gate_set.id: sensor.voltage_sequence for sensor in multiplexed_sensors.values()
            }

            # ── OUTER LOOP: PAUSE the QUA program, and set the DAC voltage ──
            with for_(sensor_idx, 0, sensor_idx < len(bias_offsets), sensor_idx + 1):
                pause()
                # During pause, will step the DAC

                wait(node.parameters.duration_after_step)

                align()

                # ── INNER LOOP: repeat the measurement n_avg times ──────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # tell the PC which shot we are on
                    align()

                    # Optionally step a particular qubit pair to the readout point.
                    if node.parameters.qubit_pair_to_step is not None:
                        dot_pair.voltage_sequence.step_to_point(f"{dot_pair.name}_measure")

                        # TODO: Verify this logic
                        # Track the sticky duration through the maximum readout pulse in the multiplexed batch
                        dot_pair.voltage_sequence.track_sticky_duration(
                            int(
                                max(
                                    k.readout_resonator.operations["readout"].length
                                    for k in multiplexed_sensors.values()
                                )
                            )
                        )

                    for i, sensor in multiplexed_sensors.items():
                        align()
                        # Play the "readout" pulse and integrate I/Q into I[i], Q[i]
                        sensor.readout_resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # Append this voltage point's I/Q to the stream buffer
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

                    # At the end of each 1D sweep, play a compensation pulse to account for any charge build-up in the bias tee
                    # This is only necessary in this program if a qubit pair was stepped. Otherwise, skip
                    if node.parameters.qubit_pair_to_step is not None:
                        for seq in sequences_in_batch.values():
                            seq.apply_compensation_pulse(
                                max_voltage=node.parameters.max_compensation_voltage,
                                go_to_zero=True,
                                return_to_zero=True,
                            )

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for i in range(num_sensors):
                # Each save() above is one voltage point.
                # .buffer(len(bias_offsets)) : group points along the plunger gate voltage axis
                # .average()        : average over all shots (n_avg repetitions)
                # Result: 1D trace I(bias_offsets), Q(bias_offsets) per sensor
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
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()

    # Get the config from the machine
    config = node.machine.generate_config()

    # ── QUA PROGRAM PAUSED: Time to step the DAC ──────────

    # Use the same sensor set as the compiled QUA program.
    sensor_names = node.namespace["sensors"].get_names()

    # Extract the gate set IDs for each sensor, and measure their current voltage to store their original offsets
    for s in sensor_names:
        gate_set_id = node.machine.sensor_dots[s].voltage_sequence.gate_set.name
        node.namespace[f"{s}_dac_offset"] = node.machine.virtual_dc_sets[gate_set_id].get_voltage(s, requery=True)

    try:
        # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            # The job is stored in the node namespace to be reused in the fetching_data run_action
            node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])

            # Loop over the sensors in the same order as the QUA program
            for multiplexed_sensors in node.namespace["sensors"].batch():

                # The plunger gate voltage values
                axis_values = node.namespace["sensor_axis_values"]

                # ── LOOP: Apply the DAC voltage at each PAUSE ──────────
                for i, y_value in enumerate(axis_values):

                    # If the job has not hit the pause() command yet, then wait 100ms
                    while not job.is_paused():
                        time.sleep(0.1)

                    # Batch the voltages that are necessary to apply
                    voltages_by_gate_set = {}
                    for _sensor_idx, sensor in multiplexed_sensors.items():
                        gate_set_id = node.machine.sensor_dots[sensor.name].voltage_sequence.gate_set.name

                        # The value to apply is the current DAC offset + the sweep value
                        value_to_play = node.namespace[f"{sensor.name}_dac_offset"] + y_value

                        # Batch the voltage by gate set name. This is so that sensors can be stepped simultaneously
                        voltages_by_gate_set.setdefault(gate_set_id, {})[sensor.name] = value_to_play

                        # Log the percentage of this innermost loop
                        pct = 100 * i / len(axis_values)
                        node.log(f"Applying {value_to_play: .4f} to the channel {sensor.name}: ({pct: .1f} %)")

                    # Finally, set the voltages batched together.
                    for gate_set_id, voltages_dict in voltages_by_gate_set.items():
                        node.machine.virtual_dc_sets[gate_set_id].set_voltages(voltages_dict)

                    time.sleep(node.parameters.dac_settling_time_s)

                    # job.resume() allows the QUA program to continue
                    job.resume()

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
    finally:
        # At the end of all the loops, re-apply the original DAC offsets.
        node.log("Re-applying initial offsets.")
        for s in sensor_names:
            gate_set_id = node.machine.sensor_dots[s].voltage_sequence.gate_set.name
            node.machine.virtual_dc_sets[gate_set_id].set_voltages({s: node.namespace[f"{s}_dac_offset"]})


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
    node.results["figures"] = plot_all(node.results["ds_fit"], node.namespace["sensors"])
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
