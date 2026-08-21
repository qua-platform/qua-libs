# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array

from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName
from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.sensor_dot import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    generate_simulated_dataset,
    plot_all,
    apply_compensation_pulse,
    refresh_voltage_sequences,
)
from calibration_utils.common_utils.experiment import get_sensors
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        SENSOR DOT GATE SWEEP (OPX)
This sequence sweeps the sensor-dot gate bias using the OPX (AC line of a bias-tee) and measures the sensor response via
RF reflectometry. At each bias offset, a readout pulse is sent to the sensor resonator and the reflected signal is
demodulated into the 'I' and 'Q' quadratures. The sweep is averaged to improve SNR and post-processed to extract the
recommended operating point (maximum-sensitivity bias).

Prerequisites:
    - Connect the AC line of the bias-tee connected to the sensor dot to one OPX channel.
    - QUAM initialised (e.g. ``quam_config/populate_quam_state_*.py``).
    - SensorDot readout resonators calibrated (time-of-flight/offsets/gains + readout frequency, amplitude, duration).
    - (Recommended) Use an external DAC to hold a DC offset while the OPX performs fast sweeps.

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
    - Adds/updates the SensorDot ``MEASURE`` voltage point using ``optimal_bias`` for each successful sensor.
"""


node = QualibrationNode[Parameters, Quam](
    name="03a_sensor_gate_sweep_opx", description=description, parameters=Parameters()
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
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 1D sensor sweep and the QUA pulse sequence."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Get the relevant sensor dots from the node
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots  # number of repetitions averaged at each sensor plunger voltage
    ramp_duration = node.parameters.ramp_duration  # duration of the ramp to the next plunger voltage

    # The voltage bias offset - set of voltages to apply on the sensor's plunger gate
    # E.g. offset_min=0 & offset_max=0.1 → sweep from Vg=0V to Vg=+0.1V
    bias_offsets = np.arange(
        node.parameters.offset_min,
        node.parameters.offset_max,
        node.parameters.offset_step,
    )

    # Metadata for data fetching: labels the saved I/Q arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(bias_offsets, attrs={"long_name": "Sensor bias offset", "units": "V"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Allocate real-time variables on the OPX:
        #   I[i], Q[i]   : demodulated quadratures for sensor i
        #   I_st[i], Q_st[i] : buffers collecting I/Q before transfer to PC
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        # Real-time variable holding the plunger gate voltage
        offset = declare(fixed)

        # If several sensors share the same AWG resources, they are grouped into batches
        for multiplexed_sensors in sensors.batch():
            refresh_voltage_sequences(node, multiplexed_sensors)

            align()  # sync all channels in this batch before starting

            # ── OUTER LOOP: repeat the full sweep n_avg times ──
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # tell the PC which shot we are on

                # ── INNER LOOP: sweep sensor plunger gate voltage ──────────
                with for_(*from_array(offset, bias_offsets)):
                    for i, sensor in multiplexed_sensors.items():
                        # Extract the readout length so that the plunger voltage is maintained during readout
                        readout_len = sensor.readout_resonator.operations["readout"].length

                        align()

                        # Ramp the plunger gate voltage to the correct coordinate and hold the voltage (duration) to include the readout time
                        sensor.ramp_to_voltages(
                            {sensor.name: offset},
                            duration=readout_len + node.parameters.duration_after_step,
                            ramp_duration=ramp_duration,
                        )

                        # While the ramp & any optional duration_after_step, the resonator is idle
                        sensor.readout_resonator.wait((ramp_duration + node.parameters.duration_after_step) // 4)

                        # Play the "readout" pulse and integrate I/Q into I[i], Q[i]
                        sensor.readout_resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # Append this voltage point's I/Q to the stream buffer
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

                # At the end of each 1D sweep, play a compensation pulse to account for any charge build-up in the bias tee
                apply_compensation_pulse(multiplexed_sensors, node.parameters.max_compensation_voltage)

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset
            for i in range(num_sensors):
                # Each save() above is one voltage point.
                # .buffer(len(bias_offsets)) : group points along the plunger gate voltage axis
                # .average()        : average over all shots (n_avg repetitions)
                # Result: 1D trace I(bias_offsets), Q(bias_offsets) per sensor
                I_st[i].buffer(len(bias_offsets)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(bias_offsets)).average().save(f"Q{i + 1}")


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
        # "samples": samples,
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

            # Add point to the VirtualGateSet instead. Associate with the relevant SensorDot
            # Optionally step the DAC to this voltage?

            sensor.add_point(
                VoltagePointName.MEASURE,
                voltages={sensor.name: optimal_offset},
                duration=sensor.readout_resonator.operations["readout"].length,
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
