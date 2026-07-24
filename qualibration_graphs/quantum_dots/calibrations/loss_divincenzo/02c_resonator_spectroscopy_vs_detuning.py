# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.resonator_spectroscopy_vs_detuning import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS DETUNING
This sequence involves measuring the resonator by sending a readout pulse and
demodulating the signals to extract the 'I' and 'Q' quadratures for all resonators
simultaneously. This is done across various readout frequencies and detuning values.
Based on the results, one can then adjust the readout frequency, choosing a
readout frequency value which shows the strongest signal.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a_resonator_spectroscopy.py).
    - Having calibrated the resonator power (node 02b_resonator_spectroscopy_vs_power.py).
    - Having identified a suitable detuning transition.

Datasets:
    - ``ds_raw``: untouched I/Q fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed sweeps plus analysis outputs (derived fields and per-sensor summary
      coordinates). Used by ``plot_data``.
    - ``fit_results``: compact per-sensor calibration dict (``FitParameters`` serialized with
      ``asdict``). Used by logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][<sensor>]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``resonator_frequency`` [Hz]: absolute readout frequency at the PCA signal peak.
    - ``frequency_shift`` [Hz]: fitted readout frequency offset at the peak.
    - ``optimal_detuning`` [V]: QD pair gate voltage at the PCA signal peak (reported only; not written to QuAM).
    - ``peak_pca_signal``: PCA signal amplitude at the peak (arb. units).

Figures (``node.results["figures"]``):
    - ``"amplitude"``: IQ background with PCA signal overlay and peak marker for each sensor.

State update:
    - The readout frequency which maximises the PCA signal: sensor.readout_resonator.intermediate_frequency.
    - ``optimal_detuning`` is stored in ``fit_results`` for reference; apply the gate voltage separately in
      subsequent experiments or via your dot-pair operating-point workflow.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02c_resonator_spectroscopy_vs_detuning",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.quantum_dot_pair = "virtual_dot_1_virtual_dot_2_pair"
    # node.parameters.sensor_names = ["virtual_sensor_1"]
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 2D frequency × detuning sweep and the QUA pulse sequence."""

    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────
    n_avg = node.parameters.num_shots  # number of repeated measurements to average

    # Sensors used for readout (each has its own resonator)
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # The QD pair whose gate voltage we sweep (e.g. barrier/plunger combination)
    node.namespace["quantum_dot_pair"] = qd_pair = node.machine.get_component(
        node.parameters.quantum_dot_pair
    )

    # Gate-voltage detuning axis (physical detuning of the dot pair, in Volts)
    detuning_min = node.parameters.detuning_start
    detuning_max = node.parameters.detuning_stop
    detuning_step = node.parameters.detuning_step
    det_array = np.arange(detuning_min, detuning_max, detuning_step)

    # Readout-frequency axis: offsets relative to each sensor's calibrated IF (Hz)
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Metadata for data fetching: tells the system how to label the saved I/Q arrays
    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "frequency": xr.DataArray(dfs, attrs={"long_name": "Frequency Detuning", "units": "Hz"}),
        "detuning": xr.DataArray(det_array, attrs={"long_name": "Quantum Dot Pair Detuning", "units": "V"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        #   I, Q, I_st, Q_st, n, n_st — same role as in 02a
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

        # Real-time sweep variables (updated inside the loops):
        det = declare(fixed)  # current QD-pair detuning voltage [V]
        df = declare(int)     # current readout frequency offset [Hz]

        # Process sensors in batches if multiplexing is needed
        for multiplexed_sensors in sensors.batch():

            # Wait until all channels in this batch are ready (synchronization barrier)
            align()

            # ── OUTERMOST LOOP: repeat the full 2D map n_avg times ───────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # send current shot index to PC (for progress bar)

                # ── MIDDLE LOOP: sweep readout frequency ───────────────────
                with for_(*from_array(df, dfs)):

                    # Retune readout tone to IF + df (same as 02a)
                    for i, sensor in multiplexed_sensors.items():
                        rr = sensor.readout_resonator
                        rr.update_frequency(df + rr.intermediate_frequency)

                    # ── INNER LOOP: sweep QD-pair detuning voltage ─────────
                    with for_(*from_array(det, det_array)):

                        align()  # sync before changing gate voltages

                        # Move the QD pair to detuning voltage `det`, hold for `point_duration`
                        qd_pair.voltage_sequence.step_to_voltages(
                            {qd_pair.name: det}, duration=node.parameters.point_duration
                        )

                        align()  # sync before readout pulses

                        # Account for any "sticky" gate pulse duration in the voltage sequence
                        # (used by the voltage engine to track timing of long pulses)
                        readout_pulse_length = sensor.readout_resonator.operations[
                            "readout" + f"_{qd_pair.name}"
                        ].length
                        qd_pair.voltage_sequence.track_sticky_duration(readout_pulse_length)

                        # Send readout pulse and demodulate into I/Q for each sensor
                        for i, sensor in multiplexed_sensors.items():
                            rr = sensor.readout_resonator
                            readout_pulse_name = "readout" + f"_{qd_pair.name}"

                            # Measure = play readout pulse + integrate I and Q
                            rr.measure(readout_pulse_name, qua_vars=(I[i], Q[i]))

                            # Store this (frequency, detuning) point's I/Q in the stream buffer
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    align()  # sync before ramping gates back

                    # Return QD gates to zero after finishing the detuning slice at this frequency
                    qd_pair.voltage_sequence.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")  # expose shot counter as "n" in the fetched dataset

            for i in range(num_sensors):
                # Each save() appends one (frequency, detuning) point.
                # .buffer(len(det_array))  → group points along detuning axis
                # .buffer(len(dfs))       → group those groups along frequency axis
                # .average()              → average over all shots (n_avg repetitions)
                I_st[i].buffer(len(det_array)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(det_array)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(data_fetcher.get("n", 0), node.parameters.num_shots, start_time=data_fetcher.t_start)
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated resonator spectroscopy vs detuning data so the full analysis pipeline can run without hardware."""
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
    node.namespace["quantum_dot_pair"] = node.machine.get_component(node.parameters.quantum_dot_pair)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process ``ds_raw``, fit the data, and store processed data plus fit outputs in ``ds_fit``."""
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        sensor_name: ("successful" if fit_result["success"] else "failed")
        for sensor_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(node.results["ds_fit"], node.namespace["sensors"])
    if not node.modes.external:
        plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    with node.record_state_updates():
        for s in node.namespace["sensors"]:
            if node.outcomes[s.name] == "failed":
                continue

            # Update the readout frequency
            s.readout_resonator.intermediate_frequency += node.results["fit_results"][s.name]["frequency_shift"]
            # TODO: any reason not to update the detuning here?

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
