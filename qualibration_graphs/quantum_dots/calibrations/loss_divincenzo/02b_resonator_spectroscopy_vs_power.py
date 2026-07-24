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
from calibration_utils.resonator_spectroscopy_vs_power import (
    Parameters,
    process_raw_dataset,
    plot_raw_data_with_fit,
    log_fitted_results,
    fit_raw_data,
    generate_simulated_dataset,
)
from quam_builder.tools.power_tools import calculate_voltage_scaling_factor
from quam_builder.architecture.quantum_dots.components import ReadoutResonatorSingle
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

from calibration_utils.common_utils.experiment import get_sensors

# %% {Node initialisation}
description = """
        RESONATOR SPECTROSCOPY VERSUS READOUT POWER
This sequence involves measuring the resonator by sending a readout pulse and
demodulating the signals to extract the 'I' and 'Q' quadratures for all resonators
simultaneously. This is done across various readout frequencies and amplitudes.
Based on the results, one can then adjust the readout amplitude, choosing a
readout amplitude value just before the observed frequency splitting.

Prerequisites:
    - Having calibrated the resonator frequency (node 02a_resonator_spectroscopy.py).
    - Having instantiated a starting readout amplitude.

Results (``node.results["fit_results"][<sensor>]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``resonator_frequency`` [Hz]: absolute readout frequency at ``optimal_power``.
    - ``frequency_shift`` [Hz]: fitted readout frequency offset at ``optimal_power``.
    - ``optimal_power`` [dBm]: readout power just below the onset of frequency splitting.

State update:
    - The readout power: sensor.readout_resonator.set_output_power()
    - The readout frequency for the optimal readout power.
"""


# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="02b_resonator_spectroscopy_vs_power",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.sensor_names = ["virtual_sensor_1", "virtual_sensor_2"]
    # node.parameters.num_shots = 1
    # node.parameters.frequency_span_in_mhz = 50
    # node.parameters.frequency_step_in_mhz = 0.1
    # node.parameters.max_power_dbm = -25
    # node.parameters.min_power_dbm = -60
    # node.parameters.num_power_points = 100
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the 2D frequency × power sweep and the QUA pulse sequence."""

    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    n_avg = node.parameters.num_shots  # repetitions averaged at each (frequency, power) point

    # Readout power axis [dBm] — the quantity we ultimately want to calibrate
    power_dbm = np.linspace(
        node.parameters.min_power_dbm,
        node.parameters.max_power_dbm,
        node.parameters.num_power_points,
    )

    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # ── Prepare QuAM for the power sweep (Python side, before QUA runs) ──
    #
    # Strategy: configure the readout pulse at MAX power in QuAM, then in QUA
    # scale it down with `amplitude_scale` to reach lower powers.
    # tracked_updates remembers these edits so they can be reverted in update_state.
    node.namespace["tracked_resonators"] = []
    for i, sensor in enumerate(sensors):
        with tracked_updates(sensor.readout_resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            if isinstance(resonator._obj, ReadoutResonatorSingle):
                # Direct amplitude on a simple readout line: set voltage for max_power_dbm
                base_amplitude = u.dBm2volts(node.parameters.max_power_dbm, Z=50)
                resonator.operations["readout"].amplitude = base_amplitude
            else:
                # More general resonator model: set output power via QuAM helper
                resonator.set_output_power(
                    power_in_dbm=node.parameters.max_power_dbm,
                    max_amplitude=node.parameters.max_amp,
                )
            node.namespace["tracked_resonators"].append(resonator)

    # Dimensionless scale factors applied in QUA to the max-amplitude pulse.
    # Geometric spacing: equal steps in log(power) from min_power to max_power.
    # amp_min corresponds to min_power_dbm; 1.0 corresponds to max_power_dbm.
    amp_min = calculate_voltage_scaling_factor(
        node.parameters.max_power_dbm, node.parameters.min_power_dbm
    )
    amps = np.geomspace(amp_min, 1, node.parameters.num_power_points)

    # Readout-frequency axis: offsets relative to calibrated IF [Hz] (same as 02a)
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span / 2, +span / 2, step)

    # Metadata for data fetching
    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "frequency_detuning": xr.DataArray(
            dfs,
            attrs={"long_name": "readout frequency detuning from IF", "units": "Hz"},
        ),
        "power": xr.DataArray(power_dbm, attrs={"long_name": "readout power", "units": "dBm"}),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:

        # Real-time variables:
        #   I, Q, I_st, Q_st, n, n_st — same role as in 02a
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()

        a = declare(fixed)   # current amplitude scale factor (0…1 relative to max power)
        df = declare(int)    # current readout frequency offset [Hz]

        for multiplexed_sensors in sensors.batch():
            align()

            # ── OUTERMOST LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                # ── MIDDLE LOOP: sweep readout frequency ─────────────────
                with for_(*from_array(df, dfs)):

                    for i, sensor in multiplexed_sensors.items():
                        rr = sensor.readout_resonator

                        # Retune readout tone to IF + df (same as 02a)
                        update_frequency(rr.name, df + rr.intermediate_frequency)

                        # ── INNER LOOP: sweep readout power ───────────────
                        # `a` multiplies the pulse amplitude configured at max power.
                        # Lower a → lower readout power.
                        with for_each_(a, amps):

                            # Readout pulse at scaled amplitude; result → I[i], Q[i]
                            rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)

                            # Let the resonator ring down before the next point
                            rr.wait(1000 * u.ns)

                            # Store this (frequency, power) point
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        # ── Post-processing on the OPX ────────────────────────────────────
        with stream_processing():
            n_st.save("n")

            for i in range(num_sensors):
                # Save order per stream: for each df, sweep all power values.
                # .buffer(len(amps))  → inner axis = power
                # .buffer(len(dfs))   → outer axis = frequency
                # .average()         → average over shots
                # Result: 2D map I(frequency, power) per sensor
                I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


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
    """Generate simulated resonator spectroscopy vs power data so the full analysis pipeline can run without hardware."""
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
    """Plot the raw and fitted data."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["sensors"])
    plt.show()
    node.results["figures"] = {"amplitude": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""
    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    # Update the state
    with node.record_state_updates():
        for s in node.namespace["sensors"]:
            if node.outcomes[s.name] == "failed":
                continue

            # Update the readout power
            for op in s.readout_resonator.operations:
                if not op.startswith("readout"):
                    continue
                if isinstance(s.readout_resonator._obj, ReadoutResonatorSingle):
                    u = unit(coerce_to_integer=True)
                    s.readout_resonator.operations[op].amplitude = u.dBm2volts(
                        node.results["fit_results"][s.name]["optimal_power"]
                    )
                else:
                    s.readout_resonator.set_output_power(
                        power_in_dbm=node.results["fit_results"][s.name]["optimal_power"],
                        max_amplitude=node.parameters.max_amp,
                        operation=op,
                    )
            # Update the readout frequency
            s.readout_resonator.intermediate_frequency += node.results["fit_results"][s.name]["frequency_shift"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
