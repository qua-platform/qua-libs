# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam

from calibration_utils.common_utils.experiment import get_sensors
from calibration_utils.time_of_flight_mw import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.runtime import simulate_and_plot

# %% {Node initialisation}
description = """
        TIME OF FLIGHT - MW FEM
 
This sequence involves sending a readout pulse and capturing the raw ADC traces.
The data undergoes post-processing to calibrate three distinct parameters:
    - Time of Flight: This represents the internal processing time and the propagation
      delay of the readout pulse. This value is utilized to offset the acquisition window relative
      to when the readout pulse is dispatched.
 
    - Analog Inputs Gain: If a signal is constrained by digitization or if it saturates
      the ADC, the variable gain of the OPX1000 MW-FEM analog input, ranging from 0 dB to 32 dB,
      can be modified to fit the signal within the ADC range of +/-0.5V. While the gain is not automatically 
      set as the state update of this node, this can be manually adjusted using the command 
      sensor.readout_resonator.opx_input.gain_db = int(___) for a ReadoutResonatorMW. 
 
Prerequisites:
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
 
Datasets:
    - ``ds_raw``: untouched raw ADC I/Q counts (``adcI``, ``adcQ``, ``adc_single_runI``,
      ``adc_single_runQ``) vs ``readout_time``, per sensor.
    - ``ds_fit``: volts-converted I/Q traces plus fitted delay/success fields used for state updates.
 
Results:
    - ``fit_results[sensor].tof_to_add``: additional time of flight to add [ns].
    - ``fit_results[sensor].success``: whether the fit met the success criteria.
 
Figures:
    - ``"single_run"``: single-shot ADC trace with fitted TOF overlay, per sensor.
    - ``"averaged_run"``: shot-averaged ADC trace with fitted TOF overlay, per sensor.
 
State update:
    - The time of flight: sensor.readout_resonator.time_of_flight
"""


node = QualibrationNode[Parameters, Quam](
    name="01b_time_of_flight_mw_fem", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.use_simulated_data = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Get the active sensors from the node and organize them by batches
    node.namespace["sensors"] = sensors = get_sensors(node)

    # Number of shots per sweep point
    n_avg = node.parameters.num_shots

    # Make some tracked changes - time of flight, readout amplitude, and readout length
    # Tracked changes store the original value, and can be reverted later
    node.namespace["tracked_resonators"] = []
    for s in sensors:
        # The resonator is an attribute of the sensor dot
        resonator = s.readout_resonator
        # Make temporary updates before running the program and revert at the end.
        with tracked_updates(resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
            if node.parameters.time_of_flight_in_ns is not None:
                resonator.time_of_flight = node.parameters.time_of_flight_in_ns
            if node.parameters.readout_length_in_ns is not None:
                resonator.operations["readout"].length = node.parameters.readout_length_in_ns
            if node.parameters.readout_amplitude_in_dBm is not None:
                resonator.set_output_power(node.parameters.readout_amplitude_in_dBm, operation="readout")

            # Populate the list of tracked resonators
            node.namespace["tracked_resonators"].append(resonator)

    # Since we are measuring an ADC stream, we populate a list of which input (1 or 2) the sensor uses
    sensor_input = [s.readout_resonator.opx_input.port_id for s in sensors]

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensor": xr.DataArray(sensors.get_names()),
        "readout_time": xr.DataArray(
            np.arange(0, node.parameters.readout_length_in_ns, 1),
            attrs={"long_name": "readout time", "units": "ns"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Allocate real-time variables on the OPX:
        #   adc_st       : stream collecting raw, real-time inputs of the OPX, per sensor
        #   n            : shot counter
        #   n_st         : stream reporting shot index to PC (progress bar)
        # Streams keyed by sensor name
        n = declare(int)
        n_st = declare_stream()
        adc_st = {sensor.name: declare_stream(adc_trace=True) for sensor in sensors}

        # Measure each batch, multiplexed by sensors
        for multiplexed_sensors in sensors.batch():
            align()  # Start with a global align

            # ── OUTER LOOP: repeat the full sweep n_avg times ──
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # Tell the PC which shot we are on

                for sensor in multiplexed_sensors.values():
                    # Reset the phase of the digital oscillator associated to the resonator element. Needed to average the cosine signal.
                    reset_if_phase(sensor.readout_resonator.name)
                    # Measure the resonator (send a readout pulse and record the raw ADC trace)
                    sensor.readout_resonator.measure("readout", stream=adc_st[sensor.name])
                    # Wait 1µs for the resonator to deplete and to let enough time for the stream processing to process the raw ADC traces
                    sensor.readout_resonator.wait(250)
                align()

        # ── Post-processing on the OPX before data reaches the PC ─────────
        with stream_processing():
            n_st.save("n")
            for i, sensor in enumerate(node.namespace["sensors"]):
                # Specify the ADC input to save based on which input the sensor is actually connected to
                if sensor_input[i] == 1:
                    stream = adc_st[i].input1()
                else:
                    stream = adc_st[i].input2()

                # Save both the averaged and single trace of the ADC input
                # Will save average:
                stream.real().average().save(f"adcI{i + 1}")
                stream.image().average().save(f"adcQ{i + 1}")
                # Will save only last run:
                stream.real().save(f"adc_single_runI{i + 1}")
                stream.image().save(f"adc_single_runQ{i + 1}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
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
    """Generate simulated IQ ADC data so the full analysis pipeline can run without hardware."""
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
    """Fit the time-of-flight delay, storing results in "ds_fit" and "fit_results"."""
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
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


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the data analysis was successful."""

    # Revert the change done at the beginning of the node
    for tracked_resonator in node.namespace.get("tracked_resonators", []):
        tracked_resonator.revert_changes()

    with node.record_state_updates():
        for s in node.namespace["sensors"]:
            if node.outcomes[s.name] == "failed":
                continue

            fit_result = node.results["fit_results"][s.name]
            if node.parameters.time_of_flight_in_ns is not None:
                s.readout_resonator.time_of_flight = node.parameters.time_of_flight_in_ns + fit_result["tof_to_add"]
            else:
                s.readout_resonator.time_of_flight += fit_result["tof_to_add"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
