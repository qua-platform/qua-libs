# %% {Imports}
from platform import machine
import sys
from pathlib import Path

# Ensure local packages are importable when running this file directly
# Script path: .../quantum_dots/calibrations/loss_divincenzo/03b_charge_stability_demo.py
# We want siblings of 'calibrations', i.e. .../quantum_dots/quam_config and .../quantum_dots/validation_utils
_QUANTUM_DOTS_DIR = Path(__file__).resolve().parents[2]

# Add the quantum_dots directory to sys.path so that quam_config and validation_utils can be imported
if str(_QUANTUM_DOTS_DIR) not in sys.path:
    sys.path.insert(0, str(_QUANTUM_DOTS_DIR))

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability.parameters import SimulationParameters as Parameters
from calibration_utils.charge_stability import (
    get_voltage_arrays,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude,
    plot_raw_phase,
    plot_change_point_overlays,
    plot_line_fit_overlays,
)
from calibration_utils.run_video_mode.simulated_video_mode import setup_simulation, create_video_mode
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from calibration_utils.common_utils.experiment import get_dots, get_sensors

description = """
            2D CHARGE STABILITY MAP
This script involves a simple 2D voltage map, done by stepping the X and Y Quantum Dots
to their corresponding voltages, sending a readout pulse, and demodulating the 'I' and 'Q'
quadratures.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the resonators coupled to the SensorDot components (nodes 02a_resonator_spectroscopy.py, 02b_resonator_spectroscopy_vs_power.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having registered the QuantumDot elements and your SensorDot elements in your QUAM state.
"""


node = QualibrationNode[Parameters, Quam](name="03b_charge_stability", description=description, parameters=Parameters())


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    # node.parameters.num_shots = 2


# Instantiate the QUAM class from the state file
node.machine = Quam.load(
    "/Users/kalidu_laptop/march_meeting/qua-libs/qualibration_graphs/quantum_dots/calibration_utils/run_video_mode/simulated_video_mode/quam_state"
)
node.parameters.virtual_gate_set_id = "main_qpu"
node.parameters.x_axis_name = "virtual_dot_1"
node.parameters.y_axis_name = "virtual_dot_2"
node.parameters.sensor_names = ["virtual_sensor_1", "virtual_sensor_2"]
node.parameters.dc_control = True
node.parameters.result_type = "I"
node.parameters.num_shots = 100
node.parameters.simulate = True
node.parameters.use_validation = True
node.parameters.run_in_video_mode = True
node.parameters.video_mode_port = 8040


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    and not node.parameters.simulate
    or node.parameters.run_in_video_mode
)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    virtual_gate_set = node.machine.virtual_gate_sets[node.parameters.virtual_gate_set_id]

    node.namespace["sensors"] = sensors = get_sensors(node)

    x_obj, y_obj = node.machine.get_component(node.parameters.x_axis_name), node.machine.get_component(
        node.parameters.y_axis_name
    )
    x_volts, y_volts = get_voltage_arrays(node)
    num_sensors = len(sensors)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
        "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
    }

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        seq = virtual_gate_set.new_sequence()

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        x = declare(fixed)
        y = declare(fixed)

        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                with for_(*from_array(x, x_volts)):
                    with for_(*from_array(y, y_volts)):

                        # Simultaneous stepping of the voltage of the virtualised gates.
                        # If ramps are preferred, specify ramp_duration as arg in simultaneous()
                        with seq.simultaneous():
                            x_obj.go_to_voltages({node.parameters.x_axis_name: x}, duration=100)
                            y_obj.go_to_voltages({node.parameters.y_axis_name: y}, duration=100)

                        align()
                        for i, sensor in multiplexed_sensors.items():
                            # Select the resonator tied to the sensor
                            rr = sensor.readout_resonator
                            # Measure using said resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # Post-measurement wait (Optional)
                            rr.wait(500)

                            # Save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
                Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_validation
    or node.parameters.run_in_video_mode
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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.run_in_video_mode
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
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Simulate validation data}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.use_validation
    or node.parameters.run_in_video_mode
)
def simulate_data(node: QualibrationNode[Parameters, Quam]):
    """Simulate the data."""
    virtual_gate_set = node.machine.virtual_gate_sets[node.parameters.virtual_gate_set_id]
    sweep_axes = node.namespace["sweep_axes"]

    x_vals = np.asarray(sweep_axes["x_volts"].values, dtype=float)
    y_vals = np.asarray(sweep_axes["y_volts"].values, dtype=float)
    sensor_names = np.asarray(sweep_axes["sensors"].values)
    num_sensors = len(sensor_names)

    # Keep simulator defaults aligned with video mode simulation.
    base_point = {
        "virtual_dot_1": -5.0e-3,
        "virtual_dot_2": -5.0e-3,
        "virtual_sensor_1": -5.0e-3,
        "virtual_sensor_2": -5.0e-3,
    }

    if node.parameters.dc_control:
        node.machine.connect_to_external_source(external_qdac=True)
        dc_set = node.machine.virtual_dc_sets[node.parameters.virtual_gate_set_id]
    else:
        dc_set = None

    simulator = setup_simulation(
        base_point=base_point,
        gate_set=virtual_gate_set,
        dc_set=dc_set,
        sensor_gate_names=["virtual_sensor_1", "virtual_sensor_2"],
    )
    if dc_set is not None:
        dc_set.set_voltages(base_point)

    num_averages = max(1, int(node.parameters.num_shots))
    I_sum = None
    Q_sum = None
    for _ in range(num_averages):
        I_shot, Q_shot = simulator.measure_data(
            x_axis_name=node.parameters.x_axis_name,
            y_axis_name=node.parameters.y_axis_name,
            x_vals=x_vals,
            y_vals=y_vals,
            n_readout_channels=num_sensors,
        )
        I_shot = np.asarray(I_shot, dtype=float)
        Q_shot = np.asarray(Q_shot, dtype=float)

        if I_sum is None:
            I_sum = I_shot
            Q_sum = Q_shot
        else:
            I_sum += I_shot
            Q_sum += Q_shot

    I_data = I_sum / num_averages
    Q_data = Q_sum / num_averages

    offset_x = dc_set.get_voltage(node.parameters.x_axis_name) if dc_set is not None else 0.0
    offset_y = dc_set.get_voltage(node.parameters.y_axis_name) if dc_set is not None else 0.0

    ds_raw = xr.Dataset(
        {
            "I": xr.DataArray(
                I_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": sensor_names,
                    "x_volts": sweep_axes["x_volts"].values + offset_x,
                    "y_volts": sweep_axes["y_volts"].values + offset_y,
                },
            ),
            "Q": xr.DataArray(
                Q_data,
                dims=["sensors", "y_volts", "x_volts"],
                coords={
                    "sensors": sensor_names,
                    "x_volts": sweep_axes["x_volts"].values + offset_x,
                    "y_volts": sweep_axes["y_volts"].values + offset_y,
                },
            ),
        }
    )

    node.results["ds_raw"] = ds_raw


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None or node.parameters.run_in_video_mode)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    # Process raw dataset (convert ADC to volts, compute amplitude)
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

    # Perform charge stability analysis
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Convert FitParameters to dictionaries for storage (JSON serializable)
    node.results["fit_results"] = {k: v.to_dict() for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""
    # Plot basic amplitude and phase maps
    fig_amplitude = plot_raw_amplitude(node.results["ds_raw"], node.namespace["sensors"])
    # fig_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["sensors"])

    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_amplitude,
        # "phase": fig_phase,
    }

    # Optionally plot detailed analysis results if fit_results are available
    if "fit_results" in node.results and node.results["fit_results"]:
        sensors = node.namespace["sensors"]
        for sensor in sensors:
            sensor_data = node.results["ds_raw"].sel(sensors=sensor.id)
            fit_params = node.results["fit_results"][sensor.id]

            # Plot change point overlays
            fig_cp = plot_change_point_overlays(sensor_data, fit_params, sensor.id)
            node.results["figures"][f"{sensor.id}_change_points"] = fig_cp

            if fit_params.get("segments"):
                fig_lines = plot_line_fit_overlays(sensor_data, fit_params, sensor.id)
                node.results["figures"][f"{sensor.id}_line_fits"] = fig_lines

    # plt.show()  # Commented out to avoid blocking in non-interactive mode


# %%
# from calibration_utils.run_video_mode import create_video_mode
@node.run_action(skip_if=node.parameters.run_in_video_mode is False)
def run_video_mode(node: QualibrationNode[Parameters, Quam]):
    x_axis_name = node.parameters.x_axis_name
    y_axis_name = node.parameters.y_axis_name
    x_span, x_points = node.parameters.x_span, node.parameters.x_points
    y_span, y_points = node.parameters.y_span, node.parameters.y_points

    create_video_mode(
        machine=node.machine,
        num_software_averages=node.parameters.num_shots,
        log=node.log,
        x_axis_name=x_axis_name,
        y_axis_name=y_axis_name,
        x_span=x_span,
        x_points=x_points,
        y_span=y_span,
        y_points=y_points,
        virtual_gate_id=node.parameters.virtual_gate_set_id,
        dc_control=node.parameters.dc_control,
        readout_pulses=[
            node.machine.sensor_dots[name].readout_resonator.operations["readout"]
            for name in node.parameters.sensor_names
        ],
        save_path="/Users/User/.qualibrate/user_storage",
        port=node.parameters.video_mode_port,
    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
