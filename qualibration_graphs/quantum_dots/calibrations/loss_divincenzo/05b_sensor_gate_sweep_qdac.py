# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualang_tools.loops import from_array

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.sensor_gate_sweep import Parameters
from calibration_utils.common_utils.experiment import get_sensors
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        CHARGE SENSOR GATE SWEEP with the QDAC-II
This sequence involves sweeping the voltage biasing the sensor gate using a QDAC channel connected to the DC line of
the bias-tee. For optimizing runtime, the QDAC preloads the list of voltage points and a digital marker from the OPX is
used in order to perform the sweep.
The OPX measures the response of the sensor dot via RF reflectometry, recording the I and Q quadratures of the demodulated signal.

The measurement performs a voltage sweep across a specified range with configurable step size. At each voltage point,
a readout pulse is sent to the resonator coupled to the sensor dot, and the reflected signal is demodulated and recorded.
A global average is performed (averaging on the most outer loop) and the data is extracted while the program is running
to display the sensor response with increasing SNR.

Prerequisites:
    - Connect the DC line of the bias-tee connected to the sensor dot to one QDAC-II channel.
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.

State update:
    - Update the optimal voltage bias of each sensor dot.
"""


node = QualibrationNode[Parameters, Quam](
    name="05b_sensor_gate_sweep_qdac", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.sensor_names = ["sensor_1"]
    # node.parameters.num_shots = 10
    # node.parameters.offset_min = -0.1
    # node.parameters.offset_max = 0.1
    # node.parameters.offset_step = 0.01
    node.parameters.simulate = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load("C:\git\qua-libs\qualibration_graphs\quantum_dots\quam_config\quam_state")


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    # Get the relevant sensor dots rom the node
    node.namespace["sensors"] = sensors = get_sensors(node)

    num_sensors = len(sensors)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots

    # The voltage offset sweep
    bias_offsets = np.arange(node.parameters.offset_min, node.parameters.offset_max, node.parameters.offset_step)

    # Set up the DC lists to load into the QDAC.
    dc_lists = {}
    for sensor in sensors:  # TODO: Issue with qcodes driver?
        dc_lists[sensor.name] = node.machine.qdac.channel(sensor.physical_channel.qdac_spec.qdac_output_port).dc_list(
            voltages=bias_offsets,
            dwell_s=10e-6,
            stepped=True,
        )
        dc_lists[sensor.name].start_on_external(trigger=sensor.physical_channel.qdac_spec.qdac_trigger_in)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(bias_offsets, attrs={"long_name": "Sensor bias offset", "units": "V"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        offset = declare(fixed)  # QUA variable for the readout frequency

        # No qubits yet at this point in the experiment - we only have sensors, batched by multiplexing. Simultaneous operation no problem
        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(offset, bias_offsets)):
                    for i, sensor in multiplexed_sensors.items():
                        # Play a digital marker to update the voltage outputted by the QDAC channel
                        sensor.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        # Wait for the voltage to settle
                        sensor.physical_channel.settle()
                        sensor.readout_resonator.align(sensor.physical_channel.name)
                        # Measure the resonator after settling the sensor bias point
                        sensor.readout_resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # wait 1µs between each iteration
                        sensor.readout_resonator.wait(250)
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                I_st[i].buffer(len(bias_offsets)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(bias_offsets)).average().save(f"Q{i + 1}")


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
        for sensor in node.namespace["sensors"]:
            if not node.results["fit_results"][sensor.name]["success"]:
                continue

            optimal_offset = 0.0  # find_optimal_offset(node.results["ds_fit"], sensor.name)
            sensor.physical_channel.offset_parameter(optimal_offset)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
