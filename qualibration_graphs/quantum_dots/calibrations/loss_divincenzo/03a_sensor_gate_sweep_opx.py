# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from dataclasses import asdict

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit
from qualang_tools.loops import from_array

from quam_builder.architecture.quantum_dots.operations.names import VoltagePointName
from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.sensor_gate_sweep import Parameters
from calibration_utils.sensor_gate_sweep import (
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    generate_simulated_dataset,
)
from calibration_utils.sensor_gate_sweep import plot_raw_phase, plot_amplitude_with_fit
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import get_sensors
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.core import tracked_updates

# %% {Node initialisation}
description = """
        CHARGE SENSOR GATE SWEEP with the OPX
This sequence involves sweeping the voltage biasing the sensor gate using the OPX connected to the AC line of the bias-tee.
A sticky element is used in order to maintain the voltage level and avoid fast voltage drops. The OPX signal can be
combined with an external DC source to increase the dynamics. The OPX measures the response of the sensor dot via RF
reflectometry, recording the I and Q quadratures of the demodulated signal.

The measurement performs a voltage sweep across a specified range with configurable step size. At each voltage point,
a readout pulse is sent to the resonator coupled to the sensor dot, and the reflected signal is demodulated and recorded.
A global average is performed (averaging on the most outer loop) and the data is extracted while the program is running
to display the sensor response with increasing SNR.

Prerequisites:
    - Connect the AC line of the bias-tee connected to the sensor dot to one OPX channel.
    - Having initialized the Quam (quam_config/populate_quam_state_*.py).
    - Having calibrated the resonators coupled to the SensorDot components.

State update:
    - Update the optimal voltage bias of each sensor dot.
"""


node = QualibrationNode[Parameters, Quam](
    name="03a_sensor_gate_sweep_opx", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.sensor_names = ["virtual_sensor_1", "virtual_sensor_2"]
    # node.parameters.num_shots = 2
    # node.parameters.offset_min = -0.4
    # node.parameters.offset_max = 0.4
    # node.parameters.offset_step = 0.01
    # node.parameters.simulate = True
    # node.parameters.use_simulated_data = True
    # node.parameters.duration_after_step = 5000
    # node.parameters.simulate = False
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
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

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

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "bias_offsets": xr.DataArray(
            bias_offsets, attrs={"long_name": "Sensor bias offset", "units": "V"}
        ),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_sensors
        )
        offset = declare(fixed)  # QUA variable for the readout frequency

        # No qubits yet at this point in the experiment - we only have sensors, batched by multiplexing. Simultaneous operation no problem
        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)

                with for_(*from_array(offset, bias_offsets)):
                    for i, sensor in multiplexed_sensors.items():
                        # Sweep the sensor bias voltage
                        readout_len = sensor.readout_resonator.operations[
                            "readout"
                        ].length
                        # sensor.physical_channel.set_dc_offset(offset=offset)
                        sensor.step_to_voltages(
                            {sensor.name: offset},
                            duration=readout_len + node.parameters.duration_after_step,
                        )
                        # Measure the resonator after settling the sensor bias point
                        sensor.voltage_sequence.step_to_voltages(
                            voltages={}, duration=node.parameters.duration_after_step
                        )
                        align()
                        sensor.voltage_sequence.step_to_voltages(
                            voltages={},
                            duration=sensor.readout_resonator.operations[
                                "readout"
                            ].length,
                        )
                        # sensor.readout_resonator.wait(node.parameters.duration_after_step // 4)
                        sensor.readout_resonator.measure(
                            "readout", qua_vars=(I[i], Q[i])
                        )
                        # save data
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

                    for i, sensor in multiplexed_sensors.items():
                        sensor.voltage_sequence.ramp_to_zero()
                    align()
                    for i, sensor in multiplexed_sensors.items():
                        sensor.voltage_sequence.ramp_to_zero(ramp_duration=16)

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
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
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
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
    fig_raw_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["sensors"])
    fig_amplitude_fit = plot_amplitude_with_fit(
        node.results["ds_raw"], node.namespace["sensors"], node.results["ds_fit"]
    )
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "phase": fig_raw_phase,
        "amplitude_gradient": fig_amplitude_fit,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the sensor data analysis was successful."""

    with node.record_state_updates():
        for sensor in node.namespace["sensors"]:
            if not node.results["fit_results"][sensor.name]["success"]:
                continue

            optimal_offset = node.results["fit_results"][sensor.name]["optimal_bias"]

            # Add point to the VirtualGateSet instead. Associate with the relevant SensorDot

            sensor.add_point(
                VoltagePointName.MEASURE,
                voltages={sensor.name: optimal_offset},
                duration=sensor.readout_resonator.operations["readout"].length,
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
