# %% {Imports}
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
from calibration_utils.charge_stability_opx import Parameters, get_voltage_arrays, ScanMode
from calibration_utils.charge_stability_opx import plot_raw_amplitude, plot_raw_phase
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.experiment import get_dots, get_sensors

description = """
            2D OPX CHARGE STABILITY MAP
This script involves a simple 2D voltage map, done by stepping the X and Y Quantum Dots
to their corresponding voltages, sending a readout pulse, and demodulating the 'I' and 'Q'
quadratures. This is performed solely using the OPX to sweep/step the axes.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the resonators coupled to the SensorDot components (nodes 02a_resonator_spectroscopy.py, 02b_resonator_spectroscopy_vs_power.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having registered the QuantumDot elements and your SensorDot elements in your QUAM state.
"""


node = QualibrationNode[Parameters, Quam](
    name="05a_charge_stability_opx", description=description, parameters=Parameters()
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.multiplexed = True
    # node.parameters.num_shots = 2
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    node.namespace["sensors"] = sensors = get_sensors(node)

    x_obj, y_obj = node.machine.get_component(node.parameters.x_axis_name), node.machine.get_component(
        node.parameters.y_axis_name
    )

    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    x_volts, y_volts = get_voltage_arrays(node)
    num_sensors = len(sensors)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
        "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
    }

    scan_mode = ScanMode.from_name(node.parameters.scan_pattern)

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
        x = declare(fixed)
        y = declare(fixed)

        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                for x, y in scan_mode.scan(x_volts, y_volts):
                    # Simultaneous stepping of the voltage of the virtualised gates.
                    # If ramps are preferred, specify ramp_duration as arg in simultaneous()
                    seq.ramp_to_voltages(
                        {x_obj.name: x, y_obj.name: y},
                        duration=node.parameters.hold_duration,
                        ramp_duration=node.parameters.ramp_duration,
                    )
                    # with seq.simultaneous(duration = node.parameters.hold_duration, ramp_duration = node.parameters.ramp_duration):
                    #     x_obj.go_to_voltages(x, duration = node.parameters.hold_duration)
                    #     y_obj.go_to_voltages(y, duration = node.parameters.hold_duration)
                    if node.parameters.pre_measurement_delay:
                        wait(node.parameters.pre_measurement_delay // 4)
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
                I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I")
                Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q")


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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""


#     load_data_id = node.parameters.load_data_id
#     # Load the specified dataset
#     node.load_from_id(node.parameters.load_data_id)
#     node.parameters.load_data_id = load_data_id
#     # Get the sensors from the loaded node parameters
#     node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""


#     fig_amplitude = plot_raw_amplitude(node.results["ds_raw"], node.namespace["sensors"])
#     fig_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["sensors"])
#     plt.show()
#     # Store the generated figures
#     node.results["figures"] = {
#         "amplitude": fig_amplitude,
#         "phase": fig_phase,
#     }


# %%
from calibration_utils.run_video_mode import create_video_mode


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
    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    #     node.save()
    pass
