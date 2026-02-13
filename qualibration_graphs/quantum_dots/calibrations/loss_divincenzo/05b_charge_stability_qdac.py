# %% {Imports}
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability_qdac import Parameters, get_voltage_arrays
from calibration_utils.charge_stability_opx import ScanMode

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.experiment import get_dots, get_sensors, _make_batchable_list_from_multiplexed

description = """
            OPX & QDAC 2D CHARGE STABILITY MAP
This script involves a simple 2D voltage map, done by stepping the X and Y Quantum Dots
to their corresponding voltages, sending a readout pulse, and demodulating the 'I' and 'Q'
quadratures. In this node, you may perform the 2D map using either OPX outputs or QDAC
voltage source outputs, which are triggered by the OPX.

Note: Currently the external v external 2D map has a large number of pause() functions, which
    increases the runtime drastically. Use with caution.

Prerequisites:
    - Having calibrated the IQ mixer/Octave connected to the readout line (node 01a_mixer_calibration.py).
    - Having calibrated the time of flight, offsets, and gains (node 01a_time_of_flight.py).
    - Having calibrated the resonators coupled to the SensorDot components (nodes 02a_resonator_spectroscopy.py, 02b_resonator_spectroscopy_vs_power.py).
    - Having initialized the QUAM state parameters for the readout pulse amplitude and duration.
    - Having registered the QuantumDot elements and your SensorDot elements in your QUAM state.
    - Having configured the QdacSpec on each of the VoltageGate objects.
    - Having configured the VirtualDCSet in your machine.
"""


node = QualibrationNode[Parameters, Quam](
    name="05b_charge_stability_qdac", description=description, parameters=Parameters()
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
node.parameters.simulate = False
node.parameters.x_from_qdac = True
node.parameters.y_from_qdac = True
node.parameters.num_shots = 100
node.parameters.x_axis_name = "virtual_dot_1"
node.parameters.y_axis_name = "virtual_dot_2"


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)

    x_obj, y_obj = node.machine.get_component(node.parameters.x_axis_name), node.machine.get_component(
        node.parameters.y_axis_name
    )
    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    x_volts, y_volts = get_voltage_arrays(node)

    node.namespace["sensors"] = sensors = get_sensors(node)

    # Connect machine to QDAC
    node.machine.connect_to_external_source(external_qdac=True)

    # The DC lists are dependent on the scan mode.
    scan_mode = ScanMode.from_name(node.parameters.scan_pattern)

    # Which one is external/not?
    x_external, y_external = node.parameters.x_from_qdac, node.parameters.y_from_qdac

    # Set up the DC lists. They are mapped to the same trigger, so no need for two triggers for QDAC/QDAC 2dmap.
    if x_external:
        dc_list_x = node.machine.qdac.channel(x_obj.physical_channel.qdac_spec.qdac_output_port).dc_list(
            voltages=(
                np.repeat(scan_mode.get_outer_loop(x_volts), len(y_volts))
                if y_external
                else scan_mode.get_outer_loop(x_volts)
            ),
            dwell_s=10e-6,
            stepped=True,
        )
        dc_list_x.start_on_external(trigger=1)

    if y_external:
        dc_list_y = node.machine.qdac.channel(y_obj.physical_channel.qdac_spec.qdac_output_port).dc_list(
            voltages=(
                np.tile(scan_mode.get_outer_loop(y_volts), len(x_volts))
                if x_external
                else scan_mode.get_outer_loop(y_volts)
            ),
            dwell_s=10e-6,
            stepped=True,
        )
        dc_list_y.start_on_external(trigger=1)

    num_sensors = len(sensors)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
        "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
    }

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions

    # Case 1: Both axes OPX voltages
    if not x_external and not y_external:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)

            for multiplexed_sensors in sensors.batch():
                align()
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(x, x_volts)):
                        with for_(*from_array(y, y_volts)):
                            seq.ramp_to_voltages(
                                {x_obj.name: x, y_obj.name: y},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
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
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")

    # Case 2: X external and Y OPX
    elif x_external and not y_external:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            x = declare(fixed)

            for multiplexed_sensors in sensors.batch():
                align()
                # We know that the X is the slow axis. Order it so that the X axis comes first
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(x, x_volts)):
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        wait(node.parameters.post_trigger_wait_ns // 4)
                        for y in scan_mode.inner_loop(y_volts):
                            seq.ramp_to_voltages(
                                {y_obj.id: y},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            align()
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
                    I_st[i].buffer(len(y_volts)).average().buffer(len(x_volts)).save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).average().buffer(len(x_volts)).save(f"Q{i}")

    # Case 3: X OPX and Y external
    elif not x_external and y_external:
        # Transpose so that the slow (Y) is on the outer loop
        node.namespace["sweep_axes"] = {
            "sensors": xr.DataArray(sensors.get_names()),
            "y_volts": xr.DataArray(y_volts, attrs={"long_name": "voltage", "units": "V"}),
            "x_volts": xr.DataArray(x_volts, attrs={"long_name": "voltage", "units": "V"}),
        }
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            y = declare(fixed)

            for multiplexed_sensors in sensors.batch():
                align()
                # We know that the Y is the slow axis. Order it so that the Y axis comes first
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(y, y_volts)):
                        y_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        wait(node.parameters.post_trigger_wait_ns // 4)
                        for x in scan_mode.inner_loop(x_volts):
                            seq.ramp_to_voltages(
                                {x_obj.id: x},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            align()
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
                    I_st[i].buffer(len(x_volts)).average().buffer(len(y_volts)).save(f"I{i}")
                    Q_st[i].buffer(len(x_volts)).average().buffer(len(y_volts)).save(f"Q{i}")

    # Case 4: Both external
    elif x_external and y_external:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            trig_counter = declare(int)

            for multiplexed_sensors in sensors.batch():
                align()
                # We know that the Y is the slow axis. Order it so that the Y axis comes first
                with for_(n, 0, n < node.parameters.num_shots, n + 1):
                    save(n, n_st)

                    with for_(trig_counter, 0, trig_counter < int(len(x_volts) * len(y_volts)), trig_counter + 1):
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")

                        wait(node.parameters.post_trigger_wait_ns // 4)
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
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate or node.parameters.use_validation
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
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.use_validation)
def simulate_data(node: QualibrationNode[Parameters, Quam]):
    """Simulate the data."""
    pass


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    # load_data_id = node.parameters.load_data_id
    # # Load the specified dataset
    # node.load_from_id(node.parameters.load_data_id)
    # node.parameters.load_data_id = load_data_id
    # # Get the sensors from the loaded node parameters
    # node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    # TODO: Implement analysis - remove pass when complete
    pass
    # # Process raw dataset (convert ADC to volts, compute amplitude)
    # node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)

    # # Perform charge stability analysis
    # node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # # Convert FitParameters to dictionaries for storage (JSON serializable)
    # node.results["fit_results"] = {k: v.to_dict() for k, v in fit_results.items()}

    # # Log the relevant information extracted from the data analysis
    # log_fitted_results(node.results["fit_results"], log_callable=node.log)
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""
    # TODO: Implement plotting - remove pass when complete
    pass
    # Plot basic amplitude and phase maps
    # fig_amplitude = plot_raw_amplitude(node.results["ds_raw"], node.namespace["sensors"])
    # # fig_phase = plot_raw_phase(node.results["ds_raw"], node.namespace["sensors"])

    # # Store the generated figures
    # node.results["figures"] = {
    #     "amplitude": fig_amplitude,
    #     # "phase": fig_phase,
    # }

    # # Optionally plot detailed analysis results if fit_results are available
    # if "fit_results" in node.results and node.results["fit_results"]:
    #     sensors = node.namespace["sensors"]
    #     for sensor in sensors:
    #         sensor_data = node.results["ds_raw"].sel(sensors=sensor.id)
    #         fit_params = node.results["fit_results"][sensor.id]

    #         # Plot change point overlays
    #         fig_cp = plot_change_point_overlays(sensor_data, fit_params, sensor.id)
    #         node.results["figures"][f"{sensor.id}_change_points"] = fig_cp

    #         if fit_params.get("segments"):
    #             fig_lines = plot_line_fit_overlays(sensor_data, fit_params, sensor.id)
    #             node.results["figures"][f"{sensor.id}_line_fits"] = fig_lines

    # plt.show()  # Commented out to avoid blocking in non-interactive mode
    pass


# %%
from calibration_utils.run_video_mode import create_video_mode


@node.run_action(skip_if=node.parameters.run_in_video_mode is False)
def run_video_mode(node: QualibrationNode[Parameters, Quam]):
    #     x_axis_name = node.parameters.x_axis_name
    #     y_axis_name = node.parameters.y_axis_name
    #     x_span, x_points = node.parameters.x_span, node.parameters.x_points
    #     y_span, y_points = node.parameters.y_span, node.parameters.y_points

    #     create_video_mode(
    #         machine=node.machine,
    #         num_software_averages=node.parameters.num_shots,
    #         log=node.log,
    #         x_axis_name=x_axis_name,
    #         y_axis_name=y_axis_name,
    #         x_span=x_span,
    #         x_points=x_points,
    #         y_span=y_span,
    #         y_points=y_points,
    #         virtual_gate_id=node.parameters.virtual_gate_set_id,
    #         dc_control=node.parameters.dc_control,
    #         readout_pulses=[
    #             node.machine.sensor_dots[name].readout_resonator.operations["readout"]
    #             for name in node.parameters.sensor_names
    #         ],
    #         save_path="/Users/User/.qualibrate/user_storage",
    #     )
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    # TODO: Uncomment when complete
    pass
    # node.save()
    pass
