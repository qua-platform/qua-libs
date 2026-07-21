# %% {Imports}
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import progress_counter_with_log
from qualang_tools.units import unit

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability import (
    Parameters,
    get_voltage_arrays,
    ScanMode,
    plot_raw_amplitude,
    plot_raw_phase,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_change_point_overlays,
    plot_line_fit_overlays,
    get_axis_names,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.common_utils.experiment import (
    get_dots,
    get_sensors,
    suppress_fetcher_axis_log_spam,
)

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
    node.parameters.num_shots = 50
    # node.parameters.scan_pattern = "spiral"
    node.parameters.x_points = 101
    node.parameters.y_points = 201
    # node.parameters.x_span = 0.4
    # node.parameters.x_points = 4
    # node.parameters.y_span = 0.4
    # node.parameters.y_points = 4
    node.parameters.x_axis_name = "virtual_dot_1_virtual_dot_2_pair"
    node.parameters.y_axis_name = "virtual_dot_2_virtual_dot_3_pair"
    # node.parameters.sensor_names = ["virtual_sensor_1"]
    # node.parameters.per_line_compensation = False
    node.parameters.run_in_video_mode = True
    # node.parameters.simulate = True
    # node.parameters.per_line_wait = 200
    # node.parameters.simulation_duration_ns = 30000
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
    node.parameters.x_axis_name, node.parameters.y_axis_name = get_axis_names(node)
    x_obj, y_obj = node.machine.get_component(
        node.parameters.x_axis_name
    ), node.machine.get_component(node.parameters.y_axis_name)
    node.namespace["axes_names"] = {"x_axis": x_obj.name, "y_axis": y_obj.name}
    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    node.namespace["voltage_points"] = x_obj.voltage_sequence.gate_set.get_macros()
    x_volts, y_volts = get_voltage_arrays(
        node
    )  # This sets the centres of x_volts and y_volts automatically to zero.
    x_offset, y_offset = node.parameters.x_center, node.parameters.y_center

    if node.parameters.dc_control:
        external_qdac = "qdac_ip" in node.machine.network
        node.machine.connect_to_external_source(external_qdac=external_qdac)
        virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]
        # If no offsets are provided, then default to 0 if dc_control is False, default to current value if dc_control is True
        if x_offset is None:
            x_offset = virtual_dc_set.get_voltage(
                node.parameters.x_axis_name, requery=True
            )
        if y_offset is None:
            y_offset = virtual_dc_set.get_voltage(
                node.parameters.y_axis_name, requery=True
            )

        node.log(
            f"Setting DC Voltages via VirtualDCSet. {node.parameters.x_axis_name} : {x_offset}V, {node.parameters.y_axis_name} : {y_offset}V"
        )
        virtual_dc_set.set_voltages(
            {
                node.parameters.x_axis_name: x_offset,
                node.parameters.y_axis_name: y_offset,
            }
        )
    else:
        # dc_control off. If offsets are None, default to zero
        if x_offset is None:
            x_offset = 0
        if y_offset is None:
            y_offset = 0
        if x_offset > 2.5 or y_offset > 2.5:
            raise ValueError("X or Y offset exceeds maximum amplified offset of 2.5V.")
        x_volts = x_volts + x_offset
        y_volts = y_volts + y_offset
    num_sensors = len(sensors)

    node.namespace["scan_mode"] = scan_mode = ScanMode.from_name(
        node.parameters.scan_pattern,
        use_precomputed_scan=node.parameters.spiral_use_precomputed_scan,
    )
    x_axis_ordered = scan_mode.get_x_axis_order(x_volts)
    y_axis_ordered = scan_mode.get_y_axis_order(y_volts)

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "y_volts": xr.DataArray(
            y_axis_ordered if not node.parameters.dc_control else y_axis_ordered + y_offset,
            attrs={"long_name": "voltage", "units": "V"},
        ),
        "x_volts": xr.DataArray(
            x_axis_ordered if not node.parameters.dc_control else x_axis_ordered + x_offset,
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }

    # node.namespace["sweep"]
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_sensors
        )

        # Scan mode defines when buffered points should be saved.
        n_buf = scan_mode.get_save_buffer_size(x_volts, y_volts)
        buf_idx = declare(int)
        save_idx = declare(int)
        I_buf = [declare(fixed, size=n_buf) for _ in range(num_sensors)]
        Q_buf = [declare(fixed, size=n_buf) for _ in range(num_sensors)]

        for multiplexed_sensors in sensors.batch():
            align()
            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                assign(buf_idx, 0)
                for x, y, save_flag in scan_mode.qua_scan(
                    seq,
                    x_obj,
                    y_obj,
                    x_volts,
                    y_volts,
                    node.parameters,
                ):
                    seq.ramp_to_voltages(
                        {x_obj.name: x, y_obj.name: y},
                        duration=node.parameters.hold_duration,
                        ramp_duration=node.parameters.ramp_duration,
                    )
                    if node.parameters.pre_measurement_delay:
                        seq.step_to_voltages(
                            {}, duration=node.parameters.pre_measurement_delay
                        )
                    align()
                    seq.step_to_voltages(
                        {},
                        duration=max(
                            s.readout_resonator.operations["readout"].length
                            for s in sensors
                        ),
                    )
                    for i, sensor in multiplexed_sensors.items():
                        # Select the resonator tied to the sensor
                        rr = sensor.readout_resonator
                        # Measure using said resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        assign(I_buf[i][buf_idx], I[i])
                        assign(Q_buf[i][buf_idx], Q[i])
                    assign(buf_idx, buf_idx + 1)

                    with if_(save_flag == 1):
                        scan_mode.compensate(seq, node.parameters)
                        with for_(save_idx, 0, save_idx < buf_idx, save_idx + 1):
                            for i, sensor in multiplexed_sensors.items():
                                save(I_buf[i][save_idx], I_st[i])
                                save(Q_buf[i][save_idx], Q_st[i])
                        assign(buf_idx, 0)

                seq.ramp_to_zero()

        with stream_processing():
            n_st.save("n")
            for i in range(num_sensors):
                scan_mode.qua_stream_processing(I_st[i], len(x_volts), len(y_volts)).save(f"I{i}")
                scan_mode.qua_stream_processing(Q_st[i], len(x_volts), len(y_volts)).save(f"Q{i}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
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
    or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Get rid of the "first axis must be qubit or qubit_pair" logger
    suppress_fetcher_axis_log_spam()
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
        print(job.execution_report())
    # Canonicalize to (sensors, x_volts, y_volts) for downstream processing.
    dataset = dataset.transpose("sensors", "x_volts", "y_volts")
    # Register the raw dataset, reordering if the scan mode requires it (e.g. spiral)
    node.results["ds_raw"] = node.namespace["scan_mode"].reorder_dataset(dataset)


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [
        node.machine.sensor_dots[name] for name in node.parameters.sensor_names
    ]


# %% {Analyse Data}
@node.run_action(
    skip_if=node.parameters.simulate
    or node.parameters.run_in_video_mode
    or not node.parameters.perform_edge_analysis
)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: v.to_dict() for k, v in fit_results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by sensors.grid_location."""
    ds_plot = node.results["ds_raw"].copy()

    point_kwargs = {}
    if node.parameters.plot_points and "voltage_points" in node.namespace:
        pair_prefix = node.machine.find_quantum_dot_pair(
            node.parameters.x_axis_name, node.parameters.y_axis_name
        )
        point_kwargs = dict(
            voltage_points=node.namespace["voltage_points"],
            x_axis_name=node.parameters.x_axis_name,
            y_axis_name=node.parameters.y_axis_name,
            pair_prefix=pair_prefix,
        )

    fig_amplitude = plot_raw_amplitude(
        ds_plot, node.namespace["sensors"], **point_kwargs
    )
    fig_phase = plot_raw_phase(ds_plot, node.namespace["sensors"], **point_kwargs)
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "amplitude": fig_amplitude,
        "phase": fig_phase,
    }
    if node.parameters.perform_edge_analysis and "fit_results" in node.results:
        for sensor in node.namespace["sensors"]:
            sensor_data = ds_plot.sel(sensors=sensor.id)
            fit_params = node.results["fit_results"].get(sensor.id, {})
            fig_cp = plot_change_point_overlays(sensor_data, fit_params, sensor.id)
            node.results["figures"][f"{sensor.id}_change_points"] = fig_cp
            if fit_params.get("segments"):
                fig_lines = plot_line_fit_overlays(sensor_data, fit_params, sensor.id)
                node.results["figures"][f"{sensor.id}_line_fits"] = fig_lines
    annotate_node_figures(node)


# %%
from calibration_utils.run_video_mode import create_video_mode


@node.run_action(skip_if=node.parameters.run_in_video_mode is False)
def run_video_mode(node: QualibrationNode[Parameters, Quam]):
    node.machine.track_integrated_voltage = True
    if node.parameters.virtual_gate_set_id is None:
        x_obj, y_obj = node.machine.get_component(
            node.parameters.x_axis_name
        ), node.machine.get_component(node.parameters.y_axis_name)
        if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
            raise ValueError(
                f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
            )
        vgs_id = x_obj.voltage_sequence.gate_set.id
    else:
        vgs_id = node.parameters.virtual_gate_set_id
    x_axis_name = node.parameters.x_axis_name
    y_axis_name = node.parameters.y_axis_name
    x_span, x_points = node.parameters.x_span, node.parameters.x_points
    y_span, y_points = node.parameters.y_span, node.parameters.y_points

    from pathlib import Path

    quam_state_path = Path(node.machine.serialiser._get_state_path()).resolve()
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
        virtual_gate_id=vgs_id,
        dc_control=node.parameters.dc_control,
        readout_pulses=[
            node.machine.sensor_dots[name].readout_resonator.operations["readout"]
            for name in node.parameters.sensor_names
        ],
        save_path=str(quam_state_path),
        port = node.parameters.video_mode_port,
        point_duration = node.parameters.hold_duration,
    )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the node results and state."""
    node.save()
