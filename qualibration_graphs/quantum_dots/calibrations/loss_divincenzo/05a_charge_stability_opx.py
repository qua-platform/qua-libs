# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability_opx import (
    Parameters,
    analyse_raw_data,
    get_voltage_arrays,
    ScanMode,
    plot_all,
    get_axis_names_and_validate,
    set_dac_offsets,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.common_utils.experiment import (
    get_sensors,
)

description = """
2D OPX CHARGE STABILITY MAP

This sequence measures a 2D charge-stability diagram by sweeping two virtual-gate
axes directly with the OPX and performing RF reflectometry on one or more sensor
dots at each (Vx, Vy) point. Charge transitions appear as edges in the
demodulated I/Q response.

The OPX controls both sweep axes here. When ``dc_control=True``, the sweep center
is held on the external DAC (VirtualDCSet) while the OPX performs the relative
sweep around that center. When ``dc_control=False``, the center is applied as an
OPX offset directly on the swept axes.

Prerequisites:
    - IQ mixer/Octave calibrated on the readout line (01a_mixer_calibration).
    - Time of flight, offsets, and gains calibrated (01a_time_of_flight).
    - Sensor resonators calibrated (02a_resonator_spectroscopy, 02b_resonator_spectroscopy_vs_power).
    - QUAM initialized with readout amplitude/duration, QuantumDot and SensorDot elements.

Datasets:
    - ``ds_raw``: untouched I/Q fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed maps plus edge-analysis outputs (when ``perform_edge_analysis=True``).
      Used by ``plot_data`` when edge analysis is enabled.

Results (``node.results["fit_results"][<sensor>]``, when ``perform_edge_analysis=True``):
    - ``success``: whether edge detection and line fitting completed.
    - ``segments``: fitted charge-transition line segments.
    - ``intersections``: detected triple-point locations.

Figures (``node.results["figures"]``):
    - ``"amplitude"``: |I + iQ| heatmap vs (x_volts, y_volts) for each sensor.
    - ``"phase"``: IQ phase heatmap vs (x_volts, y_volts) for each sensor.
    - ``"<sensor>_change_points"``: change-point overlays (when edge analysis enabled).
    - ``"<sensor>_line_fits"``: fitted transition lines (when edge analysis enabled).

State update:
    - None (diagnostic map; use VirtualGateSet voltage points or downstream nodes to set bias).
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
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Gets the axis names and validates that the axis exists in the GateSet, and that the elements are in the same GateSet
    # Additionally adds keys 'x_axis', 'y_axis', 'gate_set_id' in the node.namespace['axes_names']
    x_axis_name, y_axis_name, vgs_id = get_axis_names_and_validate(node)

    # Extract the sensors relevant for this measurement
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Number of averages
    n_avg = node.parameters.num_shots

    # Extract the existing voltage points in the gate set, and add to the namespace
    node.namespace["voltage_points"] = node.machine.virtual_gate_sets[vgs_id].get_macros()

    # Constructs the voltage arrays from node.parameters (span, points, offset). 
    # If node.parameters.dc_control = True, then the offset is not included in the OPX sweep. 
    # If node.parameters.dc_control = False, then the offset will be included in the OPX sweep. 
    x_volts, y_volts = get_voltage_arrays(node)  # This sets the centers of x_volts and y_volts automatically based on the dc_control parameter. 

    # Reorder the axis values based on the desired scan mode. 
    # If using a spiral scan mode, then optionally pre-compute the scan in Python. 
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
            y_axis_ordered,
            attrs={"long_name": "voltage", "units": "V"},
        ),
        "x_volts": xr.DataArray(
            x_axis_ordered,
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        # Fetch the VoltageSequence (run-time helper) to be used for stepping/ramping. 
        seq = node.machine.voltage_sequences[vgs_id]

        # Real-time variables:
        # n  : shot counter
        # I : the measured raw I value
        # Q : the measured raw Q value
        # n_buf : the number of measurements to take before the QUA variables are saved into the stream
        # I_buf : an array to be filled up by the I QUA variables as the measurement goes on, periodically saved to I_st
        # Q_buf : an array to be filled up by the Q QUA variables as the measurement goes on, periodically saved to Q_st
        # save_idx : a helper QUA variable to loop over the array elements to save to the respective stream

        # Streams:
        # measurement_streams : stores the per-qubit assigned value, parity difference if node.parameters.parity_measurement = True
        # i_st : stores the per-qubit raw I value from the measurement
        # q_st : stores the per-qubit raw Q value from the measurement
        # n_st : stores the shot counter n, allowing the PC to track the progress
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
            num_IQ_pairs=num_sensors
        )

        # Scan mode defines when buffered points should be saved.
        n_buf = scan_mode.get_save_buffer_size(x_volts, y_volts)
        buf_idx = declare(int)
        save_idx = declare(int)
        I_buf = [declare(fixed, size=n_buf) for _ in range(num_sensors)]
        Q_buf = [declare(fixed, size=n_buf) for _ in range(num_sensors)]

        # Loop over the multiplexed sensors in this outer array. The full 2D map will be repeated for each batch. 
        for multiplexed_sensors in sensors.batch():

            # This defines the TOTAL HOLD DURATION on each pixel, INCLUDING the readout time. 
            # So the duration for each pixel becomes hold_duration + longest readout time in this batch of multiplexed sensors
            pixel_hold_duration = node.parameters.hold_duration + max(s.readout_resonator.operations["readout"].length for s in multiplexed_sensors.values())
            
            align() # Start with a global align

            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st) # Tell the PC which shot we are on 
                assign(buf_idx, 0) # Start filling the buffer array from 0

                # ── INNER 2D LOOP: step the voltages to each pixel and measure ────────────────

                # scan_mode.qua_scan yields the x, y, and the save flag as QUA variables
                for x, y, save_flag in scan_mode.qua_scan(
                    seq,
                    x_axis_name,
                    y_axis_name,
                    x_volts,
                    y_volts,
                    node.parameters,
                ):
                    
                    # Ramp to a particular pixel, and wait for the desired time
                    seq.ramp_to_voltages(
                        {x_axis_name: x, y_axis_name: y},
                        duration=pixel_hold_duration,
                        ramp_duration=node.parameters.ramp_duration,
                    )

                    # Loop over the multiplexed sensors and measure
                    for i, sensor in multiplexed_sensors.items():
                        # Select the resonator tied to the sensor
                        rr = sensor.readout_resonator
                        # Resonator should wait until after the ramp + hold
                        rr.wait((node.parameters.ramp_duration + node.parameters.hold_duration)//4)
                        # Measure using said resonator
                        rr.measure("readout", qua_vars=(I[i], Q[i]))
                        assign(I_buf[i][buf_idx], I[i])
                        assign(Q_buf[i][buf_idx], Q[i])
                    assign(buf_idx, buf_idx + 1)

                    # Periodically compensate & save buffers into the streams.
                    with if_(save_flag == 1):
                        scan_mode.compensate(seq, node.parameters)
                        with for_(save_idx, 0, save_idx < buf_idx, save_idx + 1):
                            for i, sensor in multiplexed_sensors.items():
                                save(I_buf[i][save_idx], I_st[i])
                                save(Q_buf[i][save_idx], Q_st[i])
                        assign(buf_idx, 0)

                seq.ramp_to_zero()

        # ── Post-processing on the OPX before data reaches the PC ─────────
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
    """Connect to the OPX and simulate the QUA program."""
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
    """Connect to the OPX, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Skip the dacs if we don't need dc_control anyway
    qmm = node.machine.connect(skip_dacs = not node.parameters.dc_control)
    
    if node.parameters.dc_control: 
        # If dc_control is requested, then we must apply the offsets provided. If None is provided, then it will default to the already applied value
        set_dac_offsets(
            node, 
            dc_set_id = node.namespace["axes_names"]["gate_set_id"], 
            voltages = {
                node.namespace["axes_names"]["x_axis"] : node.parameters.x_offset, 
                node.namespace["axes_names"]["y_axis"] : node.parameters.y_offset
            }
        )

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
                node=node,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
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
    """Process ``ds_raw``, fit edge data, and store processed outputs in ``ds_fit``."""
    (
        node.results["ds_fit"],
        node.results["fit_results"],
        node.outcomes,
    ) = analyse_raw_data(node.results["ds_raw"], node, log_callable=node.log)


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Build the node figures from the raw and fitted charge-stability data."""
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

    node.results["figures"] = plot_all(
        node.results["ds_raw"],
        node.namespace["sensors"],
        ds_fit=node.results.get("ds_fit"),
        fit_results=node.results.get("fit_results"),
        perform_edge_analysis=node.parameters.perform_edge_analysis,
        **point_kwargs,
    )


# %% {Run_video_mode}
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
    """Persist the node results and any recorded state updates."""
    node.save()
