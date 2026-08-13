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

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.charge_stability_opx import (
    get_axis_names_and_validate,
)
from calibration_utils.charge_stability_qdac import (
    Parameters,
    get_voltage_arrays,
    set_dac_offsets, 
    axis_source_bools,
    prepare_dc_lists,
    build_qua_program_with_mixed_axes,
    ScanMode,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_amplitude,
    plot_raw_phase,
    plot_change_point_overlays,
    plot_line_fit_overlays,
)

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters.experiment import _make_batchable_list_from_multiplexed

from calibration_utils.common_utils.experiment import (
    get_dots,
    get_sensors,
)

# %% {Node initialisation}
description = """
        2D CHARGE STABILITY MAP (OPX + QDAC)

This sequence measures a 2D charge-stability diagram by stepping two virtual-gate axes
(X and Y) and performing RF reflectometry on one or more sensor dots at each (Vx, Vy) point.
Charge transitions appear as edges in the demodulated I/Q response.

Each axis can be swept by either the OPX (fast ramps/steps on bias tees) or the QDAC
(slow DC via preloaded dc_lists, triggered by the OPX). Mixed configurations are supported:
for example, a wide QDAC raster on one plunger with a fast OPX sweep on the other.

When ``dc_control=True``, the sweep center is held on the external DAC (VirtualDCSet) while
the OPX (or QDAC dc_list offset) performs the relative sweep around that center. When
``dc_control=False``, the center is applied as an OPX offset on axes swept by the OPX.

Prerequisites:
    - IQ mixer/Octave calibrated on the readout line (01a_mixer_calibration).
    - Time of flight, offsets, and gains calibrated (01a_time_of_flight).
    - Sensor resonators calibrated (02a_resonator_spectroscopy, 02b_resonator_spectroscopy_vs_power).
    - QUAM initialized with readout amplitude/duration, QuantumDot and SensorDot elements.
    - QdacSpec configured on each VoltageGate; VirtualDCSet configured in the machine.

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
    name="05b_charge_stability_qdac", description=description, parameters=Parameters()
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
    """Build the 2D voltage sweep and the QUA pulse sequence.

    The QUA program depends on ``x_from_qdac`` / ``y_from_qdac``:
        - OPX-only: nested loops ramp both axes on the OPX.
        - Mixed: QDAC slow axis via dc_list + OPX fast axis via ramps; OPX triggers the QDAC.
        - QDAC-only: OPX only triggers and reads out; both axes step via dc_lists.
    """
    u = unit(coerce_to_integer=True)

    # ── Experiment parameters (Python side) ──────────────────────────────

    # Gets the axis names and validates that the axis exists in the GateSet, and that the elements are in the same GateSet
    # Additionally adds keys 'x_axis', 'y_axis', 'gate_set_id' in the node.namespace['axes_names']
    x_axis_name, y_axis_name, vgs_id = get_axis_names_and_validate(node)

    # Relative sweep coordinates centered at zero; absolute values are set below
    x_volts, y_volts = get_voltage_arrays(node)

    # Extract the sensors relevant for this measurement
    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)

    # Number of averages
    n_avg = node.parameters.num_shots  # repetitions averaged at each (Vx, Vy) point

    # The scan mode is used to determine the DC lists
    scan_mode = ScanMode.from_name(node.parameters.scan_pattern)

    # Determine which axis is on the QDAC (external)
    x_ext, y_ext = axis_source_bools(node)
    if not x_ext and not y_ext: 
        raise ValueError("Neither X nor Y axis is via the QDAC. If you would like an OPX vs OPX sweep, please run node 05a")


    # ── QUA program (runs on the OPX in real time) ───────────────────────
    # Two variants depending on which axes are QDAC-swept (x_external / y_external).
    # Case 1: We have one slow axis and one fast axis
    if x_ext and not y_ext:
        build_qua_program_with_mixed_axes(
            node,
            x_axis_name,
            y_axis_name, 
            x_volts, # Values here don't matter - just the length, since we will just be sending triggers
            y_volts, # OPX values for Y axis
            scan_mode = scan_mode,
        )
    elif y_ext and not x_ext: 
        build_qua_program_with_mixed_axes(
            node,
            y_axis_name,
            x_axis_name, 
            y_volts, # Values here don't matter - just the length, since we will just be sending triggers
            x_volts, # OPX values for Y axis
            scan_mode = scan_mode,
        )

    # Case 2: Y on QDAC (slow), X on OPX (fast)
    elif x_ext and y_ext:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            # Allocate real-time variables on the OPX:
            #   I[i], Q[i]       : demodulated quadratures for sensor i
            #   I_st[i], Q_st[i] : stream buffers before transfer to PC
            #   n, n_st          : shot counter exposed for the progress bar
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            trig_counter = declare(int)

            for multiplexed_sensors in sensors.batch():
                align()
                # ── OUTER LOOP: average over shots ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # update the progress bar

                    # ── INNER LOOP: trigger counter ──────────────────────
                    # One trigger pair per pixel; dc_lists advance both X and Y
                    with for_(
                        trig_counter,
                        0,
                        trig_counter < int(len(x_volts) * len(y_volts)),
                        trig_counter + 1,
                    ):
                        # Play the triggers for both axes
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        y_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")

                        # Wait for the trigger to be processed
                        wait(node.parameters.post_trigger_wait_ns // 4)

                        # Perform muliplexed measurements on the sensors
                        # A python for loop is used so that the measurements are performed in parallel.
                        for i, sensor in multiplexed_sensors.items():
                            # Select the resonator tied to the sensor
                            rr = sensor.readout_resonator
                            # Measure using said resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # Post-measurement wait (Optional)
                            rr.wait(500)  # TODO: Make this a parameter

                            # Save the I/Q data to the streams
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])
            with stream_processing():
                n_st.save("n")  # save the shot counter for the progress bar
                for i in range(num_sensors):
                    # The averaged data for each (x, y) pixel is saved to the streams
                    # Individual shots are not retained.
                    # .buffer(len(y)).buffer(len(x)) : group into 2D grid
                    # .average() : average over all shots (n_avg repetitions)
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate or node.parameters.use_validation
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX and simulate the QUA program."""
    qmm = node.machine.connect()
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
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the OPX, execute the QUA program, and fetch raw I/Q into ``ds_raw``."""
    qmm = node.machine.connect()

    # Set the voltages based on whether a) A desired offset is supplied and b) whether the axis is external. If both are satisfied, the offset is applied.  
    voltages_to_set = {
        node.namespace["axes_names"]["x_axis"]: node.parameters.x_offset if node.parameters.x_from_qdac else None,
        node.namespace["axes_names"]["y_axis"]: node.parameters.y_offset if node.parameters.y_from_qdac else None,
    }
    set_dac_offsets(
        node,
        dc_set_id=node.namespace["axes_names"]["gate_set_id"],
        voltages=voltages_to_set,
    )

    # Prepare the DC lists here, so that we ensure that the QMM has been connected at least once, and the DC set & dacs dict exist. 

    # ── QDAC dc_lists (Python side, before QUA runs) ───────────────────────
    #
    # For each QDAC-swept axis, resolve virtual → physical voltages and preload dc_lists.
    # All lists share one external trigger; the OPX fires it once per slow-axis step.
    # QDAC/QDAC 2D maps do not require additional triggers.
    
    prepare_dc_lists(node)
    for axis_name, resolved_offset in node.namespace.get("resolved_offsets", {}).items():
        da = node.namespace["sweep_axes"][axis_name]
        node.namespace["sweep_axes"][axis_name] = da + resolved_offset
    
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
        print(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Simulate validation data}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.use_validation)
def simulate_data(node: QualibrationNode[Parameters, Quam]):
    """Generate synthetic charge-stability data for pipeline validation (placeholder)."""
    pass


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset by QUAlibrate run ID."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the sensors from the loaded node parameters
    node.namespace["sensors"] = [node.machine.sensor_dots[name] for name in node.parameters.sensor_names]


# %% {Analyse_data}
@node.run_action(skip_if=not node.parameters.perform_edge_analysis)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Run optional edge analysis: change-point detection and charge-transition line fitting.
    The fitted data are stored in "ds_fit" xarray dataset and the fit results are stored in the "fit_results" dictionary.
    """
    ds_processed = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)

    # Convert FitParameters to dictionaries for storage (JSON serializable)
    node.results["fit_results"] = {k: v.to_dict() for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)


# %% {Plot_data}
@node.run_action()
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot amplitude and phase heatmaps; optionally overlay edge-analysis results."""
    if "ds_fit" in node.results:
        ds_plot = node.results["ds_fit"]
    else:
        ds_plot = process_raw_dataset(node.results["ds_raw"].copy(deep=True), node)

    fig_amplitude = plot_raw_amplitude(
        ds_plot,
        node.namespace["sensors"],
        x_axis_name=node.namespace["axes_names"]["x_axis"],
        y_axis_name=node.namespace["axes_names"]["y_axis"],
    )
    fig_phase = plot_raw_phase(
        ds_plot,
        node.namespace["sensors"],
        x_axis_name=node.namespace["axes_names"]["x_axis"],
        y_axis_name=node.namespace["axes_names"]["y_axis"],
    )
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


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist datasets, figures, and QUAM state to the QUAlibrate database."""
    node.save()
