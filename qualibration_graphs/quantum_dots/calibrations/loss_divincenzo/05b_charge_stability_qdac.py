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
from calibration_utils.charge_stability import (
    OPXQDACParameters as Parameters,
    get_voltage_arrays,
    prepare_dc_lists,
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

    # Virtual-gate components defining the X/Y sweep axes (must share a VirtualGateSet)
    x_obj, y_obj = node.machine.get_component(node.parameters.x_axis_name), node.machine.get_component(
        node.parameters.y_axis_name
    )
    node.namespace["axes_names"] = {"x_axis": x_obj.name, "y_axis": y_obj.name}
    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X axis and Y axis elements belong to different VirtualGateSet. x: {x_obj.voltage_sequence.gate_set.id}, y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    # Relative sweep coordinates centered at zero; absolute values are set below
    x_volts, y_volts = get_voltage_arrays(node)

    node.namespace["sensors"] = sensors = get_sensors(node)
    num_sensors = len(sensors)
    n_avg = node.parameters.num_shots  # repetitions averaged at each (Vx, Vy) point

    # Register the QDAC driver so dc_lists can be preloaded before execution
    node.machine.connect_to_external_source(external_qdac=True)

    # The scan mode is used to determine the DC lists
    scan_mode = ScanMode.from_name(node.parameters.scan_pattern)

    # Determine which axis is on the QDAC (external)
    x_external, y_external = node.parameters.x_from_qdac, node.parameters.y_from_qdac

    # ── Sweep center: VirtualDCSet (dc_control) vs OPX offset ────────────
    #
    # dc_control=True  → center held on external DAC; QDAC axes add center to dc_list values
    # dc_control=False → center applied as OPX offset on axes swept by the OPX

    if node.parameters.dc_control:
        # Sweep center is on the external QDAC
        dc_set = node.machine.virtual_dc_sets[vgs_id]
        # If None, then default to current virtual values.
        if node.parameters.x_center is None:
            node.parameters.x_center = dc_set.get_voltage(node.parameters.x_axis_name)
        if node.parameters.y_center is None:
            node.parameters.y_center = dc_set.get_voltage(node.parameters.y_axis_name)

        if not x_external:
            # OPX-swept X axis: park the QDAC at the center so the OPX performs the relative sweep
            dc_set.set_voltages({node.parameters.x_axis_name: node.parameters.x_center})

        if not y_external:
            # OPX-swept Y axis: park the QDAC at the center so the OPX performs the relative sweep
            dc_set.set_voltages({node.parameters.y_axis_name: node.parameters.y_center})

    else:
        # Sweep center is on the OPX
        if node.parameters.x_center is None:
            # If None, then default to zero.
            node.parameters.x_center = 0
        if node.parameters.y_center is None:
            # If None, then default to zero.
            node.parameters.y_center = 0
        # When dc_control is False, the center is applied as an OPX offset on axes swept by the OPX.
        # This means that, for these axes, the x/y_volts array needs to be mutated to include the center offset.
        if not x_external:
            x_volts = x_volts + node.parameters.x_center
        if not y_external:
            y_volts = y_volts + node.parameters.y_center

    # ── QDAC dc_lists (Python side, before QUA runs) ───────────────────────
    #
    # For each QDAC-swept axis, resolve virtual → physical voltages and preload dc_lists.
    # All lists share one external trigger; the OPX fires it once per slow-axis step.
    # QDAC/QDAC 2D maps do not require additional triggers.

    if x_external:
        x_array = x_volts + node.parameters.x_center  # absolute QDAC voltages must include the center offset
        prepare_dc_lists(
            node=node,
            virtual_dc_set_id=vgs_id,
            axis_name=node.parameters.x_axis_name,
            axis_values=(
                np.repeat(
                    x_array, len(y_volts)
                )  # Both X and Y axes are applied by the QDAC with a single trigger from the OPX
                if y_external
                else scan_mode.get_outer_loop(x_array)  # mixed: X slow, Y on OPX
            ),
        )

    if y_external:
        y_array = y_volts + node.parameters.y_center  # absolute QDAC voltages must include the center offset
        prepare_dc_lists(
            node=node,
            virtual_dc_set_id=vgs_id,
            axis_name=node.parameters.y_axis_name,
            axis_values=(
                np.tile(
                    y_array, len(x_volts)
                )  # Both X and Y axes are applied by the QDAC with a single trigger from the OPX
                if x_external
                else scan_mode.get_outer_loop(y_array)  # mixed: Y slow, X on OPX
            ),
        )

    # Metadata for data fetching: labels the saved I/Q arrays when results come back from the OPX
    node.namespace["sweep_axes"] = {
        "sensors": xr.DataArray(sensors.get_names()),
        "x_volts": xr.DataArray(
            (x_volts if (not node.parameters.dc_control and not x_external) else x_volts + node.parameters.x_center),
            attrs={"long_name": "voltage", "units": "V"},
        ),
        "y_volts": xr.DataArray(
            (y_volts if (not node.parameters.dc_control and not y_external) else y_volts + node.parameters.y_center),
            attrs={"long_name": "voltage", "units": "V"},
        ),
    }

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    #
    # Four variants depending on which axes are QDAC-swept (x_external / y_external).

    # Case 1: both axes on the OPX — standard nested raster over (x, y)
    if not x_external and not y_external:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            # Allocate real-time variables on the OPX:
            #   I[i], Q[i]       : demodulated quadratures for sensor i
            #   I_st[i], Q_st[i] : stream buffers before transfer to PC
            #   n, n_st          : shot counter exposed for the progress bar
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            x = declare(fixed)  # values on the x axis
            y = declare(fixed)  # values on the y axis
            for multiplexed_sensors in sensors.batch():
                align()
                # ── OUTER LOOP: average over shots ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # update the progress bar
                    # ── MIDDLE LOOP: slow axis (x) ─────────────────────────
                    with for_(*from_array(x, x_volts)):
                        if node.parameters.per_line_wait > 0:
                            # Optional settle time at the start of each scan line
                            # Set the y value to the first value on the y axis
                            assign(y, float(y_volts[0]))
                            seq.ramp_to_voltages(
                                {x_obj.name: x, y_obj.name: y},
                                duration=node.parameters.per_line_wait,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                        # ── INNER LOOP: fast axis (y) ──────────────────────
                        with for_(*from_array(y, y_volts)):
                            seq.ramp_to_voltages(
                                {x_obj.name: x, y_obj.name: y},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            if node.parameters.pre_measurement_delay > 0:
                                # Additional delay before the measurement pulse
                                # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                                seq.step_to_voltages({}, duration=node.parameters.pre_measurement_delay)
                            align()

                            # Perform muliplexed measurements on the sensors
                            # A python for loop is used so that the measurements are performed in parallel.
                            for i, sensor in multiplexed_sensors.items():
                                # Select the resonator tied to the sensor
                                rr = sensor.readout_resonator
                                # Measure using the selected resonator
                                rr.measure("readout", qua_vars=(I[i], Q[i]))
                                # Post-measurement wait (Optional)
                                rr.wait(500)  # TODO: Make this a parameter

                                # Save the I/Q data to the streams
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        if node.parameters.per_line_compensation:
                            seq.ramp_to_zero()
                            # TODO: Add a compensation pulse
                    seq.ramp_to_zero()
            # ── Post-processing on the OPX before data reaches the PC ──
            with stream_processing():
                n_st.save("n")  # save the shot counter for the progress bar
                for i in range(num_sensors):
                    # The averaged data for each (x, y) pixel is saved to the streams
                    # Individual shots are not retained.
                    # .buffer(len(y)).buffer(len(x)) : group into 2D grid (y fast, x slow)
                    # .average() : average over all shots (n_avg repetitions)
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")

    # Case 2: X on QDAC (slow), Y on OPX (fast)
    elif x_external and not y_external:
        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            # Allocate real-time variables on the OPX:
            #   I[i], Q[i]       : demodulated quadratures for sensor i
            #   I_st[i], Q_st[i] : stream buffers before transfer to PC
            #   n, n_st          : shot counter exposed for the progress bar
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            x = declare(fixed)  # values on the x axis
            # y = declare(fixed) # values on the y axis

            for multiplexed_sensors in sensors.batch():
                align()
                # ── OUTER LOOP: average over shots ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # update the progress bar
                    # ── MIDDLE LOOP: slow axis (x) ─────────────────────────
                    with for_(*from_array(x, x_volts)):
                        # one QDAC trigger per x_volts step (dc_list advances X)
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        if node.parameters.per_line_wait > 0:
                            # Optional settle time at the start of each scan line
                            # Set the y value to the first value on the y axis
                            seq.ramp_to_voltages(
                                {y_obj.id: float(y_volts[0])},
                                duration=node.parameters.per_line_wait,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                        # Wait for QDAC to settle at the new X voltage
                        # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                        seq.step_to_voltages({}, duration=node.parameters.post_trigger_wait_ns)
                        # ── INNER LOOP: fast axis (y) ──────────────────────
                        # OPX ramps Y through y_volts (scan_mode sets order)
                        for y in scan_mode.inner_loop(y_volts):
                            seq.ramp_to_voltages(
                                {y_obj.id: y},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            align()
                            if node.parameters.pre_measurement_delay > 0:
                                # Additional delay before the measurement pulse
                                # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                                seq.step_to_voltages({}, duration=node.parameters.pre_measurement_delay)
                            align()

                            # Perform muliplexed measurements on the sensors
                            # A python for loop is used so that the measurements are performed in parallel.
                            for i, sensor in multiplexed_sensors.items():
                                # Select the resonator tied to the sensor
                                rr = sensor.readout_resonator
                                # Measure using the selected resonator
                                rr.measure("readout", qua_vars=(I[i], Q[i]))
                                # Post-measurement wait (Optional)
                                rr.wait(500)  # TODO: Make this a parameter

                                # Save data
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        if node.parameters.per_line_compensation:
                            seq.ramp_to_zero()
                            # TODO: Add a compensation pulse
                    seq.ramp_to_zero()
            # ── Post-processing on the OPX before data reaches the PC ──
            with stream_processing():
                n_st.save("n")  # save the shot counter for the progress bar
                for i in range(num_sensors):
                    # The averaged data for each (x, y) pixel is saved to the streams
                    # Individual shots are not retained.
                    # .buffer(len(y)).buffer(len(x)) : group into 2D grid (y fast, x slow)
                    # .average() : average over all shots (n_avg repetitions)
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(f"Q{i}")

    # Case 3: Y on QDAC (slow), X on OPX (fast)
    elif not x_external and y_external:
        # Transpose the measurement order so that the slow (Y) axis is on the outer loop and the fast (X) axis is on the inner loop
        node.namespace["sweep_axes"] = {
            "sensors": xr.DataArray(sensors.get_names()),
            "y_volts": xr.DataArray(
                (
                    y_volts
                    if (not node.parameters.dc_control and not y_external)
                    else y_volts + node.parameters.y_center
                ),
                attrs={"long_name": "voltage", "units": "V"},
            ),
            "x_volts": xr.DataArray(
                (
                    x_volts
                    if (not node.parameters.dc_control and not x_external)
                    else x_volts + node.parameters.x_center
                ),
                attrs={"long_name": "voltage", "units": "V"},
            ),
        }

        with program() as node.namespace["qua_program"]:
            seq = node.machine.voltage_sequences[vgs_id]

            # Allocate real-time variables on the OPX:
            #   I[i], Q[i]       : demodulated quadratures for sensor i
            #   I_st[i], Q_st[i] : stream buffers before transfer to PC
            #   n, n_st          : shot counter exposed for the progress bar
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)
            y = declare(fixed)

            for multiplexed_sensors in sensors.batch():
                align()
                # ── OUTER LOOP: average over shots ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # update the progress bar
                    # ── MIDDLE LOOP: slow axis (y) ─────────────────────────
                    # one QDAC trigger per y_volts step (dc_list advances Y)
                    with for_(*from_array(y, y_volts)):
                        y_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")
                        if node.parameters.per_line_wait > 0:
                            # Optional settle time at the start of each scan line
                            # Set the x value to the first value on the x axis
                            seq.ramp_to_voltages(
                                {x_obj.id: float(x_volts[0])},
                                duration=node.parameters.per_line_wait,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                        # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                        seq.step_to_voltages({}, duration=node.parameters.post_trigger_wait_ns)
                        # ── INNER LOOP: fast axis (x) ──────────────────────
                        # OPX ramps X through x_volts (scan_mode sets order)
                        for x in scan_mode.inner_loop(x_volts):
                            seq.ramp_to_voltages(
                                {x_obj.id: x},
                                duration=node.parameters.hold_duration,
                                ramp_duration=node.parameters.ramp_duration,
                            )
                            align()
                            if node.parameters.pre_measurement_delay > 0:
                                # Additional delay before the measurement pulse
                                # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                                seq.step_to_voltages({}, duration=node.parameters.pre_measurement_delay)
                            align()

                            # Perform muliplexed measurements on the sensors
                            # A python for loop is used so that the measurements are performed in parallel.
                            for i, sensor in multiplexed_sensors.items():
                                # Select the resonator tied to the sensor
                                rr = sensor.readout_resonator
                                # Measure using the selected resonator
                                rr.measure("readout", qua_vars=(I[i], Q[i]))
                                # Post-measurement wait (Optional)
                                rr.wait(500)  # TODO: Make this a parameter

                                # Save the I/Q data to the streams
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        if node.parameters.per_line_compensation:
                            seq.ramp_to_zero()
                            # TODO: Add a compensation pulse
                    seq.ramp_to_zero()
            # ── Post-processing on the OPX before data reaches the PC ──
            with stream_processing():
                n_st.save("n")  # save the shot counter for the progress bar
                for i in range(num_sensors):
                    # The averaged data for each (x, y) pixel is saved to the streams
                    # Individual shots are not retained.
                    # .buffer(len(x)).buffer(len(y)) : group into 2D grid (x fast, y slow)
                    # .average() : average over all shots (n_avg repetitions)
                    I_st[i].buffer(len(x_volts)).buffer(len(y_volts)).average().save(f"I{i}")
                    Q_st[i].buffer(len(x_volts)).buffer(len(y_volts)).average().save(f"Q{i}")

    # Case 4: both axes on the QDAC — OPX only triggers and reads out
    elif x_external and y_external:
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
