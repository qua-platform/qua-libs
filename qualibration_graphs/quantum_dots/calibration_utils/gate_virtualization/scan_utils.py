"""Common 2D scan program builder for gate virtualization nodes.

Encapsulates the four OPX/QDAC scan cases so that each node only needs
to specify which gates to sweep and then call ``create_2d_scan_program``.
"""

from typing import Dict, Tuple

import numpy as np
import xarray as xr
from qm.qua import (
    align,
    declare,
    fixed,
    for_,
    program,
    save,
    stream_processing,
    wait,
)
from qualang_tools.loops import from_array

from calibration_utils.charge_stability.scan_modes import ScanMode
from calibration_utils.gate_virtualization.parameters import (
    GateVirtualizationBaseParameters,
    get_voltage_arrays,
)


def _read_qdac_voltage(node, gate_obj) -> float:
    """Read the current DC voltage of a QDAC channel for *gate_obj*.

    Returns the voltage in the same units used by the QDAC (volts).
    """
    port = gate_obj.physical_channel.qdac_spec.qdac_output_port
    return float(node.machine.qdac.channel(port).read_dc())


def setup_qdac_dc_lists(
    node,
    x_obj,
    y_obj,
    x_volts: np.ndarray,
    y_volts: np.ndarray,
    scan_mode: ScanMode,
    x_external: bool,
    y_external: bool,
):
    """Configure QDAC DC lists for any axes driven by the QDAC.

    Parameters
    ----------
    node : QualibrationNode
        The active calibration node (provides ``node.machine``).
    x_obj, y_obj : VoltageGate-like
        Gate components for the X and Y axes.
    x_volts, y_volts : np.ndarray
        Voltage arrays for each axis.
    scan_mode : ScanMode
        The scan pattern (raster, switch_raster, ...).
    x_external, y_external : bool
        Whether each axis is driven by the QDAC.
    """
    if x_external:
        dc_list_x = node.machine.qdac.channel(
            x_obj.physical_channel.qdac_spec.qdac_output_port
        ).dc_list(
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
        dc_list_y = node.machine.qdac.channel(
            y_obj.physical_channel.qdac_spec.qdac_output_port
        ).dc_list(
            voltages=(
                np.tile(scan_mode.get_outer_loop(y_volts), len(x_volts))
                if x_external
                else scan_mode.get_outer_loop(y_volts)
            ),
            dwell_s=10e-6,
            stepped=True,
        )
        dc_list_y.start_on_external(trigger=1)


def _build_sweep_axes(
    sensors,
    x_volts: np.ndarray,
    y_volts: np.ndarray,
    x_axis_name: str,
    y_axis_name: str,
) -> Dict[str, xr.DataArray]:
    """Build the sweep_axes dict consumed by XarrayDataFetcher."""
    return {
        "sensors": xr.DataArray(sensors.get_names()),
        "x_volts": xr.DataArray(
            x_volts,
            attrs={"long_name": f"{x_axis_name} voltage", "units": "V"},
        ),
        "y_volts": xr.DataArray(
            y_volts,
            attrs={"long_name": f"{y_axis_name} voltage", "units": "V"},
        ),
    }


def _measurement_block(multiplexed_sensors, I, I_st, Q, Q_st, node):
    """Emit the QUA measurement + save block for all sensors."""
    align()
    for i, sensor in multiplexed_sensors.items():
        rr = sensor.readout_resonator
        rr.measure("readout", qua_vars=(I[i], Q[i]))
        rr.wait(500)
        save(I[i], I_st[i])
        save(Q[i], Q_st[i])


def create_2d_scan_program(node, sensors):
    """Create a QUA program for a 2D voltage scan.

    Supports four scan modes depending on ``x_from_qdac`` and ``y_from_qdac``
    parameter flags:
    - Both OPX (nested QUA loops)
    - X QDAC / Y OPX
    - X OPX / Y QDAC
    - Both QDAC (trigger-only loop)

    Parameters
    ----------
    node : QualibrationNode
        The active node. Must have ``node.parameters`` with the
        ``GateVirtualizationBaseParameters`` fields and ``node.machine``.
    sensors : BatchableList[SensorDot]
        Sensor dots to read out.

    Returns
    -------
    qua_program
        The compiled QUA program.
    sweep_axes : dict
        Sweep axis metadata for ``XarrayDataFetcher``.
    """
    params: GateVirtualizationBaseParameters = node.parameters

    x_obj = node.machine.get_component(params.x_axis_name)
    y_obj = node.machine.get_component(params.y_axis_name)

    if x_obj.voltage_sequence.gate_set.id != y_obj.voltage_sequence.gate_set.id:
        raise ValueError(
            f"X and Y axes belong to different VirtualGateSets. "
            f"x: {x_obj.voltage_sequence.gate_set.id}, "
            f"y: {y_obj.voltage_sequence.gate_set.id}"
        )
    vgs_id = x_obj.voltage_sequence.gate_set.id

    x_external = params.x_from_qdac
    y_external = params.y_from_qdac

    # For QDAC axes without an explicit centre, read the current DAC value
    # so the sweep is centred on the gate's operating point.
    x_center = params.x_center
    y_center = params.y_center
    if x_external and x_center is None:
        x_center = _read_qdac_voltage(node, x_obj)
    if y_external and y_center is None:
        y_center = _read_qdac_voltage(node, y_obj)

    x_volts, y_volts = get_voltage_arrays(node, x_center=x_center, y_center=y_center)
    num_sensors = len(sensors)
    scan_mode = ScanMode.from_name(params.scan_pattern)

    # ---- QDAC setup ----
    if x_external or y_external:
        node.machine.connect_to_external_source(external_qdac=True)
        setup_qdac_dc_lists(
            node, x_obj, y_obj, x_volts, y_volts, scan_mode, x_external, y_external
        )

    # ---- Sweep axes ----
    sweep_axes = _build_sweep_axes(
        sensors, x_volts, y_volts, params.x_axis_name, params.y_axis_name
    )

    qua_prog = None

    # ---- Case 1: Both OPX ----
    if not x_external and not y_external:
        with program() as qua_prog:
            seq = node.machine.voltage_sequences[vgs_id]
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                num_IQ_pairs=num_sensors
            )
            x = declare(fixed)
            y = declare(fixed)
            for multiplexed_sensors in sensors.batch():
                align()
                with for_(n, 0, n < params.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(x, x_volts)):
                        with for_(*from_array(y, y_volts)):
                            seq.ramp_to_voltages(
                                {x_obj.name: x, y_obj.name: y},
                                duration=params.hold_duration,
                                ramp_duration=params.ramp_duration,
                            )
                            if params.pre_measurement_delay > 0:
                                seq.step_to_voltages(
                                    {}, duration=params.pre_measurement_delay
                                )
                            _measurement_block(
                                multiplexed_sensors, I, I_st, Q, Q_st, node
                            )
                        if params.per_line_compensation:
                            seq.apply_compensation_pulse()
                    seq.apply_compensation_pulse()
            with stream_processing():
                n_st.save("n")
                for i in range(num_sensors):
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(
                        f"I{i}"
                    )
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(
                        f"Q{i}"
                    )

    # ---- Case 2: X QDAC / Y OPX ----
    elif x_external and not y_external:
        with program() as qua_prog:
            seq = node.machine.voltage_sequences[vgs_id]
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                num_IQ_pairs=num_sensors
            )
            x = declare(fixed)
            for multiplexed_sensors in sensors.batch():
                align()
                with for_(n, 0, n < params.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(x, x_volts)):
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play(
                            "trigger"
                        )
                        seq.step_to_voltages(
                            {}, duration=params.post_trigger_wait_ns
                        )
                        for y in scan_mode.inner_loop(y_volts):
                            seq.ramp_to_voltages(
                                {y_obj.id: y},
                                duration=params.hold_duration,
                                ramp_duration=params.ramp_duration,
                            )
                            if params.pre_measurement_delay > 0:
                                seq.step_to_voltages(
                                    {}, duration=params.pre_measurement_delay
                                )
                            _measurement_block(
                                multiplexed_sensors, I, I_st, Q, Q_st, node
                            )
                        if params.per_line_compensation:
                            seq.apply_compensation_pulse()
                    seq.apply_compensation_pulse()
            with stream_processing():
                n_st.save("n")
                for i in range(num_sensors):
                    I_st[i].buffer(len(y_volts)).average().buffer(
                        len(x_volts)
                    ).save(f"I{i}")
                    Q_st[i].buffer(len(y_volts)).average().buffer(
                        len(x_volts)
                    ).save(f"Q{i}")

    # ---- Case 3: X OPX / Y QDAC ----
    elif not x_external and y_external:
        sweep_axes = _build_sweep_axes(
            sensors, y_volts, x_volts, params.y_axis_name, params.x_axis_name
        )
        sweep_axes = {
            "sensors": sweep_axes["sensors"],
            "y_volts": xr.DataArray(
                y_volts,
                attrs={"long_name": f"{params.y_axis_name} voltage", "units": "V"},
            ),
            "x_volts": xr.DataArray(
                x_volts,
                attrs={"long_name": f"{params.x_axis_name} voltage", "units": "V"},
            ),
        }
        with program() as qua_prog:
            seq = node.machine.voltage_sequences[vgs_id]
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                num_IQ_pairs=num_sensors
            )
            y = declare(fixed)
            for multiplexed_sensors in sensors.batch():
                align()
                with for_(n, 0, n < params.num_shots, n + 1):
                    save(n, n_st)
                    with for_(*from_array(y, y_volts)):
                        y_obj.physical_channel.qdac_spec.opx_trigger_out.play(
                            "trigger"
                        )
                        seq.step_to_voltages(
                            {}, duration=params.post_trigger_wait_ns
                        )
                        for x in scan_mode.inner_loop(x_volts):
                            seq.ramp_to_voltages(
                                {x_obj.id: x},
                                duration=params.hold_duration,
                                ramp_duration=params.ramp_duration,
                            )
                            if params.pre_measurement_delay > 0:
                                seq.step_to_voltages(
                                    {}, duration=params.pre_measurement_delay
                                )
                            _measurement_block(
                                multiplexed_sensors, I, I_st, Q, Q_st, node
                            )
                        if params.per_line_compensation:
                            seq.apply_compensation_pulse()
                    seq.apply_compensation_pulse()
            with stream_processing():
                n_st.save("n")
                for i in range(num_sensors):
                    I_st[i].buffer(len(x_volts)).average().buffer(
                        len(y_volts)
                    ).save(f"I{i}")
                    Q_st[i].buffer(len(x_volts)).average().buffer(
                        len(y_volts)
                    ).save(f"Q{i}")

    # ---- Case 4: Both QDAC ----
    elif x_external and y_external:
        with program() as qua_prog:
            seq = node.machine.voltage_sequences[vgs_id]
            I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(
                num_IQ_pairs=num_sensors
            )
            trig_counter = declare(int)
            for multiplexed_sensors in sensors.batch():
                align()
                with for_(n, 0, n < params.num_shots, n + 1):
                    save(n, n_st)
                    with for_(
                        trig_counter,
                        0,
                        trig_counter < int(len(x_volts) * len(y_volts)),
                        trig_counter + 1,
                    ):
                        x_obj.physical_channel.qdac_spec.opx_trigger_out.play(
                            "trigger"
                        )
                        wait(params.post_trigger_wait_ns // 4)
                        _measurement_block(
                            multiplexed_sensors, I, I_st, Q, Q_st, node
                        )
            with stream_processing():
                n_st.save("n")
                for i in range(num_sensors):
                    I_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(
                        f"I{i}"
                    )
                    Q_st[i].buffer(len(y_volts)).buffer(len(x_volts)).average().save(
                        f"Q{i}"
                    )

    return qua_prog, sweep_axes
