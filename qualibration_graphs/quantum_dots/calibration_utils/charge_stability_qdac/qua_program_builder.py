import numpy as np

from calibration_utils.common_utils import get_sensors
from calibration_utils.charge_stability_qdac.dc_list_prep import select_scan_trigger
from calibration_utils.charge_stability_qdac.helper_utils import build_sweep_axes

from qualang_tools.loops import from_array
from qm.qua import program, declare, align, save, fixed, for_, stream_processing, FUNCTIONS, wait
from qualibrate.core import QualibrationNode

__all__ = [
    "build_qua_program_with_mixed_axes",
]


def build_qua_program_with_mixed_axes(
    node: QualibrationNode,
    slow_axis_name: str,
    fast_axis_name: str,
    slow_axis_values: np.ndarray,
    fast_axis_values: np.ndarray,
    scan_mode,
):
    slow_obj = node.machine.get_component(slow_axis_name)
    fast_obj = node.machine.get_component(fast_axis_name)

    slow_axis_len = len(slow_axis_values)

    sensors = node.namespace["sensors"]
    num_sensors = len(sensors)

    n_avg = node.parameters.num_shots
    vgs_id = node.namespace["axes_names"]["gate_set_id"]
    virtual_dc_set = node.machine.virtual_dc_sets[vgs_id]
    _, trigger_gate_name = select_scan_trigger(slow_axis_name, slow_axis_name, virtual_dc_set)
    trigger_gate = virtual_dc_set.channels[trigger_gate_name]

    build_sweep_axes(
        node,
        x_volts=slow_axis_values if slow_axis_name == node.namespace["axes_names"]["x_axis"] else fast_axis_values,
        y_volts=slow_axis_values if slow_axis_name == node.namespace["axes_names"]["y_axis"] else fast_axis_values,
        slow_axis_name=slow_axis_name,
        fast_axis_name=fast_axis_name,
        scan_mode=scan_mode,
    )

    # ── QUA program (runs on the OPX in real time) ───────────────────────
    with program() as node.namespace["qua_program"]:
        seq = node.machine.voltage_sequences[vgs_id]

        # Allocate real-time variables on the OPX:
        #   I[i], Q[i]       : demodulated quadratures for sensor i
        #   I_st[i], Q_st[i] : stream buffers before transfer to PC
        #   n, n_st          : shot counter exposed for the progress bar
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables(num_IQ_pairs=num_sensors)

        slow_counter = declare(int)
        fast_value = declare(fixed)  # values on the x axis
        # y = declare(fixed) # values on the y axis

        for multiplexed_sensors in sensors.batch():
            align()

            # This defines the TOTAL HOLD DURATION on each pixel, INCLUDING the readout time.
            # So the duration for each pixel becomes hold_duration + longest readout time in this batch of multiplexed sensors
            pixel_hold_duration = node.parameters.hold_duration + max(
                s.readout_resonator.operations["readout"].length for s in multiplexed_sensors.values()
            )
            # ── OUTER LOOP: slow axis (x) ─────────────────────────
            with for_(slow_counter, 0, slow_counter < slow_axis_len, slow_counter + 1):
                # one QDAC trigger per x_volts step (dc_list advances X)
                trigger_gate.physical_channel.play("trigger")

                # ── MIDDLE LOOP: average over shots ───────────────────────
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)  # update the progress bar
                    seq.ramp_to_zero(ramp_duration=node.parameters.ramp_duration, reset_tracker=True)
                    align()

                    if node.parameters.per_line_wait > 0:
                        # Optional settle time at the start of each scan line
                        # Set the y value to the first value on the y axis
                        seq.ramp_to_voltages(
                            {fast_obj.name: float(fast_axis_values[0])},
                            duration=node.parameters.per_line_wait,
                            ramp_duration=node.parameters.ramp_duration,
                        )

                    # Wait for QDAC to settle at the new X voltage
                    # First a global wait command in clock cycles, and then ensure that the sticky duration for this time is also tracked.
                    wait(node.parameters.post_trigger_wait_ns // 4)
                    seq.track_sticky_duration(node.parameters.post_trigger_wait_ns)
                    # seq.step_to_voltages({}, duration=node.parameters.post_trigger_wait_ns)

                    # ── INNER LOOP: fast axis (y) ──────────────────────
                    # OPX ramps Y through y_volts (scan_mode sets order)
                    with for_(*from_array(fast_value, fast_axis_values)):
                        seq.ramp_to_voltages(
                            {fast_obj.name: fast_value},
                            duration=pixel_hold_duration,
                            ramp_duration=node.parameters.ramp_duration,
                        )

                        # Perform muliplexed measurements on the sensors
                        # A python for loop is used so that the measurements are performed in parallel.
                        for i, sensor in multiplexed_sensors.items():
                            # Select the resonator tied to the sensor
                            rr = sensor.readout_resonator
                            # Wait for the ramp + hold duration
                            rr.wait(
                                (
                                    node.parameters.post_trigger_wait_ns
                                    + node.parameters.ramp_duration
                                    + node.parameters.hold_duration
                                )
                                // 4
                            )
                            # Measure using the selected resonator
                            rr.measure("readout", qua_vars=(I[i], Q[i]))
                            # Post-measurement wait (Optional)
                            rr.wait(500)  # TODO: Make this a parameter

                            # Save data
                            save(I[i], I_st[i])
                            save(Q[i], Q_st[i])

                    if node.parameters.per_line_compensation:
                        seq.apply_compensation_pulse(max_voltage=node.parameters.max_compensation_voltage)
                        seq.ramp_to_zero(ramp_duration=node.parameters.ramp_duration, reset_tracker=True)

                seq.ramp_to_zero(ramp_duration=node.parameters.ramp_duration, reset_tracker=True)
        # ── Post-processing on the OPX before data reaches the PC ──
        with stream_processing():
            n_st.save("n")  # save the shot counter for the progress bar
            for i in range(num_sensors):
                # The averaged data for each (x, y) pixel is saved to the streams
                # Individual shots are not retained.
                # .buffer(len(y)).buffer(len(x)) : group into 2D grid (y fast, x slow)
                # .average() : average over all shots (n_avg repetitions)
                I_st[i].buffer(len(fast_axis_values)).buffer(n_avg).map(FUNCTIONS.average()).buffer(
                    len(slow_axis_values)
                ).save(f"I{i}")
                Q_st[i].buffer(len(fast_axis_values)).buffer(n_avg).map(FUNCTIONS.average()).buffer(
                    len(slow_axis_values)
                ).save(f"Q{i}")
