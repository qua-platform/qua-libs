import numpy as np

from calibration_utils.common_utils import get_sensors
from calibration_utils.charge_stability_qdac.helper_utils import build_sweep_axes

from qualang_tools.loops import from_array
from qm.qua import program, declare, align, save, fixed, for_, stream_processing, int
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
            # ── OUTER LOOP: average over shots ───────────────────────
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)  # update the progress bar
                # ── MIDDLE LOOP: slow axis (x) ─────────────────────────
                with for_(slow_counter, 0, slow_counter < slow_axis_len, slow_counter + 1):

                    # one QDAC trigger per x_volts step (dc_list advances X)
                    slow_obj.physical_channel.qdac_spec.opx_trigger_out.play("trigger")

                    if node.parameters.per_line_wait > 0:
                        # Optional settle time at the start of each scan line
                        # Set the y value to the first value on the y axis
                        seq.ramp_to_voltages(
                            {fast_obj.name: float(fast_axis_values[0])},
                            duration=node.parameters.per_line_wait,
                            ramp_duration=node.parameters.ramp_duration,
                        )

                    # Wait for QDAC to settle at the new X voltage
                    # ``seq.step_to_voltages`` is required to ensure that the compensation pulse for the bias tee is properly calculated.
                    seq.step_to_voltages({}, duration=node.parameters.post_trigger_wait_ns)

                    # ── INNER LOOP: fast axis (y) ──────────────────────
                    # OPX ramps Y through y_volts (scan_mode sets order)
                    for fv in scan_mode.inner_loop(fast_axis_values):
                        seq.ramp_to_voltages(
                            {fast_obj.name: fv},
                            duration=node.parameters.hold_duration,
                            ramp_duration=node.parameters.ramp_duration,
                        )

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
                        seq.apply_compensation_pulse(max_voltage=node.parameters.max_compensation_voltage)
                        seq.ramp_to_zero()

                seq.ramp_to_zero()
        # ── Post-processing on the OPX before data reaches the PC ──
        with stream_processing():
            n_st.save("n")  # save the shot counter for the progress bar
            for i in range(num_sensors):
                # The averaged data for each (x, y) pixel is saved to the streams
                # Individual shots are not retained.
                # .buffer(len(y)).buffer(len(x)) : group into 2D grid (y fast, x slow)
                # .average() : average over all shots (n_avg repetitions)
                I_st[i].buffer(len(fast_axis_values)).buffer(len(slow_axis_values)).average().save(f"I{i}")
                Q_st[i].buffer(len(fast_axis_values)).buffer(len(slow_axis_values)).average().save(f"Q{i}")
