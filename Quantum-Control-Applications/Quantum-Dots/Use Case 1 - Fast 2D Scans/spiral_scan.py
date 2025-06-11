"""
Created on 14/12/2021
@author barnaby
"""

from qm.qua import *
from macros import (
    round_to_fixed,
    measurement_macro,
    measurement_macro_with_pulses,
    spiral_order,
)
import numpy as np
from scipy import signal
from qm import QuantumMachinesManager
from qm.simulate import SimulationConfig, LoopbackInterface
from configuration import config, qop_ip
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close

##############################
# Program-specific variables #
##############################
inserted_sequence = False  # True will use the measurement_macro_with_pulses macro with interleaved pulses
wait_after_pulse = 1000  # waiting time (ns) before measuring after applying the pulses

n_avg = 100  # Number of averaging loops
cooldown_time = 200 // 4  # Resonator cooldown time in clock cycles (4ns)

# Relevant elements
measured_element = "RF"
x_element = "LB"
y_element = "RB"

# 2D scan parameters
x_amp = 0.1  # The scan is defined as +/- x_amp/2
y_amp = 0.1  # The scan is defined as +/- y_amp/2
resolution = 11  # Number of points along x and y, it must be odd to form a spiral
x_axis = np.linspace(-x_amp / 2, x_amp / 2, resolution) * config["waveforms"]["jump"].get("sample")
y_axis = np.linspace(-y_amp / 2, y_amp / 2, resolution) * config["waveforms"]["jump"].get("sample")

# Perturbation parameters
ramp_to_zero_duration = 100
wait_time = 16 // 4

assert resolution % 2 == 1, "the resolution must be odd {}".format(resolution)

x_step_size = round_to_fixed(2 * x_amp / (resolution - 1))
y_step_size = round_to_fixed(2 * y_amp / (resolution - 1))

###################
# The QUA program #
###################
with program() as spiral_scan:
    i = declare(int)  # an index variable for the x index
    j = declare(int)  # an index variable for the y index

    x = declare(fixed)  # a variable to keep track of the x coordinate
    y = declare(fixed)  # a variable to keep track of the x coordinate
    x_st = declare_stream()
    y_st = declare_stream()
    average = declare(int)  # an index variable for the average
    moves_per_edge = declare(int)  # the number of moves per edge [1, resolution]
    completed_moves = declare(int)  # the number of completed move [0, resolution ** 2]
    movement_direction = declare(fixed)  # which direction to move {-1., 1.}

    # declaring the measured variables and their streams
    I, Q = declare(fixed), declare(fixed)
    I_stream, Q_stream = declare_stream(), declare_stream()

    with for_(average, 0, average < n_avg, average + 1):
        # initialising variables
        assign(moves_per_edge, 1)
        assign(completed_moves, 0)
        assign(movement_direction, +1)
        assign(x, 0.0)
        assign(y, 0.0)
        save(x, x_st)
        save(y, y_st)
        # for the first pixel it is unnecessary to move before measuring
        if inserted_sequence:
            measurement_macro_with_pulses(
                x_element=x_element,
                y_element=y_element,
                measured_element=measured_element,
                wait_before_meas=wait_after_pulse,
                I=I,
                I_stream=I_stream,
                Q=Q,
                Q_stream=Q_stream,
            )
        else:
            measurement_macro(
                measured_element=measured_element,
                I=I,
                I_stream=I_stream,
                Q=Q,
                Q_stream=Q_stream,
            )

        with while_(completed_moves < resolution * (resolution - 1)):
            # for_ loop to move the required number of moves in the x direction
            with for_(i, 0, i < moves_per_edge, i + 1):
                assign(x, x + movement_direction * x_step_size * 0.5)  # updating the x location
                save(x, x_st)
                save(y, y_st)
                # if the x coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
                with if_(x == 0.0):
                    ramp_to_zero(x_element, duration=ramp_to_zero_duration)
                # playing the constant pulse to move to the next pixel
                with else_():
                    play("jump" * amp(movement_direction * x_step_size), x_element)

                # Make sure that we measure after the pulse has settled
                align(x_element, y_element, measured_element)
                if wait_time >= 4:  # if logic to enable wait_time = 0 without error
                    wait(wait_time, measured_element)

                if inserted_sequence:
                    measurement_macro_with_pulses(
                        x_element=x_element,
                        y_element=y_element,
                        measured_element=measured_element,
                        wait_before_meas=wait_after_pulse,
                        I=I,
                        I_stream=I_stream,
                        Q=Q,
                        Q_stream=Q_stream,
                    )
                else:
                    measurement_macro(
                        measured_element=measured_element,
                        I=I,
                        I_stream=I_stream,
                        Q=Q,
                        Q_stream=Q_stream,
                    )

            # for_ loop to move the required number of moves in the y direction
            with for_(j, 0, j < moves_per_edge, j + 1):
                assign(y, y + movement_direction * y_step_size * 0.5)
                save(x, x_st)
                save(y, y_st)
                # if the y coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
                with if_(y == 0.0):
                    ramp_to_zero(y_element, duration=ramp_to_zero_duration)
                # playing the constant pulse to move to the next pixel
                with else_():
                    play("jump" * amp(movement_direction * y_step_size), y_element)

                # Make sure that we measure after the pulse has settled
                align(x_element, y_element, measured_element)
                if wait_time >= 4:  # if logic to enable wait_time = 0 without error
                    wait(wait_time, measured_element)

                if inserted_sequence:
                    measurement_macro_with_pulses(
                        x_element=x_element,
                        y_element=y_element,
                        measured_element=measured_element,
                        wait_before_meas=wait_after_pulse,
                        I=I,
                        I_stream=I_stream,
                        Q=Q,
                        Q_stream=Q_stream,
                    )
                else:
                    measurement_macro(
                        measured_element=measured_element,
                        I=I,
                        I_stream=I_stream,
                        Q=Q,
                        Q_stream=Q_stream,
                    )

            # updating the variables
            assign(completed_moves, completed_moves + 2 * moves_per_edge)  # * 2 because moves in both x and y
            assign(movement_direction, movement_direction * -1)  # *-1 as subsequent steps in the opposite direction
            assign(moves_per_edge, moves_per_edge + 1)  # moving one row/column out so need one more move_per_edge

        # filling in the final x row, which was not covered by the previous for_ loop
        with for_(i, 0, i < moves_per_edge - 1, i + 1):
            assign(x, x + movement_direction * x_step_size * 0.5)  # updating the x location
            save(x, x_st)
            save(y, y_st)
            # if the x coordinate should be 0, ramp to zero to remove fixed point arithmetic errors accumulating
            with if_(x == 0.0):
                ramp_to_zero(x_element, duration=ramp_to_zero_duration)
            # playing the constant pulse to move to the next pixel
            with else_():
                play("jump" * amp(movement_direction * x_step_size), x_element)

            # Make sure that we measure after the pulse has settled
            align(x_element, y_element, measured_element)
            if wait_time >= 4:
                wait(wait_time, measured_element)

            if inserted_sequence:
                measurement_macro_with_pulses(
                    x_element=x_element,
                    y_element=y_element,
                    measured_element=measured_element,
                    wait_before_meas=wait_after_pulse,
                    I=I,
                    I_stream=I_stream,
                    Q=Q,
                    Q_stream=Q_stream,
                )
            else:
                measurement_macro(
                    measured_element=measured_element,
                    I=I,
                    I_stream=I_stream,
                    Q=Q,
                    Q_stream=Q_stream,
                )

        # aligning and ramping to zero to return to initial state
        align(x_element, y_element, measured_element)
        # ramp_to_zero(x_element, duration=ramp_to_zero_duration)
        # ramp_to_zero(y_element, duration=ramp_to_zero_duration)

    with stream_processing():
        for stream_name, stream in zip(["I", "Q"], [I_stream, Q_stream]):
            stream.buffer(resolution**2).average().save(stream_name)
        x_st.save_all("x")
        y_st.save_all("y")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True

if simulation is True:
    simulation_duration = 400000  # ns

    job = qmm.simulate(
        config=config,
        program=spiral_scan,
        simulate=SimulationConfig(
            duration=int(simulation_duration // 4),
            include_analog_waveforms=True,
            simulation_interface=LoopbackInterface(
                latency=280,
                connections=[
                    ("con1", 1, "con1", 1),
                    ("con1", 2, "con1", 1),
                ],  # connecting output 4 to input 1
            ),
        ),
    )
    # plotting the waveform outputted by the OPX
    plt.figure("simulated output samples")
    output_samples = job.get_simulated_samples()
    output_samples.con1.plot()
    plt.show()

    # fetching the data
    result_handles = job.result_handles
    I_handle, Q_handle, x_handle, y_handle = (
        result_handles.I,
        result_handles.Q,
        result_handles.x,
        result_handles.y,
    )
    I, Q, x, y = (
        I_handle.fetch_all(),
        Q_handle.fetch_all(),
        x_handle.fetch_all(),
        y_handle.fetch_all(),
    )

    # reshaping the data into the correct order and shape
    order = spiral_order(np.sqrt(I.size))
    I = I[order]
    Q = Q[order]

    # quantifying the robustness of the method against the high-pass filtering effect of the bias-tee
    def butter_highpass_filter(data, tau):
        res = signal.butter(1, 1 / tau, btype="high", analog=False)
        return signal.lfilter(res[0], res[1], data)

    V = output_samples.con1.analog["1"]
    V_filter = butter_highpass_filter(V, tau=2000000)
    plt.figure()
    plt.plot(V)
    plt.plot(V_filter)
    print(f"Averaged error per step: {np.average(np.abs(V-V_filter)[:130000])*1000:.2f} mV")

else:
    # Open a quantum machine
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(spiral_scan)
    # fetch the data
    results = fetching_tool(job, ["I", "Q", "iteration"], mode="live")
    # Live plot
    fig = plt.figure()
    interrupt_on_close(fig, job)

    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.start_time)
        # reshaping the data into the correct order and shape
        order = spiral_order(np.sqrt(I.size))
        I = I[order]
        Q = Q[order]
        # Plot results
        plt.cla()
        plt.pcolor(x_axis, y_axis, np.sqrt(I**2 + Q**2))
        plt.title("Stability diagram with spiral scan")
        plt.xlabel(x_element + "_scan [V]")
        plt.ylabel(y_element + "_scan [V]")
