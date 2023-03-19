"""
Created on 14/12/2021
@author barnaby
@author jdh
"""

from qm.qua import *
from qualang_tools.loops import from_array
import numpy as np
import time


"""
QUA MACROS 
"""


def round_to_fixed(x, number_of_bits=12):
    """
    function which rounds 'x' to 'number_of_bits' of precision to help reduce the accumulation of fixed point arithmetic errors
    """
    return round((2 ** number_of_bits) * x) / (2 ** number_of_bits)


def measurement_macro_with_pulses(x_element, y_element, measured_element, wait_before_meas, I, I_stream, Q, Q_stream):
    # jump downwards diagonally to initialise the spin state
    play("jump" * amp(+0.05), x_element, duration=100)
    play("jump" * amp(-0.05), y_element, duration=100)

    # jump upwards diagonally to potentially move to the S-T_ avoided crossing
    play("jump" * amp(-0.1), x_element, duration=100)
    play("jump" * amp(+0.1), y_element, duration=100)

    # return to initial value
    play("jump" * amp(+0.05), x_element, duration=100)
    play("jump" * amp(-0.05), y_element, duration=100)

    align(x_element, y_element, measured_element)

    # wait for 1us before measuring
    # wait(wait_before_meas // 4, measured_element)

    measure("measure", measured_element, None, demod.full("cos", I), demod.full("sin", Q))
    save(I, I_stream)
    save(Q, Q_stream)


def measurement_macro(measured_element, I, I_stream, Q, Q_stream):
    measure(
        "measure",
        measured_element,
        None,
        demod.full("cos", I),
        demod.full("sin", Q),
    )
    save(I, I_stream)
    save(Q, Q_stream)


def spiral_order(N: int):
    # casting to int if necessary
    if not isinstance(N, int):
        N = int(N)
    # asserting that N is odd
    N = N if N % 2 == 1 else N + 1

    # setting i, j to be in the middle of the image
    i, j = (N - 1) // 2, (N - 1) // 2

    # creating array to hold the ordering
    order = np.zeros(shape=(N, N), dtype=int)

    sign = +1  # the direction which to move along the respective axis
    number_of_moves = 1  # the number of moves needed for the current edge
    total_moves = 0  # the total number of moves completed so far

    # spiralling outwards along x edge then y
    while total_moves < N ** 2 - N:
        for _ in range(number_of_moves):
            i = i + sign  # move one step in left (sign = -1) or right (sign = +1)
            total_moves = total_moves + 1
            order[i, j] = total_moves  # updating the ordering array

        for _ in range(number_of_moves):
            j = j + sign  # move one step in down (sign = -1) or up (sign = +1)
            total_moves = total_moves + 1
            order[i, j] = total_moves
        sign = sign * -1  # the next moves will be in the opposite direction
        number_of_moves = number_of_moves + 1  # the next edges will require one more step

    # filling the final x edge, which cannot cleanly be done in the above while loop
    for _ in range(number_of_moves - 1):
        i = i + sign  # move one step in left (sign = -1) or right (sign = +1)
        total_moves = total_moves + 1
        order[i, j] = total_moves

    return order


def do2d(x_element, x_amplitude, x_resolution,
         y_element, y_amplitude, y_resolution,
         n_averages, I, Q, I_stream, Q_stream,
         x_stream, y_stream, wait_time):
    """
    Performs a two-dimensional raster scan for a stability diagram measurement, for instance.

    The x_amplitude and y_amplitude variables are set to the desired output of the OPX
    channels. The amplitude of the waveform configured in the config is taken into account
    and rescaled to make it the desired value set in these variables.

    """

    x_axis = np.linspace(-x_amplitude / 2, x_amplitude / 2, x_resolution)
    y_axis = np.linspace(-y_amplitude / 2, y_amplitude / 2, y_resolution)

    x = declare(fixed)
    y = declare(fixed)

    # step size for each axis
    dx = round_to_fixed((x_amplitude) / (x_resolution - 1))
    dy = round_to_fixed((y_amplitude) / (y_resolution - 1))

    # flags for checking if we are measuring the first element (in which case measure but do not move)
    y_move_flag = declare(bool)
    x_move_flag = declare(bool)

    # variable for averages
    n = declare(int)

    # averaging loop
    with for_(n, 0, n < n_averages, n + 1):
        # set the x axis to the starting value
        play('constant' * amp(x_axis[0]), x_element)

        # assign the x flag to false (do not move for first iteration)
        assign(x_move_flag, False)

        with for_(*from_array(x, x_axis)):
            play('constant' * amp(dx), x_element, condition=x_move_flag)

            play('constant' * amp(y_axis[0]), y_element)
            assign(y_move_flag, False)

            with for_(*from_array(y, y_axis)):
                # make sure that we measure after the pulse has settled
                if wait_time >= 4:  # if logic to enable wait_time = 0 without error
                    wait(wait_time, "RF")

                play("constant" * amp(dy), y_element, condition=y_move_flag)

                measure(
                    "measure",
                    "RF",
                    None,
                    demod.full("cos", I),
                    demod.full("sin", Q),
                )

                save(I, I_stream)
                save(Q, Q_stream)
                save(y, y_stream)
                save(x, x_stream)

                assign(y_move_flag, True)

            ramp_to_zero(y_element)
            assign(x_move_flag, True)

        ramp_to_zero(x_element)

def generic_macro(variable, stream):
    measure(
        "measure",
        "RF",
        None,
        demod.full("cos", variable),
    )

    save(variable, stream)

"""
PYTHON MACROS
"""

def reshape_for_do2d(data: np.ndarray, qdac_x_resolution, qdac_y_resolution, opx_x_resolution,
                     opx_y_resolution):
    """
    Reshapes data from a large do2d scan using the opx and qdac. This is necessary because the averaging cannot take
    place on the opx in this case due to a quirk of the averaging protocol in stream processing.
    """

    to_stack = data.reshape(qdac_x_resolution, qdac_y_resolution, opx_x_resolution, opx_y_resolution)
    stacked = np.hstack([np.vstack(array) for array in to_stack])

    return stacked




