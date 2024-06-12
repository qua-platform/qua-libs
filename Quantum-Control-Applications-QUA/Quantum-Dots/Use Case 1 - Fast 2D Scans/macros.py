"""
Created on 14/12/2021
@author barnaby
"""

from qm.qua import *
import numpy as np


def round_to_fixed(x, number_of_bits=12):
    """
    function which rounds 'x' to 'number_of_bits' of precision to help reduce the accumulation of fixed point arithmetic errors
    """
    return round((2**number_of_bits) * x) / (2**number_of_bits)


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
    while total_moves < N**2 - N:
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
