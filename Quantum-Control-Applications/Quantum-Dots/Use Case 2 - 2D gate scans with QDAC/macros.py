"""
Created on 19/03/2023
@author jdh
"""

from qm.qua import *
from qualang_tools.loops import from_array
import numpy as np
import time


"""
QUA MACROS 
"""


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




