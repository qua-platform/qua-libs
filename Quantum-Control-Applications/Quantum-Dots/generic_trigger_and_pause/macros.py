"""
Created on 19/03/2023
@author jdh
"""

from qm.qua import *
import numpy as np


"""
QUA MACROS 
"""


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




