"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weights...) these macros will need to be modified accordingly.
"""

from qm.qua import *

##############
# QUA macros #
##############


# Macro for measuring the qubit state with single shot
def readout_macro(threshold=None, state=None, I=None, Q=None):
    """
    A macro for performing the single-shot readout, with the ability to perform state discrimination.
    If `threshold` is given, the information in the `I` quadrature will be compared against the threshold and `state`
    would be `True` if `I > threshold`.
    Note that it is assumed that the results are rotated such that all the information is in the `I` quadrature.

    :param threshold: Optional. The threshold to compare `I` against.
    :param state: A QUA variable for the state information, only used when a threshold is given.
        Should be of type `bool`. If not given, a new variable will be created
    :param I: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :param Q: A QUA variable for the information in the `Q` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: Three QUA variables populated with the results of the readout: (`state` (only if threshold is not None), `I`, `Q`)
    """
    if I is None:
        I = declare(fixed)
    if Q is None:
        Q = declare(fixed)
    if (threshold is not None) and (state is None):
        state = declare(bool)
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "rotated_sin", I),
        dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
    )
    if threshold is not None:
        assign(state, I > threshold)
        return state, I, Q
    else:
        return I, Q


# Macro for measuring the averaged ground and excited states for calibration
def ge_averaged_measurement(cooldown_time, n_avg):
    """Macro measuring the qubit's ground and excited states n_avg times. The averaged values for the corresponding I
    and Q quadratures can be retrieved using the stream processing context manager `Ig_st.average().save("Ig")` for instance.

    :param cooldown_time: cooldown time between two successive qubit state measurements in clock cycle unit (4ns).
    :param n_avg: number of averaging iterations. Must be a python integer.
    :return: streams for the 'I' and 'Q' data for the ground and excited states respectively: [Ig_st, Qg_st, Ie_st, Qe_st].
    """
    n = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()
    with for_(n, 0, n < n_avg, n + 1):
        # Ground state calibration
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("cos", "sin", I),
            dual_demod.full("minus_sin", "cos", Q),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(I, Ig_st)
        save(Q, Qg_st)

        # Excited state calibration
        align("qubit", "resonator")
        play("x180", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("cos", "sin", I),
            dual_demod.full("minus_sin", "cos", Q),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(I, Ie_st)
        save(Q, Qe_st)

        return Ig_st, Qg_st, Ie_st, Qe_st
