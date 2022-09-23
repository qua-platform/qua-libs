"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weights...) these macros will need to be modified accordingly.
"""

from qm.qua import *

##############
# QUA macros #
##############


def reset_qubit(method, **kwargs):
    """
    Macro to reset the qubit state.

    If method is 'cooldown', then the variable cooldown_time (in clock cycles) must be provided as a python integer > 4.

    **Example**: reset_qubit('cooldown', cooldown_times=500)

    If method is 'active', then 3 parameters are available as listed below.

    **Example**: reset_qubit('active', threshold=-0.003, max_tries=3)

    :param method: Method the reset the qubit state. Can be either 'cooldown' or 'active'.
    :type method: str
    :key cooldown_time: qubit relaxation time in clock cycle, needed if method is 'cooldown'. Must be an integer > 4.
    :key threshold: threshold to discriminate between the ground and excited state, needed if method is 'active'.
    :key max_tries: python integer for the maximum number of tries used to perform active reset,
        needed if method is 'active'. Must be an integer > 0 and default value is 1.
    :key Ig: A QUA variable for the information in the `I` quadrature used for active reset. If not given, a new
        variable will be created. Must be of type `Fixed`.
    :return:
    """
    if method == "cooldown":
        # Check cooldown_time
        cooldown_time = kwargs.get("cooldown_time", None)
        if (cooldown_time is None) or (cooldown_time < 4):
            raise Exception("'cooldown_time' must be an integer > 4 clock cycles")
        # Reset qubit state
        wait(cooldown_time, "qubit")
    elif method == "active":
        # Check threshold
        threshold = kwargs.get("threshold", None)
        if threshold is None:
            raise Exception("'threshold' must be specified for active reset.")
        # Check max_tries
        max_tries = kwargs.get("max_tries", 1)
        if (max_tries is None) or (not float(max_tries).is_integer()) or (max_tries < 1):
            raise Exception("'max_tries' must be an integer > 0.")
        # Check Ig
        Ig = kwargs.get("Ig", None)
        # Reset qubit state
        return active_reset(threshold, max_tries=max_tries, Ig=Ig)


# Macro for performing active reset until successful for a given number of tries.
def active_reset(threshold, max_tries=1, Ig=None):
    """Macro for performing active reset until successful for a given number of tries.

    :param threshold: threshold for the 'I' quadrature discriminating between ground and excited state.
    :param max_tries: python integer for the maximum number of tries used to perform active reset. Must >= 1.
    :param Ig: A QUA variable for the information in the `I` quadrature. Should be of type `Fixed`. If not given, a new
        variable will be created
    :return: A QUA variable for the information in the `I` quadrature and the number of tries after success.
    """
    if Ig is None:
        Ig = declare(fixed)
    if (max_tries < 1) or (not float(max_tries).is_integer()):
        raise Exception("max_count must be an integer >= 1.")
    # Initialize Ig to be > threshold
    assign(Ig, threshold + 2**-28)
    # Number of tries for active reset
    counter = declare(int)
    # Reset the number of tries
    assign(counter, 0)

    # Perform active feedback
    align("qubit", "resonator")
    # Use a while loop and counter for other protocols and tests
    with while_((Ig > threshold) & (counter < max_tries)):
        # Measure the resonator
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Ig),
        )
        # Play a pi pulse to get back to the ground state
        play("pi", "qubit", condition=(Ig > threshold))
        # Increment the number of tries
        assign(counter, counter + 1)
    return Ig, counter


# Macro for measuring the qubit state with single shot
def single_measurement(threshold=None, state=None, I=None, Q=None):
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
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
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
            dual_demod.full("cos", "out1", "sin", "out2", I),
            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(I, Ig_st)
        save(Q, Qg_st)

        # Excited state calibration
        align("qubit", "resonator")
        play("pi", "qubit")
        align("qubit", "resonator")
        measure(
            "readout",
            "resonator",
            None,
            dual_demod.full("cos", "out1", "sin", "out2", I),
            dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(I, Ie_st)
        save(Q, Qe_st)

        return Ig_st, Qg_st, Ie_st, Qe_st
