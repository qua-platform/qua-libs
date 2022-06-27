"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weigths...) these macros will need to be modified accordingly.
"""

from qm.qua import *

##############
# QUA macros #
##############

# Macro for measuring the ground and excited states with single shot
def ge_singleshot_measurement(cooldown_time):
    """Macro measuring the singleshot qubit's ground and excited states.

    :param cooldown_time: cooldown time between two successive qubit state measurements in clock cycle unit (4ns).
    :return: singleshot I and Q data for the ground and excited states respectively [Ig, Qg, Ie, Qe].
    """
    Ig = declare(fixed)
    Qg = declare(fixed)
    Ie = declare(fixed)
    Qe = declare(fixed)

    # Ground state measurement
    align("qubit", "resonator")
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Ig),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Qg),
    )
    wait(cooldown_time, "resonator", "qubit")

    # Excited state measurement
    align("qubit", "resonator")
    play("pi", "qubit")
    align("qubit", "resonator")
    measure(
        "readout",
        "resonator",
        None,
        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", Ie),
        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Qe),
    )
    wait(cooldown_time, "resonator", "qubit")

    return Ig, Qg, Ie, Qe


# Macro for measuring the averaged ground and excited states for calibration
def ge_averaged_measurement(cooldown_time, n_avg):
    """Macro measuring the qubit's ground and excited states n_avg times. The averaged values for the corresponding I
    and Q quadratures can be retrieved using the stream processing context manager `Ig_st.average().save("Ig")` for instance.

    :param cooldown_time: cooldown time between two successive qubit state measurements in clock cycle unit (4ns).
    :param n_avg: number of averaging iterations. Must be a python integer.
    :return: streams for the I and Q data for the ground and excited states respectively: [Ig_st, Qg_st, Ie_st, Qe_st].
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
