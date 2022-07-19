"""
This file contains useful QUA macros meant to simplify and ease QUA programs.
All the macros below have been written and tested with the basic configuration. If you modify this configuration
(elements, operations, integration weigths...) these macros will need to be modified accordingly.
"""


def get_c2c_time(job, pulse1, pulse2):
    """
    Returns the center-to-center time between two pulses. The calculation is based from the simulation.

    :param job: a simulation ``QmJob`` object. ex: job = qmm.simulate()
    :param pulse1: tuple containing the element and pulse number for the 1st pulse. Note that if the element contain an IQ pair, then the pulse numbers have to be counted in pairs. ex: ('ensemble', 2) correspond to the 2nd pulse played from the element 'ensemble' since the numbers '0' and '1' are the I and Q components of the first pulse.
    :param pulse2: tuple containing the element and pulse number for the 2nd pulse. ex: ('resonator', 0)
    :return: center-to-center time (in ns) between the two pulses.  Note that if the element contains an IQ pair, then the pulse numbers have to be counted in pairs. ex: ('ensemble', 2) correspond to the 2nd pulse played from the element 'ensemble' since the numbers '0' and '1' are the I and Q components of the first pulse.
    """
    analog_wf = job.simulated_analog_waveforms()
    element1 = pulse1[0]
    pulse_nb1 = pulse1[1]
    element2 = pulse2[0]
    pulse_nb2 = pulse2[1]

    time2 = (
        analog_wf["elements"][element2][pulse_nb2]["timestamp"]
        + analog_wf["elements"][element2][pulse_nb2]["duration"] / 2
    )
    time1 = (
        analog_wf["elements"][element1][pulse_nb1]["timestamp"]
        + analog_wf["elements"][element1][pulse_nb1]["duration"] / 2
    )

    return time2 - time1
