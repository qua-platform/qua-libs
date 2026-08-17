"""QUA macros for trapped-ion single-ion experiments."""

from qm.qua import *

from configuration import (
    cooling_len,
    detection_len,
    detection_threshold,
    repump_len,
    shelving_len,
    raman_b_if,
)

##############
# QUA macros #
##############


def doppler_cool():
    play("constant", "repump", duration=cooling_len // 4)
    play("constant", "cooling", duration=cooling_len // 4)
    align()


def state_preparation():
    play("constant", "repump", duration=repump_len // 4)
    align()


def shelve():
    play("constant", "shelving", duration=shelving_len // 4)
    align()


def play_raman(freq, operation, duration=None):
    """Play the same operation on both Raman beams simultaneously."""
    update_frequency("raman_a", freq-raman_b_if)
    align()
    if duration is None:
        play(operation, "raman_a")
        play(operation, "raman_b")
    else:
        play(operation, "raman_a", duration=duration)
        play(operation, "raman_b", duration=duration)
    align()


def measure_fluorescence(counts, times):
    play("constant", "detection", duration=detection_len // 4)
    measure("readout", "pmt", time_tagging.analog(times, detection_len, counts))


def measure_state(counts, times, state):
    measure_fluorescence(counts, times)
    assign(state, Cast.to_int(counts <= detection_threshold))
    

def plot_state_and_histogram(fig, ax_state, ax_hist, 
                             title, state_label_x, state_label_y, pop_x, pop, 
                             hist_label_x, hist_label_y, counts):
    ax_hist.cla()
    ax_hist.hist(counts, bins=30)
    ax_hist.axvline(detection_threshold, color="r", linestyle="--")
    ax_hist.set_xlabel(hist_label_x)
    ax_hist.set_ylabel(hist_label_y)
    
    ax_state.cla()
    ax_state.plot(pop_x, pop, "o")
    ax_state.set_xlabel(state_label_x)
    ax_state.set_ylabel(state_label_y)
    ax_state.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()

