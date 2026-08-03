"""QUA macros for trapped-ion single-ion experiments."""

from qm.qua import *

from configuration import (
    cooling_len,
    detection_len,
    detection_threshold,
    repump_len,
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


def measure_fluorescence(counts, times):
    play("constant", "detection", duration=detection_len // 4)
    measure("readout", "pmt", time_tagging.analog(times, detection_len, counts))


def measure_state(counts, times, state):
    measure_fluorescence(counts, times)
    assign(state, Cast.to_int(counts <= detection_threshold))
    

def plot_state_and_histogram(fig, ax_state, ax_hist, title, state_label_x, state_label_y, pop_x, pop, hist_label_x, hist_label_y, counts):
    ax_hist.cla()
    ax_state.cla()
    ax_hist.hist(counts, bins=30)
    ax_hist.axvline(detection_threshold, color="r", linestyle="--")
    ax_hist.set_xlabel(hist_label_x)
    ax_hist.set_ylabel(hist_label_y)
    ax_state.plot(pop_x, pop, "o")
    ax_state.set_xlabel(state_label_x)
    ax_state.set_ylabel(state_label_y)
    ax_state.set_ylim(0, 1)
    fig.suptitle(title)
    fig.tight_layout()

