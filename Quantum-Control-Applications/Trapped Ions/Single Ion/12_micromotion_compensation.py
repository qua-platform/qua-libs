"""Micromotion compensation by photon-RF correlation.

The trap RF shares a 10 MHz reference with the OPX, so a fixed OPX time is a
fixed RF phase. Photons are time-tagged while Doppler cooling, converted to
absolute time (window timestamp + tag) and folded modulo the RF period. Excess
micromotion shows up as a cosine modulation of the folded histogram: step the
compensation voltages until the modulation is flat (contrast -> 0).

The same PMT is split onto two elements ("pmt", "pmt_2"). Each tags at 50% duty
and they are staggered by one window, so together they cover the timeline
without gaps. Use pmt_elements = ("pmt",) if you only have one analog input.

Before you run, adjust for your setup:
  - trap_rf_freq (configuration.py): the ACTUAL trap drive frequency. A wrong
    value flattens the histogram and the measurement means nothing.
  - tag_to_ns (configuration.py): 0.5 ns on OPX1000 / QOP >= 3.5.0, else 1.0.
  - The "pmt" elements in configuration.py: analog input ports, time_of_flight
    and the timeTaggingParameters thresholds, which are detector specific.
  - The "comp_x"/"comp_y" elements: ports and voltage range of your electrodes.
    If your trap is biased by an external DAC, drive that instead of the OPX.
  - The cooling laser must be red-detuned on a steep part of the line, otherwise
    the micromotion Doppler shift produces little intensity modulation.

This measures micromotion only along the cooling-beam k-vector. Nulling both
radial directions needs a second, non-collinear beam (or a second viewport).
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.units import unit

from configuration import config, qop_ip, cluster_name, pmt_max_tags, rf_window_len, tag_to_ns, trap_rf_freq

u = unit(coerce_to_integer=True)

###############################################################################
# Set these for your experiment (hardware values live in configuration.py)    #
###############################################################################
comp_x_voltage = 0.0  # compensation electrode voltages [V]; step these to null
comp_y_voltage = 0.0  # the contrast printed at the end of the run
pmt_elements = ("pmt", "pmt_2")  # ("pmt",) for a single analog input
n_windows = 500  # tagging windows per element; more photons = lower noise floor
bins_per_period = 32
simulate = False
###############################################################################

window_cc = u.to_clock_cycles(rf_window_len)
rf_period_ns = 1e9 / trap_rf_freq
acq_cc = 2 * n_windows * window_cc

# Echo the two values that silently invalidate the result if they are wrong.
print(f"Trap RF {trap_rf_freq / 1e6:.6f} MHz (period {rf_period_ns:.3f} ns), time-tag unit {tag_to_ns} ns")

with program() as rf_correlation:
    n = declare(int)
    i = declare(int)
    tags = [declare(int, size=pmt_max_tags) for _ in pmt_elements]
    counts = [declare(int) for _ in pmt_elements]
    tags_st = [declare_stream() for _ in pmt_elements]
    counts_st = [declare_stream() for _ in pmt_elements]

    set_dc_offset("comp_x", "single", comp_x_voltage)
    set_dc_offset("comp_y", "single", comp_y_voltage)
    # No align() after this: the lasers stay on while the PMT elements tag.
    play("constant", "cooling", duration=acq_cc)
    play("constant", "repump", duration=acq_cc)

    for ch, element in enumerate(pmt_elements):
        wait(4 + ch * window_cc, element)  # offset channel 2 by one window
        with for_(n, 0, n < n_windows, n + 1):
            # The "tag" pulse is exactly one window long, so measure + wait give
            # 50% duty and the two channels interleave without overlapping.
            measure(
                "tag",
                element,
                time_tagging.analog(tags[ch], rf_window_len, counts[ch]),
                timestamp_stream=f"ts{ch}",  # window start, in clock cycles
            )
            save(counts[ch], counts_st[ch])
            with for_(i, 0, i < counts[ch], i + 1):
                save(tags[ch][i], tags_st[ch])
            if len(pmt_elements) > 1:
                wait(window_cc, element)  # off-half, covered by the other element

    with stream_processing():
        for ch in range(len(pmt_elements)):
            tags_st[ch].save_all(f"tags{ch}")
            counts_st[ch].save_all(f"counts{ch}")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

if simulate:
    job = qmm.simulate(config, rf_correlation, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    try:
        job = qm.execute(rf_correlation)
        job.result_handles.wait_for_all_values()

        def fetch(name):
            data = job.result_handles.get(name).fetch_all()
            if isinstance(data, dict):
                data = data["value"]
            return np.asarray(data).ravel()

        # Absolute arrival time = window start (4 ns per clock cycle) + tag.
        photon_ns = np.concatenate(
            [
                np.repeat(fetch(f"ts{ch}"), fetch(f"counts{ch}").astype(int)) * 4.0
                + fetch(f"tags{ch}") * tag_to_ns
                for ch in range(len(pmt_elements))
            ]
        )
        phase = np.mod(photon_ns, rf_period_ns) / rf_period_ns  # 0 to 1 within one RF cycle
        contrast = 2 * abs(np.mean(np.exp(2j * np.pi * phase)))
        # Unmodulated light still gives sqrt(pi/N) from shot noise. A contrast at
        # this level means "compensated as far as this many photons can tell".
        floor = np.sqrt(np.pi / len(photon_ns))
        print(f"{len(photon_ns)} photons, RF contrast = {contrast:.3f} (shot-noise floor {floor:.3f})")

        counts_per_bin, edges = np.histogram(phase, bins=bins_per_period, range=(0, 1))
        plt.bar(edges[:-1] * rf_period_ns, counts_per_bin, width=rf_period_ns / bins_per_period)
        plt.xlabel(f"Time within one RF cycle [ns] (period = {rf_period_ns:.2f} ns)")
        plt.ylabel("Counts")
        plt.title(f"RF correlation - contrast {contrast:.3f} at ({comp_x_voltage} V, {comp_y_voltage} V)")
        plt.axhline(len(photon_ns) / bins_per_period, color="k", linestyle="--", label="unmodulated mean")
        plt.legend()
        plt.tight_layout()
        plt.show()

    finally:
        qm.close()
