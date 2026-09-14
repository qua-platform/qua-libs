"""Micromotion compensation by RF tickle of a radial mode.

A weak AC tone ("tickle") is applied to a trap electrode at the radial secular
frequency while the detection beam is on. On resonance the ion is driven into a
larger orbit, Doppler-broadens and moves off the detection beam, so the count
rate drops. An ion sitting away from the RF null picks up more RF field as it
swings, so the drop grows with the displacement: step the compensation voltage
and keep the voltage where the drop is smallest.

Each voltage is measured twice, with and without the tickle, and the result is
the relative dip (off - on) / off. That cancels slow laser-power drift, which
would otherwise look like a change in compensation.

Compared with 12_micromotion_compensation.py (photon-RF correlation): the
tickle needs no RF phase reference and no shared clock, just a counter and one
AC electrode, so it is the fast coarse method. It is comparative only - it
locates the minimum but gives no micromotion amplitude in physical units. Use
the correlation script for the quantitative measurement.

Before you run, adjust for your setup:
  - tickle_freq: the radial secular frequency of your trap. Measure it first
    with 13_motional_mode_identification.py; a wrong value means no resonance
    and a flat, meaningless scan.
  - tickle_scale: start small. Too strong ejects the ion or saturates the dip
    so every voltage looks equally bad; too weak gives no dip at all. Aim for
    roughly a 10-30 % drop at the starting voltage.
  - The "tickle" element in configuration.py: output port and tickle_amp.
  - The "comp_x"/"comp_y" elements: ports and voltage range of your electrodes.
    If your trap is biased by an external DAC, drive that instead of the OPX.
  - The detection beam should be near resonance, where motion reduces the
    count rate most steeply.

This is sensitive to the radial mode you drive. Repeat on the other radial mode
(and on comp_y) to null both directions.
"""

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.units import unit

from configuration import config, qop_ip, cluster_name, detection_len
from macros import doppler_cool, measure_fluorescence

u = unit(coerce_to_integer=True)

###############################################################################
# Set these for your experiment (hardware values live in configuration.py)    #
###############################################################################
tickle_freq = 2 * u.MHz  # radial secular frequency, from 13_motional_mode_identification
tickle_scale = 1.0  # scales tickle_amp; weak enough to keep the ion trapped
comp_voltages = np.round(np.linspace(-0.3, 0.3, 13), 4)  # comp_x scan [V]
comp_y_voltage = 0.0  # the other axis, held fixed during this scan [V]
n_shots = 100  # shots per voltage, per tickle state
settle_time = 200 * u.us  # electrode settling after a voltage step
simulate = False
###############################################################################

settle_cc = u.to_clock_cycles(settle_time)
probe_cc = detection_len // 4  # tickle covers the whole detection window

with program() as tickle_compensation:
    n = declare(int)
    v = declare(fixed)
    counts = declare(int)
    times = declare(int, size=1000)
    on_st = declare_stream()
    off_st = declare_stream()

    update_frequency("tickle", tickle_freq)
    set_dc_offset("comp_y", "single", comp_y_voltage)

    with for_(*from_array(v, comp_voltages)):
        set_dc_offset("comp_x", "single", v)
        wait(settle_cc)
        with for_(n, 0, n < n_shots, n + 1):
            # With tickle: it plays on its own element, so it overlaps the
            # detection beam and the PMT window started by measure_fluorescence.
            doppler_cool()
            play("constant" * amp(tickle_scale), "tickle", duration=probe_cc)
            measure_fluorescence(counts=counts, times=times)
            save(counts, on_st)

            # Reference shot, identical but without the tickle.
            doppler_cool()
            measure_fluorescence(counts=counts, times=times)
            save(counts, off_st)

    with stream_processing():
        on_st.buffer(n_shots).save_all("on")
        off_st.buffer(n_shots).save_all("off")

qmm = QuantumMachinesManager(host=qop_ip, cluster_name=cluster_name)

if simulate:
    job = qmm.simulate(config, tickle_compensation, SimulationConfig(duration=10_000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config, close_other_machines=True)
    try:
        job = qm.execute(tickle_compensation)
        job.result_handles.wait_for_all_values()

        def fetch(name):
            data = job.result_handles.get(name).fetch_all()
            if isinstance(data, dict):
                data = data["value"]
            return np.asarray(data, dtype=float)

        # Shape (voltages, shots) thanks to buffer(n_shots) in the stream processing.
        on, off = fetch("on"), fetch("off")
        mean_on, mean_off = on.mean(axis=1), off.mean(axis=1)
        dip = (mean_off - mean_on) / mean_off  # 0 = no response to the tickle

        best = int(np.argmin(dip))
        print(f"Reference counts {mean_off.mean():.1f} per shot")
        print(f"Smallest tickle dip {dip[best]:.3f} at comp_x = {comp_voltages[best]:.4f} V")

        fig, (ax_dip, ax_counts) = plt.subplots(1, 2, figsize=(11, 4))
        ax_dip.plot(comp_voltages, dip, "o-")
        ax_dip.axvline(comp_voltages[best], color="r", linestyle="--", label=f"min at {comp_voltages[best]:.4f} V")
        ax_dip.set_xlabel("comp_x voltage [V]")
        ax_dip.set_ylabel("Relative dip (off - on) / off")
        ax_dip.set_title(f"Tickle response at {tickle_freq / 1e6:.3f} MHz")
        ax_dip.legend()

        ax_counts.plot(comp_voltages, mean_off, "o-", label="tickle off")
        ax_counts.plot(comp_voltages, mean_on, "s-", label="tickle on")
        ax_counts.set_xlabel("comp_x voltage [V]")
        ax_counts.set_ylabel("Counts per shot")
        ax_counts.set_title("Raw fluorescence (check the ion is still trapped)")
        ax_counts.legend()

        fig.tight_layout()
        plt.show()

    finally:
        qm.close()
