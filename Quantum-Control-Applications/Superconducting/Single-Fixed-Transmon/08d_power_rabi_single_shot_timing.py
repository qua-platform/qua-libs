"""
POWER RABI SINGLE-SHOT TIMING

This diagnostic is derived from ``08c_power_rabi_error_amplification.py``.
It executes exactly one iteration of every loop and records the start time of
the single x180 play operation and the single readout measure operation.

Timestamp streams require QOP 2.2 or newer. The fetched values are expressed
in 4 ns clock cycles, so the host-side analysis converts them to nanoseconds.
The timestamps mark operation starts; pulse ends are inferred from the pulse
lengths in ``configuration.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.results.data_handler import DataHandler

from configuration import *

##################
#   Parameters   #
##################

CLOCK_CYCLE_NS = 4

# Each array and counter below contains exactly one iteration.
n_avg = 1
amplitudes = np.array([1.0])
nb_of_pulses = np.array([1], dtype=int)

save_data_dict = {
    "n_avg": n_avg,
    "amplitudes": amplitudes,
    "nb_of_pulses": nb_of_pulses,
    "clock_cycle_ns": CLOCK_CYCLE_NS,
    "config": config,
}


def fetched_values(raw_result):
    """Return result-handle data as a flat NumPy array across QOP API formats."""
    if isinstance(raw_result, dict) and "value" in raw_result:
        raw_result = raw_result["value"]
    elif getattr(getattr(raw_result, "dtype", None), "names", None) and "value" in raw_result.dtype.names:
        raw_result = raw_result["value"]
    return np.asarray(raw_result).reshape(-1)


def require_single_value(result_handles, handle_name):
    """Fetch one value and fail clearly if the nominal single shot was not single."""
    values = fetched_values(result_handles.get(handle_name).fetch_all())
    if values.size != 1:
        raise RuntimeError(f"Expected one value in '{handle_name}', received {values.size}: {values}")
    return values[0]


def analyze_timing(play_cycles, measure_cycles):
    """Convert timestamp cycles to ns and derive relative operation timing."""
    play_start_ns = int(play_cycles) * CLOCK_CYCLE_NS
    measure_start_ns = int(measure_cycles) * CLOCK_CYCLE_NS

    qubit_end_ns = play_start_ns + x180_len
    readout_pulse_end_ns = measure_start_ns + readout_len
    adc_acquisition_start_ns = measure_start_ns + time_of_flight
    adc_acquisition_end_ns = adc_acquisition_start_ns + readout_len
    measure_from_play_start_ns = measure_start_ns - play_start_ns
    align_to_measure_gap_ns = measure_start_ns - qubit_end_ns

    return {
        "play_start_ns": play_start_ns,
        "measure_start_ns": measure_start_ns,
        "qubit_end_ns": qubit_end_ns,
        "readout_pulse_end_ns": readout_pulse_end_ns,
        "adc_acquisition_start_ns": adc_acquisition_start_ns,
        "adc_acquisition_end_ns": adc_acquisition_end_ns,
        "measure_from_play_start_ns": measure_from_play_start_ns,
        "align_to_measure_gap_ns": align_to_measure_gap_ns,
        "shot_span_to_adc_end_ns": adc_acquisition_end_ns - play_start_ns,
    }


def print_timing_report(play_cycles, measure_cycles, timing):
    """Print absolute timestamps and relative timing after all fetching is done."""
    print("\nSingle-shot operation timing")
    print("-----------------------------------------------")
    print(f"x180 play start:       {int(play_cycles):>8} cycles = {timing['play_start_ns']:>10} ns")
    print(f"readout measure start: {int(measure_cycles):>8} cycles = {timing['measure_start_ns']:>10} ns")
    print("-----------------------------------------------")
    print(f"measure start - play start: {timing['measure_from_play_start_ns']} ns")
    print(f"expected x180 end:          {timing['qubit_end_ns']} ns")
    print(f"measure start - x180 end:   {timing['align_to_measure_gap_ns']} ns")
    print(f"expected readout pulse end: {timing['readout_pulse_end_ns']} ns")
    print(f"expected ADC capture start: {timing['adc_acquisition_start_ns']} ns")
    print(f"expected ADC capture end:   {timing['adc_acquisition_end_ns']} ns")
    print(f"play start -> ADC end:      {timing['shot_span_to_adc_end_ns']} ns")


def plot_timing(timing):
    """Plot both operations relative to the start of the x180 pulse."""
    measure_relative_ns = timing["measure_start_ns"] - timing["play_start_ns"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.broken_barh([(0, x180_len)], (20, 7), facecolors="tab:blue", label="x180 play")
    ax.broken_barh(
        [(measure_relative_ns, readout_len)],
        (5, 7),
        facecolors="tab:orange",
        label="readout measure",
    )
    ax.axvline(0, color="tab:blue", linestyle="--", linewidth=1)
    ax.axvline(measure_relative_ns, color="tab:orange", linestyle="--", linewidth=1)
    ax.set_yticks([8.5, 23.5], labels=["resonator", "qubit"])
    ax.set_xlabel("Time relative to x180 start [ns]")
    ax.set_title("Single-shot QUA operation timing")
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


###################
# The QUA program #
###################

with program() as power_rabi_single_shot_timing:
    n = declare(int)
    a = declare(fixed)
    n_rabi = declare(int)
    n2 = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # All four loops execute exactly once with the parameters defined above.
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(n_rabi, nb_of_pulses)):
            with for_(*from_array(a, amplitudes)):
                with for_(n2, 0, n2 < n_rabi, n2 + 1):
                    play("x180" * amp(a), "qubit", timestamp_stream="x180_timestamps")

                # align() makes the measurement wait for the qubit timeline.
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                    timestamp_stream="readout_timestamps",
                )
                wait(thermalization_time * u.ns, "resonator")
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.save("I")
        Q_st.save("Q")

#####################################
#  Open communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###################
# Execute program #
###################

qm = qmm.open_qm(config)
job = qm.execute(power_rabi_single_shot_timing)

# Fetch only after the one-shot program is complete.
result_handles = job.result_handles
result_handles.wait_for_all_values()

# Fetch each value explicitly using get().fetch_all() on separate lines
I_raw = result_handles.get("I").fetch_all()
Q_raw = result_handles.get("Q").fetch_all()
play_cycles_raw = result_handles.get("x180_timestamps").fetch_all()
measure_cycles_raw = result_handles.get("readout_timestamps").fetch_all()

I = float(fetched_values(I_raw))
Q = float(fetched_values(Q_raw))
play_cycles = int(fetched_values(play_cycles_raw))
measure_cycles = int(fetched_values(measure_cycles_raw))

I_volts = float(u.demod2volts(I, readout_len))
Q_volts = float(u.demod2volts(Q, readout_len))
timing = analyze_timing(play_cycles, measure_cycles)

print_timing_report(play_cycles, measure_cycles, timing)
print(f"I = {I_volts:.6g} V, Q = {Q_volts:.6g} V")
fig = plot_timing(timing)

# Save raw timestamps, converted timestamps, derived relative timing, and IQ.
script_name = Path(__file__).name
save_data_dict.update(
    {
        "timestamp_clock_cycles": {
            "x180_play": play_cycles,
            "readout_measure": measure_cycles,
        },
        "timestamp_ns": {
            "x180_play": timing["play_start_ns"],
            "readout_measure": timing["measure_start_ns"],
        },
        "relative_timing_ns": timing,
        "I_data": I,
        "Q_data": Q,
        "I_volts": I_volts,
        "Q_volts": Q_volts,
        "fig_timing": fig,
    }
)
data_handler = DataHandler(root_data_folder=save_dir)
data_handler.additional_files = {script_name: script_name, **default_additional_files}
data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

plt.show()
