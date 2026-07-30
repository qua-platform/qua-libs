"""
POWER RABI ERROR-AMPLIFICATION LOOP TIMING WITH TIMESTAMP TOOLS

This example runs a small multi-loop power-Rabi-style program and uses the
automatic loop-index inference in ``timestamp_tools`` to extract the timestamps
for one selected sweep point inside the full program.

The program keeps nested loops (``n_avg``, pulse-count sweep, amplitude sweep,
and the inner error-amplification loop). ``TimestampRecorder`` records every
execution, and ``select_shot()`` maps one sweep coordinate back to the
corresponding timestamps.
"""

from pathlib import Path

import numpy as np
from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.results import DataHandler

from configuration import *
from timestamp_tools import TimestampRecorder

##################
#   Parameters   #
##################

n_avg = 3
amplitudes = np.linspace(0.95, 1.05, 5)
nb_of_pulses = np.array([1, 3, 5], dtype=int)

# Select one sweep point inside the nested loops.
target_avg_index = 1
target_pulse_index = 1
target_amp_index = 2

save_data_dict = {
    "n_avg": n_avg,
    "amplitudes": amplitudes,
    "nb_of_pulses": nb_of_pulses,
    "target_avg_index": target_avg_index,
    "target_pulse_index": target_pulse_index,
    "target_amp_index": target_amp_index,
    "config": config,
}


###################
# The QUA program #
###################

with program() as power_rabi_loop_timing:
    n = declare(int)
    a = declare(fixed)
    n_rabi = declare(int)
    n2 = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(n_rabi, nb_of_pulses)):
            with for_(*from_array(a, amplitudes)):
                with for_(n2, 0, n2 < n_rabi, n2 + 1):
                    play("x180" * amp(a), "qubit")

                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                wait(thermalization_time * u.ns, "resonator")
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).buffer(n_avg).save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).buffer(n_avg).save("Q")

###################
# Execute program #
###################

timing = TimestampRecorder(power_rabi_loop_timing, clock_cycle_ns=4)
measure_name = next(name for name in timing.names if name.startswith("measure_"))
measure_layout = timing.loop_mapper.layout(measure_name)

loop_indices = {
    0: target_avg_index,
    1: target_pulse_index,
    2: target_amp_index,
}

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
qm = qmm.open_qm(config)
job = qm.execute(timing.program)

timestamp_results = timing.fetch(job, wait_for_all=False, timeout_s=180)
selected_shot = timestamp_results.select_shot(loop_indices, reference=measure_name)
measure_shot = selected_shot[measure_name]
play_name = next(name for name in timing.names if name.startswith("play_"))
play_shot = selected_shot[play_name]

print("\nLoop axes inferred for measure statement")
print("-------------------------------------------------------------")
for axis_index, axis in enumerate(measure_layout.axes):
    print(
        f"axis {axis_index}: variable={axis.variable}, kind={axis.kind}, "
        f"size={axis.iteration_count}, values={axis.values}"
    )

print("\nSelected sweep point")
print("-------------------------------------------------------------")
print(f"requested indices: avg={target_avg_index}, pulses={target_pulse_index}, amp={target_amp_index}")
print(f"resolved loop indices: {measure_shot['loop_indices']}")
print(f"flat measure occurrence: {measure_shot['occurrence']}")
print(f"expected measure occurrence: {timestamp_results.occurrence_at(measure_name, loop_indices)}")

print("\nTimestamps at selected sweep point")
print("-------------------------------------------------------------")
print(
    f"{play_name}: {len(play_shot['nanoseconds'])} play timestamp(s) = "
    f"{np.asarray(play_shot['nanoseconds']).tolist()} ns"
)
print(f"{measure_name}: {measure_shot['time_ns']} ns")

if len(play_shot["nanoseconds"]) > 0:
    print(f"first play -> measure gap: {measure_shot['time_ns'] - play_shot['nanoseconds'][0]} ns")

print("\nSanity check against full timestamp history")
print("-------------------------------------------------------------")
print(f"total measure timestamps fetched: {timestamp_results[measure_name].occurrences}")
print(f"inferred total measure executions: {timestamp_results.expected_occurrences(measure_name)}")
print(f"total play timestamps fetched: {timestamp_results[play_name].occurrences}")
print(f"inferred total play executions: {timestamp_results.expected_occurrences(play_name)}")

script_name = Path(__file__).name
save_data_dict.update(
    {
        "selected_shot": {
            name: {
                key: (value.tolist() if hasattr(value, "tolist") else int(value) if key.endswith("_ns") or key.endswith("_cycle") else value)
                if not isinstance(value, slice)
                else {"start": value.start, "stop": value.stop}
                for key, value in payload.items()
            }
            for name, payload in selected_shot.items()
        },
        "loop_axes": [axis.__dict__ for axis in measure_layout.axes],
        "measure_name": measure_name,
        "play_name": play_name,
        "total_measure_timestamps": int(timestamp_results[measure_name].occurrences),
        "total_play_timestamps": int(timestamp_results[play_name].occurrences),
    }
)
data_handler = DataHandler(root_data_folder=save_dir)
data_handler.additional_files = {script_name: script_name, **default_additional_files}
data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

try:
    job.cancel()
except Exception:
    pass
