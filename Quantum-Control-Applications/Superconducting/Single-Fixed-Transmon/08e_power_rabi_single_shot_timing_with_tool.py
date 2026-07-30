"""
POWER RABI SINGLE-SHOT TIMING WITH QUALANG TOOLS

This example is the reusable-tool version of
``08d_power_rabi_single_shot_timing.py``. Every loop executes exactly once.
``TimestampRecorder`` consumes the completed, naturally written QUA program,
creates an instrumented copy automatically and converts fetched 4 ns clock
cycles to nanoseconds after hardware execution.

Timestamp streams require QOP 2.2 or newer.
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

n_avg = 1
amplitudes = np.array([1.0])
nb_of_pulses = np.array([1], dtype=int)

save_data_dict = {
    "n_avg": n_avg,
    "amplitudes": amplitudes,
    "nb_of_pulses": nb_of_pulses,
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

    # The QUA body is written normally: no timestamp-specific calls or names.
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
        I_st.save("I")
        Q_st.save("Q")

###################
# Execute program #
###################

# Instrument a copy after the natural QUA program has been fully constructed.
timing = TimestampRecorder(power_rabi_single_shot_timing, clock_cycle_ns=4)

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
qm = qmm.open_qm(config)
job = qm.execute(timing.program)

# This waits for program completion and returns both raw cycles and nanoseconds.
timestamp_results = timing.fetch(job)
timestamp_rows = timestamp_results.as_rows(reference=0)

result_handles = job.result_handles
I = float(require_single_value(result_handles, "I"))
Q = float(require_single_value(result_handles, "Q"))
I_volts = float(u.demod2volts(I, readout_len))
Q_volts = float(u.demod2volts(Q, readout_len))

print("\nSingle-shot operation timestamps")
print("-------------------------------------------------------------")
print("name       type       element      cycles       ns    relative ns")
for row in timestamp_rows:
    print(
        f"{row['name']:<10} {row['operation_type']:<10} {row['element']:<10} "
        f"{row['clock_cycle']:>8} {row['time_ns']:>8} {row['relative_ns']:>14}"
    )
print("-------------------------------------------------------------")
print(f"I = {I_volts:.6g} V, Q = {Q_volts:.6g} V")

script_name = Path(__file__).name
save_data_dict.update(
    {
        "timestamp_clock_cycles": timestamp_results.clock_cycles,
        "timestamp_ns": timestamp_results.nanoseconds,
        "timestamp_rows": timestamp_rows,
        "I_data": I,
        "Q_data": Q,
        "I_volts": I_volts,
        "Q_volts": Q_volts,
    }
)
data_handler = DataHandler(root_data_folder=save_dir)
data_handler.additional_files = {script_name: script_name, **default_additional_files}
data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
