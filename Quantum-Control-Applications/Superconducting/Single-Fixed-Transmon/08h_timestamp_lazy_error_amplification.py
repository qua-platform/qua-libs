"""
LAZY TIMESTAMP EXAMPLE — POWER RABI ERROR AMPLIFICATION

Same QUA program as ``08c_power_rabi_error_amplification.py``. The only fork is
*after* ``with program()``:

  - Normal path:  ``qm.execute(power_rabi_err)`` and fetch I/Q as usual.
  - Timing path:  wrap once with ``TimestampRecorder``, execute the clone, call
                  ``print_shot()`` — no changes inside the QUA body.

Set ``measure_timing = True`` to take the timing fork; ``False`` runs the
experiment exactly like 08c.

The inner error-amplification loop produces multiple play timestamps at one sweep
point; ``print_shot()`` lists each pulse start, length, end, and gaps.

Timestamp streams require QOP 2.2 or newer.
"""

import numpy as np
from qm import QuantumMachinesManager
from qm.qua import *
from qualang_tools.loops import from_array

from configuration import *
from timestamp_tools import TimestampRecorder

##################
#   Parameters   #
##################

# Small sweep so this demo finishes quickly; your real experiment can stay large.
n_avg = 3
amplitudes = np.linspace(0.95, 1.05, 5)
nb_of_pulses = np.array([1, 3, 5], dtype=int)

# False → run like 08c (execute the program, fetch averaged I/Q).
# True  → same program, but instrument a copy and print timing at sweep points.
measure_timing = True

###################
# The QUA program #
###################
# Written once, unchanged — identical whether you measure timing or not.

with program() as power_rabi_err:
    n = declare(int)
    a = declare(fixed)
    n_rabi = declare(int)
    n2 = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

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
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).average().save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(nb_of_pulses)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
qm = qmm.open_qm(config)

if not measure_timing:
    ###########################
    # Normal path (like 08c)  #
    ###########################
    job = qm.execute(power_rabi_err)

    result_handles = job.result_handles
    I = result_handles.get("I").fetch_all()
    Q = result_handles.get("Q").fetch_all()
    print(f"\nNormal experiment data: I shape {np.asarray(I).shape}, Q shape {np.asarray(Q).shape}")

else:
    ###########################
    # Optional timing path    #
    ###########################
    timing = TimestampRecorder(power_rabi_err, config=config)
    job = qm.execute(timing.program)

    result = timing.fetch(job, wait_for_all=False, timeout_s=180)

    # shot=(avg, pulse_count_index, amp_index)
    result.print_shot()              # (0, 0, 0): 1 play, first amp
    result.print_shot((0, 1, 2))     # (0, 1, 2): 3 plays at amp index 2
    result.print_shot((1, 2, 4))     # (1, 2, 4): 5 plays at last amp

    try:
        job.cancel()
    except Exception:
        pass
