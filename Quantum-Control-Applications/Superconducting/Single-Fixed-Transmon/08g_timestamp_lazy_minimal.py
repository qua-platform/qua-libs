"""
LAZY MINIMAL TIMESTAMP EXAMPLE

Same QUA program you would normally run (e.g. ``08b_power_rabi.py``). The only
fork is *after* ``with program()``:

  - Normal path:  ``qm.execute(power_rabi)`` and fetch I/Q as usual.
  - Timing path:  wrap once with ``TimestampRecorder``, execute the clone, call
                  ``print_shot()`` — no changes inside the QUA body.

Set ``measure_timing = True`` to take the timing fork; ``False`` runs the
experiment exactly like before.

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
amplitudes = np.linspace(0.2, 1.0, 5)

# False → run like 08b (execute the program, fetch averaged I/Q).
# True  → same program, but instrument a copy and print timing at one sweep point.
measure_timing = True

###################
# The QUA program #
###################
# Written once, unchanged — identical whether you measure timing or not.

with program() as power_rabi:
    n = declare(int)
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amplitudes)):
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
        I_st.buffer(n_avg).buffer(len(amplitudes)).map(FUNCTIONS.average()).save("I")
        Q_st.buffer(n_avg).buffer(len(amplitudes)).map(FUNCTIONS.average()).save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)
qm = qmm.open_qm(config)

if not measure_timing:
    ###########################
    # Normal path (like 08b)  #
    ###########################
    # You already do this: compile and run the program as written.
    job = qm.execute(power_rabi)

    result_handles = job.result_handles
    I = result_handles.get("I").fetch_all()
    Q = result_handles.get("Q").fetch_all()
    print(f"\nNormal experiment data: I shape {np.asarray(I).shape}, Q shape {np.asarray(Q).shape}")

else:
    ###########################
    # Optional timing path    #
    ###########################
    # The original ``power_rabi`` is not edited. TimestampRecorder builds an
    # instrumented clone; only the clone is executed on hardware.
    timing = TimestampRecorder(power_rabi, config=config)
    job = qm.execute(timing.program)  # note: timing.program, not power_rabi

    # Do not wait for the full I/Q sweep — timestamp streams are enough.
    result = timing.fetch(job, wait_for_all=False, timeout_s=120)

    # One call → play time, measure time, and play→measure gap at one sweep point.
    result.print_shot()          # shot=0: first avg, first amplitude
    result.print_shot((1, 2))    # shot=(avg_index, amp_index)

    # Experiment streams are still running in the background; cancel if you only
    # wanted timing and do not need the averaged I/Q vectors.
    try:
        job.cancel()
    except Exception:
        pass
