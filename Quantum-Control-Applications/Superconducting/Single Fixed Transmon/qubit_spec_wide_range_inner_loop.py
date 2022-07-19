"""
qubit_spec_wide_range_inner_loop.py: Performs a wide 1D frequency sweep on the qubit, measuring the resonator while also
sweeping an external LO source.
In this version, the external LO source is being swept in the inner loop in order to minimize noise.
"""
from time import sleep
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

n_avg = 1000

cooldown_time = 5 * qubit_T1 // 4

f_min = 20e6
f_max = 100e6
df = 0.1e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

f_min_external = 3e9 - f_min
f_max_external = 8e9 - f_max
df_external = f_max - f_min
freqs_external = np.arange(f_min_external, f_max_external + 0.1, df_external)

with program() as qubit_spec:
    i = declare(int)
    n = declare(int)
    n_st = declare_stream()
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(i, 0, i < len(freqs_external), i + 1):
            pause()  # This waits until it is resumed from python
            with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
                update_frequency("qubit", f)
                play("saturation", "qubit")
                align("qubit", "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                save(I, I_st)
                save(Q, Q_st)
                wait(cooldown_time, "resonator")

    with stream_processing():
        I_st.buffer(len(freqs)).buffer(len(freqs_external)).average().save("I")
        Q_st.buffer(len(freqs)).buffer(len(freqs_external)).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

###############
# Run Program #
###############
qm = qmm.open_qm(config)

job = qm.execute(qubit_spec)

res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
skip_first = True


for i in range(n_avg):
    for j in range(len(freqs_external)):
        while not job.is_paused():
            sleep(0.1)
            pass
        if not skip_first:
            # This is the latest data
            I = I_handle.fetch_all()
            Q = Q_handle.fetch_all()
        else:
            skip_first = False

        lo_source.set_freq(freqs_external[j])  # Replace by your own function
        job.resume()
