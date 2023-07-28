"""
qubit_spec_wide_range.py: Performs a wide 1D frequency sweep on the qubit, measuring the resonator while also sweeping
an external LO source.
In this version, the external LO source is being swept in the external loop in order to minimize run time.
"""
from time import sleep
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from qualang_tools.loops import from_array

###################
# The QUA program #
###################

n_avg = 100

cooldown_time = 5 * qubit_T1 // 4

f_min = 20e6
f_max = 100e6
df = 0.1e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

f_min_external = 3e9 - f_min
f_max_external = 4e9 - f_max
df_external = f_max - f_min
freqs_external = np.arange(f_min_external, f_max_external + 0.1, df_external)
frequency = np.array(np.concatenate([freqs + freqs_external[i] for i in range(len(freqs_external))]))

with program() as qubit_spec:
    i = declare(int)
    n = declare(int)
    n_st = declare_stream()
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(i, 0, i < len(freqs_external) + 1, i + 1):
        pause()  # This waits until it is resumed from python
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(f, freqs)):  # Notice it's <= to include f_max (This is only for integers!)
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
        save(i, n_st)

    with stream_processing():
        I_st.buffer(len(freqs)).buffer(n_avg).map(FUNCTIONS.average()).save("I")
        Q_st.buffer(len(freqs)).buffer(n_avg).map(FUNCTIONS.average()).save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

###############
# Run Program #
###############
qm = qmm.open_qm(config)

job = qm.execute(qubit_spec)
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
n_handle = res_handles.get("iteration")
skip_first = True
I_tot = []
Q_tot = []
for i in range(len(freqs_external) + 1):
    while not job.is_paused():
        sleep(0.1)
        pass
    if not skip_first:
        # This is the data from the last external frequency range
        I_handle.wait_for_values(1)
        Q_handle.wait_for_values(1)
        n_handle.wait_for_values(1)
        I = I_handle.fetch_all()
        Q = Q_handle.fetch_all()
        iteration = n_handle.fetch_all()
        I_tot.append(I)
        Q_tot.append(Q)
        # Progress bar
        progress_counter(iteration, len(freqs_external))
        # Plot results
        plt.subplot(211)
        plt.title("Qubit spectroscopy amplitude")
        plt.plot(freqs + freqs_external[i - 1], np.sqrt(I**2 + Q**2), ",")
        plt.xlabel("qubit frequency [MHz]")
        plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
        plt.subplot(212)
        # detrend removes the linear increase of phase
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        plt.title("Qubit spectroscopy phase")
        plt.plot(freqs + freqs_external[i - 1], phase, ",")
        plt.xlabel("qubit frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.pause(0.1)
        plt.tight_layout()
    else:
        skip_first = False
    lo_source.set_freq(freqs_external[i])  # Replace by your own function
    job.resume()

I = np.concatenate(I_tot)
Q = np.concatenate(Q_tot)

plt.figure()
plt.subplot(211)
plt.title("Qubit spectroscopy amplitude")
plt.plot(frequency, np.sqrt(I**2 + Q**2), ",")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
plt.subplot(212)
# detrend removes the linear increase of phase
phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
plt.title("Qubit spectroscopy phase")
plt.plot(frequency, phase, ",")
plt.xlabel("qubit frequency [MHz]")
plt.ylabel("Phase [rad]")
plt.pause(0.1)
plt.tight_layout()
