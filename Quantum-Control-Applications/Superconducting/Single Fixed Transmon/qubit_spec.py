from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

n_avg = 10000

cooldown_time = 5 * qubit_T1 // 4

f_min = 20e6
f_max = 100e6
df = 0.1e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

with program() as qubit_spec:
    n = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(
            f, f_min, f <= f_max, f + df
        ):  # Notice it's <= to include f_max (This is only for integers!)
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
            wait(cooldown_time, "resonator")
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

###############
# Run Program #
###############
qm = qmm.open_qm(config)

job = qm.execute(qubit_spec)

res_handles = job.result_handles
I_handles = res_handles.get("I")
Q_handles = res_handles.get("Q")
I_handles.wait_for_values(1)
Q_handles.wait_for_values(1)

plt.figure()
while res_handles.is_processing():
    plt.cla()
    I = I_handles.fetch_all()
    Q = Q_handles.fetch_all()
    plt.plot(freqs, np.sqrt(I**2 + Q**2), ".")
    # plt.plot(freqs + qubit_LO, np.sqrt(I**2 + Q**2), '.')

    # If we want to plot the phase...
    phase = np.unwrap(np.angle(I + 1j * Q))

    plt.show()
    plt.pause(0.001)


plt.cla()
I = I_handles.fetch_all()
Q = Q_handles.fetch_all()
plt.plot(freqs, np.sqrt(I**2 + Q**2), ".")
# plt.plot(freqs + qubit_LO, np.sqrt(I**2 + Q**2), '.')

# If we want to plot the phase...
phase = np.unwrap(np.angle(I + 1j * Q))
