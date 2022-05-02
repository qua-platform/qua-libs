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

a_min = 0.0
a_max = 1.0
da = 0.05
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes


with program() as power_rabi:
    n = declare(int)
    n_st = declare_stream()
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's + da/2 to include a_max (This is only for fixed!)
        with for_(a, a_min, a < a_max + da / 2, a + da):
            play("gauss" * amp(a), "qubit", duration=x180_len // 4)
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
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(amps)).average().save("I")
        Q_st.buffer(len(amps)).average().save("Q")
        n_st.save('iteration')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(power_rabi)
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
iteration_handle = res_handles.get('iteration')
I_handle.wait_for_values(1)
Q_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)
next_percent = 0.1  # First time print 10%


def on_close(event):
    event.canvas.stop_event_loop()
    job.halt()


f = plt.figure()
f.canvas.mpl_connect("close_event", on_close)
print('Progress =', end=' ')

while res_handles.is_processing():
    plt.cla()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    if iteration / n_avg > next_percent:
        percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
        print(f'{percent}%', end=' ')
        next_percent = percent / 100 + 0.1  # Print every 10%

    plt.plot(amps, I, ".", label="I")
    plt.plot(amps, Q, ".", label="Q")

    plt.legend()
    plt.pause(0.1)


plt.cla()
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
iteration = iteration_handle.fetch_all()
print(f'{round(iteration/n_avg * 100)}%')
plt.plot(amps, I, ".", label="I")
plt.plot(amps, Q, ".", label="Q")

plt.legend()
