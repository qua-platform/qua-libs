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

t_min = 10
t_max = 1000
dt = 10
taus = np.arange(t_min, t_max + 0.1, dt)  # + 0.1 to add t_max to taus


with program() as time_rabi:
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's <= to include t_max (This is only for integers!)
        with for_(t, t_min, t <= t_max + dt / 2, t + dt):
            play("gauss" * amp(x180_amp / gauss_amp), "qubit", duration=t)
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
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(time_rabi)
res_handles = job.result_handles
I_handle = res_handles.get("I")
Q_handle = res_handles.get("Q")
iteration_handle = res_handles.get("iteration")
I_handle.wait_for_values(1)
Q_handle.wait_for_values(1)
iteration_handle.wait_for_values(1)
next_percent = 0.1  # First time print 10%


def on_close(event):
    event.canvas.stop_event_loop()
    job.halt()


f = plt.figure()
f.canvas.mpl_connect("close_event", on_close)
print("Progress =", end=" ")

while res_handles.is_processing():
    plt.cla()
    I = I_handle.fetch_all()
    Q = Q_handle.fetch_all()
    iteration = iteration_handle.fetch_all()
    if iteration / n_avg > next_percent:
        percent = 10 * round(iteration / n_avg * 10)  # Round to nearest 10%
        print(f"{percent}%", end=" ")
        next_percent = percent / 100 + 0.1  # Print every 10%

    plt.plot(4 * taus, I, ".", label="I")
    plt.plot(4 * taus, Q, ".", label="Q")

    plt.legend()
    plt.pause(0.1)


plt.cla()
I = I_handle.fetch_all()
Q = Q_handle.fetch_all()
iteration = iteration_handle.fetch_all()
print(f"{round(iteration/n_avg * 100)}%")
plt.plot(4 * taus, I, ".", label="I")
plt.plot(4 * taus, Q, ".", label="Q")

plt.legend()
