from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np

###################
# The QUA program #
###################

tau_min = 4
tau_max = 100
dtau = 2
taus = np.arange(tau_min, tau_max + dtau/2, dtau)  # + 0.1 to add t_max to taus

n_avg = 1e4
cooldown_time = 5 * qubit_T1 // 4

with program() as ramsey:
    n = declare(int)
    I = declare(fixed)
    I_st = declare_stream()
    Q = declare(fixed)
    Q_st = declare_stream()
    tau = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(tau, tau_min, tau <= tau_max, tau + dtau):
            play("pi_half", "qubit")
            wait(tau, "qubit")
            play("pi_half", "qubit")
            align("qubit", "resonator")
            measure("readout", "resonator", None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            wait(cooldown_time, "resonator")

    with stream_processing():
        I_st.buffer(len(taus)).average().save('I')
        Q_st.buffer(len(taus)).average().save('Q')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(ramsey)
res_handles = job.result_handles
I_handles = res_handles.get("I").fetch_all()
Q_handles = res_handles.get("Q").fetch_all()
I_handles.wait_for_values(1)
Q_handles.wait_for_values(1)

plt.figure()
while res_handles.is_processing():
    plt.cla()
    I = I_handles.fetch_all()
    Q = Q_handles.fetch_all()
    plt.plot(taus, I, '.', label='I')
    plt.plot(taus, Q, '.', label='Q')

    plt.legend()
    plt.show()
    plt.pause(0.1)


plt.cla()
I = I_handles.fetch_all()
Q = Q_handles.fetch_all()
plt.plot(taus, I, '.', label='I')
plt.plot(taus, Q, '.', label='Q')

plt.legend()
