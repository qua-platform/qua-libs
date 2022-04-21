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
amps = np.arange(a_min, a_max + da/2, da)  # + da/2 to add a_max to amplitudes


with program() as power_rabi:
    n = declare(int)
    a = declare(fixed)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(a, a_min, a < a_max + da/2, a + da):  # Notice it's + da/2 to include a_max (This is only for fixed!)
            play("gauss"*amp(a), "qubit", duration=x180_len//4)
            align("qubit", "resonator")
            measure("readout", "resonator", None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q))
            save(I, I_st)
            save(Q, Q_st)
            wait(cooldown_time, "resonator")

    with stream_processing():
        I_st.buffer(len(amps)).average().save('I')
        Q_st.buffer(len(amps)).average().save('Q')

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

qm = qmm.open_qm(config)

job = qm.execute(power_rabi)
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
    plt.plot(amps, I, '.', label='I')
    plt.plot(amps, Q, '.', label='Q')

    plt.legend()
    plt.show()
    plt.pause(0.1)


plt.cla()
I = I_handles.fetch_all()
Q = Q_handles.fetch_all()
plt.plot(amps, I, '.', label='I')
plt.plot(amps, Q, '.', label='Q')

plt.legend()

