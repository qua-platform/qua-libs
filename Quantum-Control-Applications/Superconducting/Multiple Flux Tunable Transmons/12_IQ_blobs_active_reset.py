from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import multiplexed_readout, reset_qubit


###################
# The QUA program #
###################
pts = 20000
cooldown_time = 1 * u.us


with program() as iq_blobs:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(2)]
    Q_g = [declare(fixed) for _ in range(2)]
    I_g_st = [declare_stream() for _ in range(2)]
    Q_g_st = [declare_stream() for _ in range(2)]
    I_e = [declare(fixed) for _ in range(2)]
    Q_e = [declare(fixed) for _ in range(2)]
    I_e_st = [declare_stream() for _ in range(2)]
    Q_e_st = [declare_stream() for _ in range(2)]

    with for_(n, 0, n < pts, n + 1):
        # ground iq blobs
        reset_qubit("cooldown", "q1_xy", "rr1", cooldown_time=cooldown_time)
        reset_qubit("cooldown", "q2_xy", "rr2", cooldown_time=cooldown_time)
        # reset_qubit("active", "q1_xy", "resonator", threshold=ge_threshold_q1, max_tries=10, Ig=I_g)
        align()
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=[1, 2], weights="rotated_")

        # excited iq blobs
        align()
        play("x180", "q1_xy")
        # play("x180", "q2_xy")
        align()
        multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=[1, 2], weights="rotated_")

    with stream_processing():
        for i in range(2):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

# open quantum machine
qm = qmm.open_qm(config)

# run job
job = qm.execute(iq_blobs)

# fetch data
results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()

two_state_discriminator(I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, True, True)
plt.suptitle("qubit 1")
two_state_discriminator(I_g_q2, Q_g_q2, I_e_q2, Q_e_q2, True, True)
plt.suptitle("qubit 2")
plt.show()