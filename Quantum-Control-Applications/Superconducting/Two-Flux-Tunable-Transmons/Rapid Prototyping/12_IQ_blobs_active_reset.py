from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import qua_declaration, multiplexed_readout, reset_qubit
from quam import QuAM
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
pts = 20000
cooldown_time = 1 * u.us

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)

    with for_(n, 0, n < pts, n + 1):
        # ground iq blobs
        reset_qubit("cooldown", "q0_xy", "rr0", cooldown_time=cooldown_time)
        reset_qubit("cooldown", "q1_xy", "rr1", cooldown_time=cooldown_time)
        # reset_qubit("active", "q0_xy", "resonator", threshold=machine.qubits[0].ge_threshold, max_tries=10, Ig=I_g)
        align()
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=[0, 1], weights="rotated_")

        # excited iq blobs
        align()
        play("x180", "q0_xy")
        # play("x180", "q1_xy")
        align()
        multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=[0, 1], weights="rotated_")

    with stream_processing():
        for i in range(2):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, iq_blobs, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
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
