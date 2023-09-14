
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import qua_declaration, multiplexed_readout, reset_qubit
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2
###################
# The QUA program #
###################
pts = 20000
cooldown_time = 5 * max(qb1.T1, qb2.T1)

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < pts, n + 1):
        # ground iq blobs
        # reset_qubit("cooldown", qb1.name + "_xy", rr1.name, cooldown_time=cooldown_time)
        reset_qubit("cooldown", qb2.name + "_xy", rr2.name, cooldown_time=cooldown_time)
        # wait(cooldown_time * u.ns)
        reset_qubit("active", qb1.name + "_xy", rr1.name, threshold=qb1.ge_threshold, max_tries=1, Ig=I_g[0])
        align()
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=active_qubits, weights="rotated_")

        # excited iq blobs
        align()
        # wait(cooldown_time * u.ns)
        reset_qubit("active", qb1.name + "_xy", rr1.name, threshold=qb1.ge_threshold, max_tries=10, Ig=I_g[0])
        play("x180", qb1.name + "_xy")
        play("x180", qb2.name + "_xy")
        align()
        multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=active_qubits, weights="rotated_")

    with stream_processing():
        for i in range(2):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

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
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()


