#%%
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import qua_declaration, multiplexed_readout
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].qubit_name + "_z"
q2_z = machine.qubits[active_qubits[1]].qubit_name + "_z"
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
        # ground iq blobs for both qubits
        wait(cooldown_time * u.ns)
        align()
        # play("x180", qb2.qubit_name + "_xy")
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=active_qubits, weights="rotated_")
        align()
        wait(cooldown_time * u.ns)
        # excited iq blobs for both qubits
        play("x180", qb1.qubit_name + "_xy")
        play("x180", qb2.qubit_name + "_xy")
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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

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

    angle1, threshold1, fidelity1, _, _, _, _ = two_state_discriminator(I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, True, True)
    plt.suptitle(f"{qb1.qubit_name}")
    angle2, threshold2, fidelity2, _, _, _, _ = two_state_discriminator(I_g_q2, Q_g_q2, I_e_q2, Q_e_q2, True, True)
    plt.suptitle(f"{qb2.qubit_name}")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    rr1.rotation_angle = angle1
    rr1.readout_fidelity = fidelity1
    qb1.ge_threshold = threshold1
    rr2.rotation_angle = angle2
    rr2.readout_fidelity = fidelity2
    qb2.ge_threshold = threshold2
    # machine._save("current_state.json")

# %%
