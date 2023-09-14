from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from macros import qua_declaration, multiplexed_readout
from qualang_tools.analysis import two_state_discriminator
from quam import QuAM
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)

machine.resonators[active_qubits[0]].readout_pulse_amp *= 1.5
machine.resonators[active_qubits[1]].readout_pulse_amp *= 1.5

machine.resonators[active_qubits[0]].readout_pulse_length = 2000
machine.resonators[active_qubits[1]].readout_pulse_length = 2000

config = build_config(machine)

qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

res_if_1 = rr1.f_opt - machine.local_oscillators.readout[0].freq
res_if_2 = rr2.f_opt - machine.local_oscillators.readout[0].freq


###################
# The QUA program #
###################
n_runs = 4000
amplitudes = np.arange(0.5, 1.5, 0.1)
dfs = np.arange(-1e6, 1e6, 0.1e6)
cooldown_time = 5 * max(qb1.T1, qb2.T1)

res_if_1 = rr1.f_opt - machine.local_oscillators.readout[0].freq
res_if_2 = rr2.f_opt - machine.local_oscillators.readout[0].freq

with program() as ro_freq_opt:
    n = declare(int)
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    df = declare(int)

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(*from_array(df, dfs)):
        update_frequency(rr1.name, df + res_if_1)
        update_frequency(rr2.name, df + res_if_2)
        with for_(*from_array(a, amplitudes)):
            with for_(n, 0, n < n_runs, n + 1):
                # ground iq blobs for both qubits
                wait(cooldown_time * u.ns)
                align()
                multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=active_qubits, weights="rotated_", amplitude=a)

                # excited iq blobs for both qubits
                align()
                # Wait for thermalization again in case of measurement induced transitions
                wait(cooldown_time * u.ns)
                play("x180", qb1.name + "_xy")
                play("x180", qb2.name + "_xy")
                align()
                multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=active_qubits, weights="rotated_", amplitude=a)


    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        for i in range(2):
            I_g_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"I_g_q{i}")
            Q_g_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"Q_g_q{i}")
            I_e_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"I_e_q{i}")
            Q_e_st[i].buffer(n_runs).buffer(len(amplitudes)).buffer(len(dfs)).save(f"Q_e_q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(config, ro_freq_opt, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()

else:
    # open quantum machine
    qm = qmm.open_qm(config)
    # run job
    job = qm.execute(ro_freq_opt)
    # fetch data
    results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
    I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()
    # Process the data
    fidelity_vec = [np.zeros((len(amplitudes), len(dfs))), np.zeros((len(amplitudes), len(dfs)))]
    for j in range(len(dfs)):
        for i in range(len(amplitudes)):
            _, _, fidelity_q1, _, _, _, _ = two_state_discriminator(
                I_g_q1[j][i], Q_g_q1[j][i], I_e_q1[j][i], Q_e_q1[j][i], b_print=False, b_plot=False
            )
            _, _, fidelity_q2, _, _, _, _ = two_state_discriminator(
                I_g_q2[j][i], Q_g_q2[j][i], I_e_q2[j][i], Q_e_q2[j][i], b_print=False, b_plot=False
            )
            fidelity_vec[0][i][j] = fidelity_q1
            fidelity_vec[1][i][j] = fidelity_q2

    # Plot the data
    plt.figure()
    plt.suptitle("Readout amplitude optimization")
    plt.subplot(121)
    plt.pcolor((dfs + res_if_1) / u.MHz, amplitudes * rr1.readout_pulse_amp, fidelity_vec[0])
    plt.title(f"{rr1.name}")
    plt.colorbar()
    plt.ylabel("Readout amplitude [V]")
    plt.xlabel("Readout IF [MHz]")
    plt.subplot(122)
    plt.pcolor((dfs + res_if_2) / u.MHz, amplitudes * rr2.readout_pulse_amp, fidelity_vec[1])
    plt.title(f"{rr2.name}")
    plt.colorbar()
    plt.ylabel("Readout amplitude [V]")
    plt.xlabel("Readout IF [MHz]")
    plt.tight_layout()

    rr1.f_opt += dfs[np.where(fidelity_vec[0]==np.amax(fidelity_vec[0]))[1][0]]
    rr1.readout_pulse_amp *= amplitudes[np.where(fidelity_vec[0]==np.amax(fidelity_vec[0]))[0][0]]
    rr2.f_opt += dfs[np.where(fidelity_vec[1]==np.amax(fidelity_vec[1]))[1][0]]
    rr2.readout_pulse_amp *= amplitudes[np.where(fidelity_vec[1]==np.amax(fidelity_vec[1]))[0][0]]
    rr1.readout_fidelity = np.amax(fidelity_vec[0])
    rr2.readout_fidelity = np.amax(fidelity_vec[1])
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # machine._save("current_state.json")
