from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import LoopbackInterface
from qm import SimulationConfig, generate_qua_script
from qm.qua import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from quam import QuAM
from configuration import *
from macros import qua_declaration, multiplexed_readout
from qm.octave import *
from scipy import signal
from qualang_tools.analysis import two_state_discriminator

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
n_avg = 10_000
cooldown_time  = 100_000
number_of_qubits = 9

qb_list = [i for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"I_g_{i}")
    fetching_list.append(f"Q_g_{i}")
    fetching_list.append(f"I_e_{i}")
    fetching_list.append(f"Q_e_{i}")

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, n_st = qua_declaration(nb_of_qubits=number_of_qubits)
    I_e, I_e_st, Q_e, Q_e_st, nn, nn_st = qua_declaration(nb_of_qubits=number_of_qubits)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        # wait for the resonators to relax
        wait(cooldown_time  * u.ns)
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=qb_list, weights='rotated_')

        align()

        wait(cooldown_time  * u.ns)
        for i in qb_list:
            play("x180", machine.qubits[i].name)
        align()
        multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=qb_list, weights='rotated_')

    with stream_processing():
        n_st.save('n')
        for i in range(number_of_qubits):
            I_g_st[i].save_all(f"I_g_{i}")
            Q_g_st[i].save_all(f"Q_g_{i}")
            I_e_st[i].save_all(f"I_e_{i}")
            Q_e_st[i].save_all(f"Q_e_{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
# print(generate_qua_script(iq_blobs))

if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        iq_blobs,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(iq_blobs, flags=['auto-element-thread'])
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ['n']+fetching_list, mode='live')
    while results.is_processing():
        # Fetch results
        results_fetched = results.fetch_all()
        # Progress bar
        progress_counter(results_fetched[0], n_avg, start_time=results.start_time)
    for i in qb_list:
        two_state_discriminator(results_fetched[4*i+1], results_fetched[4*i+2], results_fetched[4*i+3], results_fetched[4*i+4], True, True)
        plt.suptitle(f'qb{i}')
    plt.show()