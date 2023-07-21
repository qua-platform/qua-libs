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
from qualang_tools.addons.variables import assign_variables_to_element

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
dfs = np.arange(-0.5e6, 0.5e6, 0.02e6)
n_avg = 1_000
cooldown_time  = 100_000
number_of_qubits = 9

qb_list = [i for i in range(number_of_qubits)]
res_if_list = [machine.resonators[i].f_readout - machine.resonators[i].lo for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"D{i}")

with program() as iq_blobs:
    n = declare(int)
    n_st = declare_stream()
    I_g = [declare(fixed) for _ in range(number_of_qubits)]
    Q_g = [declare(fixed) for _ in range(number_of_qubits)]
    I_e = [declare(fixed) for _ in range(number_of_qubits)]
    Q_e = [declare(fixed) for _ in range(number_of_qubits)]
    DI = [declare(fixed) for _ in range(number_of_qubits)]
    DQ = [declare(fixed) for _ in range(number_of_qubits)]
    D = [declare(fixed) for _ in range(number_of_qubits)]
    df = declare(int)
    D_st = [declare_stream() for _ in range(number_of_qubits)]
    for i in range(number_of_qubits):
        assign_variables_to_element(f"rr{i}", I_g[i], Q_g[i])
        assign_variables_to_element(f"rr{i}", I_e[i], Q_e[i])


    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            for i in qb_list:
                update_frequency(machine.resonators[i].name, df + res_if_list[i])

            save(n, n_st)
            # wait for the resonators to relax
            wait(cooldown_time  * u.ns)
            multiplexed_readout(I_g, None, Q_g, None, resonators=qb_list, weights='rotated_')

            align()

            wait(cooldown_time  * u.ns)
            for i in qb_list:
                play("x180", machine.qubits[i].name)
            align()
            multiplexed_readout(I_e, None, Q_e, None, resonators=qb_list, weights='rotated_')

            for i in qb_list:
                assign(DI[i], (I_e[i] - I_g[i]) * 100)
                assign(DQ[i], (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI[i]*DI[i] + DQ[i]*DQ[i])
                save(D[i], D_st[i])

    with stream_processing():
        n_st.save('n')
        for i in range(number_of_qubits):
            D_st[i].buffer(len(dfs)).average().save(f"D{i}")

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
        # Plot
        for i in range(number_of_qubits):
            plt.subplot(3, 3, i + 1)
            plt.cla()
            plt.title(f"qb{i}")
            plt.xlabel("Freq (MHz)")
            plt.plot(dfs, results_fetched[i+1])
        plt.tight_layout()
        plt.pause(0.5)

    for i in qb_list:
        print(f"{i}: Shift readout frequency by {dfs[np.argmax(results_fetched[i+1])]} Hz")
    plt.show()