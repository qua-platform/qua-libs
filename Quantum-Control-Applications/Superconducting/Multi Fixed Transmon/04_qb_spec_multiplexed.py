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

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
dfs = np.arange(-20e6, +20e6, 0.1e6)
n_avg = 1_000
depletion_time = 100_000
number_of_qubits = 9
t = 14 * u.us  # Qubit pulse length

qb_list = [i for i in range(number_of_qubits)]
qb_if_list = [machine.qubits[i].f_01 - machine.qubits[i].lo for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"I{i}")
    fetching_list.append(f"Q{i}")

with program() as multi_qb_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=number_of_qubits)
    df = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # wait for the resonators to relax
            wait(depletion_time * u.ns)

            for i in qb_list:
                update_frequency(machine.qubits[i].name, df + qb_if_list[i])
                play("cw" * amp(1), machine.qubits[i].name, duration= t * u.ns)
                align(machine.resonators[i].name, machine.qubits[i].name)

            multiplexed_readout(I, I_st, Q, Q_st, resonators=qb_list, amplitude=1.0)

    with stream_processing():
        n_st.save('n')
        for i in range(number_of_qubits):
            I_st[i].buffer(len(dfs)).average().save(f"I{i}")
            Q_st[i].buffer(len(dfs)).average().save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
# print(generate_qua_script(multi_qb_spec))

if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        multi_qb_spec,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_qb_spec, flags=['auto-element-thread'])
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ['n']+fetching_list, mode='live')
    while results.is_processing():
        # Fetch results
        results_fetched = results.fetch_all()
        # Progress bar
        progress_counter(results_fetched[0], n_avg, start_time=results.start_time)
        # Data analysis
        s_list = [u.demod2volts(results_fetched[2*i+1] + 1j * results_fetched[2*i+2], machine.resonators[0].readout_pulse_length) for i in range(number_of_qubits)]
        # Plot
        for i in range(number_of_qubits):
            plt.subplot(3, 3, i + 1)
            plt.cla()
            plt.title(f"qb{i} - f_cent: {machine.qubits[i].f_01 / u.MHz}")
            plt.xlabel("det (MHz)")
            plt.plot(dfs / u.MHz, np.abs(s_list[i]))
            # plt.plot(dfs / u.MHz, signal.detrend(np.unwrap(np.angle(s_list[i]))))

        plt.tight_layout()
        plt.pause(0.1)
    plt.show()

# machine.qubits[0].xy.f_01 =
# machine.qubits[1].xy.f_01 =
# machine._save("quam_state.json")
