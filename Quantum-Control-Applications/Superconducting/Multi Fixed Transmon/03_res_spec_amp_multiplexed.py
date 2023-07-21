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

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_state.json", flat_data=False)
config = build_config(machine)

###################
# The QUA program #
###################
dfs = np.arange(-2e6, +2e6, 0.1e6)
n_avg = 10
depletion_time = 100_000
number_of_qubits = 9
amps = np.arange(0.2, 1.99, 0.10)

resonators_list = [i for i in range(number_of_qubits)]
res_if_list = [machine.resonators[i].f_readout - machine.resonators[i].lo for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"I{i}")
    fetching_list.append(f"Q{i}")

with program() as multi_res_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=number_of_qubits)
    df = declare(int)
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # wait for the resonators to relax

            for i in range(number_of_qubits):
                update_frequency(machine.resonators[i].name, df + res_if_list[i])

            with for_(*from_array(a, amps)):

                wait(depletion_time * u.ns)
                multiplexed_readout(I, I_st, Q, Q_st, resonators=resonators_list, amplitude=a)

    with stream_processing():
        n_st.save('n')
        for i in range(number_of_qubits):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).save(f"I{i}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).save(f"Q{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False

if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        multi_res_spec,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec, flags=['auto-element-thread'])
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ['n']+fetching_list, mode='live')
    while results.is_processing():
        # Fetch results
        results_fetched = results.fetch_all()
        # Progress bar
        progress_counter(results_fetched[0], n_avg, start_time=results.start_time)
        # Data analysis
        s_list = [u.demod2volts(results_fetched[2*i+1] + 1j * results_fetched[2*i+2], machine.resonators[0].readout_pulse_length) for i in range(number_of_qubits)]
        A_list = [np.abs(s) for s in s_list]

        for A in A_list:
            row_sums = A.sum(axis=0)
            A /= row_sums[np.newaxis, :]

        # Plot
        for i in range(number_of_qubits):
            plt.subplot(3, 3, i + 1)
            plt.cla()
            plt.title(f"rr{i+1} - f_cent: {machine.resonators[i].f_readout / u.MHz}")
            plt.xlabel("pre-factor")
            plt.ylabel("det (MHz)")
            plt.pcolor(amps, dfs / u.MHz, A_list[i])

        plt.tight_layout()
        plt.pause(0.1)