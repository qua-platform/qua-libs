from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate import LoopbackInterface
from qm import SimulationConfig, generate_qua_script
from qm.qua import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from quam import QuAM
from configuration import *
from macros import qua_declaration_w_state, multiplexed_readout_w_state, active_reset
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
amps = np.arange(0.1, 1.7, 0.01)
n_avg = 1_000
depletion_time = 100_000
number_of_qubits = 9
err_amp = 1

qb_list = [i for i in range(number_of_qubits)]
qb_if_list = [machine.qubits[i].f_01 - machine.qubits[i].lo for i in range(number_of_qubits)]
threshold_list = [machine.resonators[i].ge_threshold for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"I{i}")
    fetching_list.append(f"Q{i}")
    fetching_list.append(f"state{i}")
    # machine.qubits[i].pi_length = 40 # define qubit length
    # machine.qubits[i].pi_amp = 40 # define qubit length

with program() as power_rabi:
    I, I_st, Q, Q_st, state, state_st, n, n_st = qua_declaration_w_state(nb_of_qubits=number_of_qubits)
    a = declare(fixed)
    counter = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):

            # active-reset
            with for_(counter, 0, counter < 3, counter+1):
                multiplexed_readout_w_state(I, None, Q, None, state, None, th=threshold_list, resonators=qb_list)
                align()  # help auto-element-thread
                wait(4)
                align()  # help auto-element-thread
                for i in qb_list:
                    play('x180', machine.qubits[i].name, condition=state[i])

            align()  # help auto-element-thread

            # Loop for error amplification (perform many qubit pulses)
            for i in range(err_amp):
                for i in qb_list:
                    play("x180" * amp(a), machine.qubits[i].name)

            align()

            multiplexed_readout_w_state(I, I_st, Q, Q_st, state, state_st, th=threshold_list, resonators=qb_list)

    with stream_processing():
        n_st.save('n')
        for i in range(number_of_qubits):
            I_st[i].buffer(len(amps)).average().save(f"I{i}")
            Q_st[i].buffer(len(amps)).average().save(f"Q{i}")
            state_st[i].boolean_to_int().buffer(len(amps)).average().save(f"state{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

simulate = False
print(generate_qua_script(power_rabi))

if simulate:
    # simulate the test_config QUA program
    job = qmm.simulate(
        config,
        power_rabi,
        SimulationConfig(
            11000, simulation_interface=LoopbackInterface([("con1", 1, "con1", 1), ("con1", 2, "con1", 2)], latency=250)
        ),
    )
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(power_rabi, flags=['auto-element-thread'])
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
            plt.xlabel("pre-factor")
            # plt.plot(amps, results_fetched[3*i+1])
            # plt.plot(amps, results_fetched[3*i+2])
            plt.plot(amps, results_fetched[3*i+3])

        plt.tight_layout()
        plt.pause(0.1)
    plt.show()

# machine.qubits[0].pi_amp =
# machine.qubits[1].pi_amp =
# machine._save("quam_bootstrap_state.json")