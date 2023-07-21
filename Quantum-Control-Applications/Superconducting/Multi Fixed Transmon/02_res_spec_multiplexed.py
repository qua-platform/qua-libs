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
n_avg = 1_000
depletion_time = 100_000
number_of_qubits = 9

resonators_list = [i for i in range(number_of_qubits)]
res_if_list = [machine.resonators[i].f_readout - machine.resonators[i].lo for i in range(number_of_qubits)]
fetching_list = []
for i in range(number_of_qubits):
    fetching_list.append(f"I{i}")
    fetching_list.append(f"Q{i}")

with program() as multi_res_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=number_of_qubits)
    df = declare(int)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # wait for the resonators to relax
            wait(depletion_time * u.ns)

            for i in resonators_list:
                update_frequency(machine.resonators[i].name, df + res_if_list[i])

            multiplexed_readout(I, I_st, Q, Q_st, resonators=resonators_list, amplitude=1.0)

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
    results = fetching_tool(job, ["n"], mode='live')
    while results.is_processing():
        n = results.fetch_all()[0]
        progress_counter(n, n_avg, start_time=results.start_time)
    results = fetching_tool(job, fetching_list)
    # Fetch results
    results_fetched = results.fetch_all()
    # Data analysis
    s_list = [u.demod2volts(results_fetched[2*i] + 1j * results_fetched[2*i+1], machine.resonators[0].readout_pulse_length) for i in range(number_of_qubits)]
    # Plot
    fig, ax = plt.subplots(3, 3, figsize=(12, 9))
    for i in range(number_of_qubits):
        row = i // 3
        col = i % 3
        ax[row, col].plot(machine.resonators[i].f_readout / u.MHz + dfs / u.MHz, np.abs(s_list[i]))
        ax[row, col].set_title(f"rr {i}")
        ax[row, col].set_xlabel("Freq (MHz)")
    plt.tight_layout()
    plt.show()

# machine._save("quam_bootstrap_state.json")

# try:
#     from qualang_tools.plot.fitting import Fit
# 
#     fit = Fit()
#     plt.figure()
#     plt.subplot(121)
#     res_1 = fit.reflection_resonator_spectroscopy((machine.resonators[0].f_res + dfs) / u.MHz, np.abs(s0), plot=True)
#     plt.xlabel("rr1 IF (MHz)")
#     machine.resonators[0].f_res = res_1["f"] * u.MHz
#     machine.resonators[0].f_opt = machine.resonators[0].f_readout
#     plt.subplot(122)
#     res_2 = fit.reflection_resonator_spectroscopy((machine.resonators[1].f_res + dfs) / u.MHz, np.abs(s1), plot=True)
#     plt.xlabel("rr21 IF (MHz)")
#     machine.resonators[1].f_res = res_2["f"] * u.MHz
#     machine.resonators[1].f_opt = machine.resonators[1].f_readout
# except (Exception,):
#     pass
# %%
