"""
Ramsey like sequence to detect parity
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from macros import *


##################
# State and QuAM #
##################
experiment = "qp_parity_switching_multiplexed"
debug = True
simulate = False
charge_lines = [0, 1]
injector_list = [0, 1]
digital = [1, 2, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment

config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

###################
# The QUA program #
###################
n_avg = 1e4
quarter_precession = []
for q in qubit_list:
    quarter_precession.append(int(machine.qubits[q].idle_time_parity * 1e9 // 4))  # in clock cycles

# QUA program
with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    it = declare(int)
    it_st = declare_stream()

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(
                    machine.qubits[q].name + "_charge",
                    "single",
                    machine.get_charge_bias_point(j, "working_point").value,
                )

    with for_(it, 0, it < n_avg, it + 1):
        for i, q in enumerate(qubit_list):
            play("x90", machine.qubits[q].name)
            wait(quarter_precession[i], machine.qubits[q].name)
            play("y90", machine.qubits[q].name)
            # align() -- no need bc qb-rr share same core and messes up multiplexing
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                demod.full("rotated_cos", I[i], "out1"),
                demod.full("rotated_sin", Q[i], "out1"),
            )
            save(I[i], I_st[i])
            save(Q[i], Q_st[i])

        save(it, it_st)
        wait_cooldown_time(10e-3, simulate)  # repetition rate

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].save_all(f"I{q}")
            Q_st[i].save_all(f"Q{q}")
        it_st.save(f"iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, T1, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(T1)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    it_state = 0
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

    sbplot = [231, 232, 233, 234, 235, 236]

    a = []
    # Live plotting
    tell = "Qubit"
    if debug:
        fig = plt.figure()
        interrupt_on_close(fig, job)
        figures.append(fig)
    for i, q in enumerate(qubit_list):
        tell = tell + " " + str(q) + ","
        # Get results from QUA program
        a.append(f"I{q}")
        a.append(f"Q{q}")
    a.append("iteration")
    print(tell)
    my_results = fetching_tool(job, a, mode="live")
    while my_results.is_processing() and it_state < n_avg - 1:
        # Fetch results
        data = my_results.fetch_all()
        for i, q in enumerate(qubit_list):
            qubit_data[i]["I"] = data[0 + i * 2]
            qubit_data[i]["Q"] = data[1 + i * 2]
        it_state = data[-1]
        # Progress bar
        progress_counter(it_state, n_avg, start_time=my_results.start_time)
        # live plot
        if debug:
            for i, q in enumerate(qubit_list):
                # plt.figure(i)
                plt.subplot(sbplot[i])
                plt.cla()
                pnts = len(qubit_data[i]["I"])
                pnts_array = np.arange(0, pnts, 1)
                plt.plot(pnts_array * 10e-3, qubit_data[i]["I"])
                plt.xlabel("Time [s]")
                plt.ylabel("I [a.u.]")
                plt.title("Qubit" + str(q))
                plt.pause(1)
                plt.tight_layout()

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
