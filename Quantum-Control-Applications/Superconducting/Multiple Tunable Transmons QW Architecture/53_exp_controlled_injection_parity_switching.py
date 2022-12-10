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
from config import NUMBER_OF_QUBITS_W_CHARGE


##################
# State and QuAM #
##################
experiment = "controlled_qp_parity_switching"
debug = True
simulate = False
qubit_w_charge_list = [0, 1]
qubit_wo_charge_list = [2, 3, 4, 5]
qubit_list = [0, 5]  # you can shuffle the order at which you perform the experiment
injector_list = [0, 1]
digital = [1, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
qp_repetition_rate = 100e-6  # in nanoseconds

config = machine.build_config(digital, qubit_w_charge_list, qubit_wo_charge_list, injector_list, gate_shape)

###################
# The QUA program #
###################
qp_iter = 50e-3/100e-6  # 20 Hz repetition rate divided by 100 us qp measurement rate
n_reps = 60
injector_min = 20 // 4
injector_max = 400000 // 4
d_injector = 50000 // 4
injector_lens = np.arange(injector_min, injector_max+d_injector/2, d_injector)
quarter_precession = []
for q in qubit_list:
    if q in qubit_w_charge_list:
        quarter_precession.append(int((1e9/(4 * machine.qubits[q].df)) // 4))  # in clock cycles
    else:
        quarter_precession.append(int((1e9/(4 * machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].df)) // 4))  # in clock cycles

# QUA program
with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    it = declare(int)
    it_st = declare_stream()
    t = declare(int, value=5)
    t_count = declare(int, value=0)
    rep = declare(int)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        if q in qubit_w_charge_list:
            set_dc_offset(machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(q, "working_point").value)

    with for_(*from_array(t, injector_lens)):
        with for_(rep, 0, rep < n_reps, rep+1):
            play('injector', machine.qp_injectors[0].name, duration=t)
            with for_(it, 0, it < qp_iter, it + 1):
                for i, q in enumerate(qubit_list):
                    if q in qubit_w_charge_list:
                        play("x90", machine.qubits[q].name)
                        wait(quarter_precession[i], machine.qubits[q].name)
                        play("y90", machine.qubits[q].name)
                    else:
                        play("x90", machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                        wait(quarter_precession[i], machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                        play("y90", machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                align()
                for i, q in enumerate(qubit_list):
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("rotated_cos", I[i], "out1"),
                        demod.full("rotated_sin", Q[i], "out1"),
                    )
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                wait_cooldown_time(qp_repetition_rate, simulate)  # repetition rate of 100 microseconds
        save(t_count, it_st)
        assign(t_count, t_count+1)

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(qp_iter*n_reps).save(f"I{q}")
            Q_st[i].buffer(qp_iter*n_reps).save(f"Q{q}")
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
    qmm.clear_all_job_results()
    job = qm.execute(T1)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    it_state = 0
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

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
    while my_results.is_processing() and it_state < len(injector_lens) - 1:
        # Fetch results
        data = my_results.fetch_all()
        for i, q in enumerate(qubit_list):
            qubit_data[i]["I"] = data[0+i*2]
            qubit_data[i]["Q"] = data[1+i*2]
        it_state = data[-1]
        # Progress bar
        progress_counter(it_state, len(injector_lens), start_time=my_results.start_time)
        # live plot
        if debug:
            for i, q in enumerate(qubit_list):
                plt.figure(i)
                plt.subplot(211)
                plt.cla()
                pnts = len(qubit_data[i]["I"])
                pnts_array = np.arange(0, pnts, 1)
                plt.plot(pnts_array * qp_repetition_rate, qubit_data[i]["I"])
                plt.xlabel('Time [s]')
                plt.ylabel('I [a.u.]')
                plt.title('Qubit' + str(q))
                plt.subplot(212)
                plt.cla()
                pnts = len(qubit_data[i]["Q"])
                pnts_array = np.arange(0, pnts, 1)
                plt.plot(pnts_array * qp_repetition_rate, qubit_data[i]["Q"])
                plt.xlabel('Time [s]')
                plt.ylabel('Q [a.u.]')
                plt.pause(1)
                plt.tight_layout()

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
