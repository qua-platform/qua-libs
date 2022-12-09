"""
Use IQ blobs measurements to find optimal readout frequency
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *
from config import NUMBER_OF_QUBITS_W_CHARGE

##################
# State and QuAM #
##################
experiment = "readout_freq_optimization"
debug = True
simulate = False
qubit_w_charge_list = [0, 1]
qubit_wo_charge_list = [2, 3, 4, 5]
qubit_list = [0, 5]  # you can shuffle the order at which you perform the experiment
injector_list = [0, 1]
digital = [1, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_w_charge_list, qubit_wo_charge_list, injector_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e2

# Frequency scan
span = 1e6
df = 0.01e6
freq = [np.arange(machine.get_readout_IF(i) - span, machine.get_readout_IF(i) + span + df / 2, df) for i in qubit_list]

with program() as readout_opt:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    I_g = [declare(fixed) for _ in range(len(qubit_list))]
    Q_g = [declare(fixed) for _ in range(len(qubit_list))]
    I_g_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_g_st = [declare_stream() for _ in range(len(qubit_list))]
    I_e = [declare(fixed) for _ in range(len(qubit_list))]
    Q_e = [declare(fixed) for _ in range(len(qubit_list))]
    I_e_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_e_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)

    for i, q in enumerate(qubit_list):
        if q in qubit_w_charge_list:
            set_dc_offset(
                machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(q, "working_point").value
            )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                update_frequency(machine.readout_resonators[q].name, f)
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    demod.full("cos", I_g[i], "out1"),
                    demod.full("sin", Q_g[i], "out1"),
                )
                wait_cooldown_time_fivet1(q, machine, simulate, qubit_w_charge_list)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

                align()  # global align

                if q in qubit_w_charge_list:
                    play("x180", machine.qubits[q].name)
                else:
                    play("x180", machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    demod.full("cos", I_e[i], "out1"),
                    demod.full("sin", Q_e[i], "out1"),
                )
                wait_cooldown_time_fivet1(q, machine, simulate, qubit_w_charge_list)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            n_st[i].save(f"iteration{q}")
            # mean values
            I_g_st[i].buffer(len(freq[i])).average().save(f"I_g_avg{q}")
            Q_g_st[i].buffer(len(freq[i])).average().save(f"Q_g_avg{q}")
            I_e_st[i].buffer(len(freq[i])).average().save(f"I_e_avg{q}")
            Q_e_st[i].buffer(len(freq[i])).average().save(f"Q_e_avg{q}")
            # variances
            (
                ((I_g_st[i].buffer(len(freq[i])) * I_g_st[i].buffer(len(freq[i]))).average())
                - (I_g_st[i].buffer(len(freq[i])).average() * I_g_st[i].buffer(len(freq[i])).average())
            ).save(f"I_g_var{q}")
            (
                ((Q_g_st[i].buffer(len(freq[i])) * Q_g_st[i].buffer(len(freq[i]))).average())
                - (Q_g_st[i].buffer(len(freq[i])).average() * Q_g_st[i].buffer(len(freq[i])).average())
            ).save(f"Q_g_var{q}")
            (
                ((I_e_st[i].buffer(len(freq[i])) * I_e_st[i].buffer(len(freq[i]))).average())
                - (I_e_st[i].buffer(len(freq[i])).average() * I_e_st[i].buffer(len(freq[i])).average())
            ).save(f"I_e_var{q}")
            (
                ((Q_e_st[i].buffer(len(freq[i])) * Q_e_st[i].buffer(len(freq[i]))).average())
                - (Q_e_st[i].buffer(len(freq[i])).average() * Q_e_st[i].buffer(len(freq[i])).average())
            ).save(f"Q_e_var{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, readout_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(readout_opt)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i,q in enumerate(qubit_list):
        # Live plotting
        print("Qubit " + str(q))
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        qubit_data[i]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(
            job,
            [
                f"I_g_avg{q}",
                f"Q_g_avg{q}",
                f"I_e_avg{q}",
                f"Q_e_avg{q}",
                f"I_g_var{q}",
                f"Q_g_var{q}",
                f"I_e_var{q}",
                f"Q_e_var{q}",
                f"iteration{q}",
            ],
            mode="live",
        )
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["I_g_avg"] = data[0]
            qubit_data[i]["Q_g_avg"] = data[1]
            qubit_data[i]["I_e_avg"] = data[2]
            qubit_data[i]["Q_e_avg"] = data[3]
            qubit_data[i]["I_g_var"] = data[4]
            qubit_data[i]["Q_g_var"] = data[5]
            qubit_data[i]["I_e_var"] = data[6]
            qubit_data[i]["Q_e_var"] = data[7]
            qubit_data[i]["iteration"] = data[8]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
            # Derive SNR
            Z = (qubit_data[i]["I_e_avg"] - qubit_data[i]["I_g_avg"]) + 1j * (
                qubit_data[i]["Q_e_avg"] - qubit_data[i]["Q_g_avg"]
            )
            var = (
                qubit_data[i]["I_g_var"]
                + qubit_data[i]["Q_g_var"]
                + qubit_data[i]["I_e_var"]
                + qubit_data[i]["Q_e_var"]
            ) / 4
            SNR = ((np.abs(Z)) ** 2) / (2 * var)
            if debug:
                plt.subplot(311)
                plt.cla()
                plt.plot(freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq, SNR, ".-")
                plt.title(f"{experiment} qubit {q}")
                plt.ylabel("SNR")
                plt.subplot(312)
                plt.cla()
                plt.plot(freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq, qubit_data[i]["I_g_avg"])
                plt.plot(freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq, qubit_data[i]["I_e_avg"])
                plt.legend(("ground", "excited"))
                plt.ylabel("I [a.u.]")
                plt.subplot(313)
                plt.cla()
                plt.plot(freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq, qubit_data[i]["Q_g_avg"])
                plt.plot(freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq, qubit_data[i]["Q_e_avg"])
                plt.legend(("ground", "excited"))
                plt.ylabel("Q [a.u.]")
                plt.xlabel("Readout frequency [Hz]")
                plt.tight_layout()
                plt.pause(0.1)
        # Find the readout frequency that maximizes the SNR
        f_opt = freq[i][np.argmax(SNR)]
        SNR_opt = SNR[np.argmax(SNR)]
        print(
            f"Previous optimal readout frequency: {machine.readout_resonators[q].f_opt*1e-9:.6f} GHz with SNR = {SNR[len(freq[i])//2+1]:.2f}"
        )
        machine.readout_resonators[q].f_opt = (
            f_opt + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq
        )
        print(f"New optimal readout frequency: {machine.readout_resonators[q].f_opt*1e-9:.6f} GHz with SNR = {SNR_opt:.2f}")
        print(f"New resonance IF frequency: {machine.get_readout_IF(q) * 1e-6:.3f} MHz")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
