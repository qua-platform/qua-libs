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


##################
# State and QuAM #
##################
experiment = "readout_freq_optimization"
debug = True
simulate = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 1000

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

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * int(machine.qubits[q].t1 * 1e9) // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "working_point").value)

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            with for_(*from_array(f, freq[q])):
                update_frequency(machine.readout_resonators[q].name, f)
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_g[q]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_g[q]),
                )
                wait(cooldown_time, machine.readout_resonators[q].name)
                save(I_g[q], I_g_st[q])
                save(Q_g[q], Q_g_st[q])

                align()  # global align

                play("x180", machine.qubits[q].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_e[q]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_e[q]),
                )
                wait(cooldown_time, machine.readout_resonators[q].name)
                save(I_e[q], I_e_st[q])
                save(Q_e[q], Q_e_st[q])

            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            n_st[q].save(f"iteration{q}")
            # mean values
            I_g_st[q].buffer(len(freq[q])).average().save(f"I_g_avg{q}")
            Q_g_st[q].buffer(len(freq[q])).average().save(f"Q_g_avg{q}")
            I_e_st[q].buffer(len(freq[q])).average().save(f"I_e_avg{q}")
            Q_e_st[q].buffer(len(freq[q])).average().save(f"Q_e_avg{q}")
            # variances
            (
                ((I_g_st[q].buffer(len(freq[q])) * I_g_st[q].buffer(len(freq[q]))).average())
                - (I_g_st[q].buffer(len(freq[q])).average() * I_g_st[q].buffer(len(freq[q])).average())
            ).save(f"I_g_var{q}")
            (
                ((Q_g_st[q].buffer(len(freq[q])) * Q_g_st[q].buffer(len(freq[q]))).average())
                - (Q_g_st[q].buffer(len(freq[q])).average() * Q_g_st[q].buffer(len(freq[q])).average())
            ).save(f"Q_g_var{q}")
            (
                ((I_e_st[q].buffer(len(freq[q])) * I_e_st[q].buffer(len(freq[q]))).average())
                - (I_e_st[q].buffer(len(freq[q])).average() * I_e_st[q].buffer(len(freq[q])).average())
            ).save(f"I_e_var{q}")
            (
                ((Q_e_st[q].buffer(len(freq[q])) * Q_e_st[q].buffer(len(freq[q]))).average())
                - (Q_e_st[q].buffer(len(freq[q])).average() * Q_e_st[q].buffer(len(freq[q])).average())
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
    for q in range(len(qubit_list)):
        # Live plotting
        print("Qubit " + str(q))
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        qubit_data[q]["iteration"] = 0
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
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I_g_avg"] = data[0]
            qubit_data[q]["Q_g_avg"] = data[1]
            qubit_data[q]["I_e_avg"] = data[2]
            qubit_data[q]["Q_e_avg"] = data[3]
            qubit_data[q]["I_g_var"] = data[4]
            qubit_data[q]["Q_g_var"] = data[5]
            qubit_data[q]["I_e_var"] = data[6]
            qubit_data[q]["Q_e_var"] = data[7]
            qubit_data[q]["iteration"] = data[8]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # Derive SNR
            Z = (qubit_data[q]["I_e_avg"] - qubit_data[q]["I_g_avg"]) + 1j * (
                qubit_data[q]["Q_e_avg"] - qubit_data[q]["Q_g_avg"]
            )
            var = (
                qubit_data[q]["I_g_var"]
                + qubit_data[q]["Q_g_var"]
                + qubit_data[q]["I_e_var"]
                + qubit_data[q]["Q_e_var"]
            ) / 4
            SNR = ((np.abs(Z)) ** 2) / (2 * var)
            if debug:
                plt.subplot(311)
                plt.cla()
                plt.plot(freq[q] + machine.readout_lines[q].lo_freq, SNR, ".-")
                plt.title(f"{experiment} qubit {q}")
                plt.ylabel("SNR")
                plt.subplot(312)
                plt.cla()
                plt.plot(freq[q] + machine.readout_lines[q].lo_freq, qubit_data[q]["I_g_avg"])
                plt.plot(freq[q] + machine.readout_lines[q].lo_freq, qubit_data[q]["I_e_avg"])
                plt.legend(("ground", "excited"))
                plt.ylabel("I [a.u.]")
                plt.subplot(313)
                plt.cla()
                plt.plot(freq[q] + machine.readout_lines[q].lo_freq, qubit_data[q]["Q_g_avg"])
                plt.plot(freq[q] + machine.readout_lines[q].lo_freq, qubit_data[q]["Q_e_avg"])
                plt.legend(("ground", "excited"))
                plt.ylabel("Q [a.u.]")
                plt.xlabel("Readout frequency [Hz]")
                plt.tight_layout()
                plt.pause(0.1)
        # Find the readout frequency that maximizes the SNR
        f_opt = freq[q][np.argmax(SNR)]
        SNR_opt = SNR[np.argmax(SNR)]
        print(
            f"Previous optimal readout frequency: {machine.readout_resonators[q].f_opt*1e-9:.6f} GHz with SNR = {SNR[len(freq[q])//2+1]:.2f}"
        )
        machine.readout_resonators[q].f_opt = (
            f_opt + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq
        )
        print(
            f"New optimal readout frequency: {machine.readout_resonators[q].f_opt*1e-9:.6f} GHz with SNR = {SNR_opt:.2f}"
        )
        print(f"New resonance IF frequency: {machine.get_readout_IF(q) * 1e-6:.3f} MHz")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
