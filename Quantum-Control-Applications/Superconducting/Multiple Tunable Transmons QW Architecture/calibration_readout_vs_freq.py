"""
readout_freq_opt.py: uses IQ blobs measurements to find optimal readout frequency
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime


##################
# State and QuAM #
##################
experiment = "readout_freq_opt"
debug = True
simulate = False
qubit_list = [0, 1]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

cooldown_time = 5 * u.us // 4

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

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_qubits(True, qubit_list, i)
        set_dc_offset(machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                update_frequency(machine.readout_resonators[i].name, f)
                measure(
                    "readout",
                    machine.readout_resonators[i].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_g[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_g[i]),
                )
                wait(cooldown_time, machine.readout_resonators[i].name)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])

                align()  # global align

                play("x180", machine.qubits[i].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[i].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_e[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_e[i]),
                )
                wait(cooldown_time, machine.readout_resonators[i].name)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])

            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            n_st[i].save(f"iteration{i}")
            # mean values
            I_g_st[i].buffer(len(freq[i])).average().save(f"I_g_avg{i}")
            Q_g_st[i].buffer(len(freq[i])).average().save(f"Q_g_avg{i}")
            I_e_st[i].buffer(len(freq[i])).average().save(f"I_e_avg{i}")
            Q_e_st[i].buffer(len(freq[i])).average().save(f"Q_e_avg{i}")
            # variances
            (
                ((I_g_st[i].buffer(len(freq[i])) * I_g_st[i].buffer(len(freq[i]))).average())
                - (I_g_st[i].buffer(len(freq[i])).average() * I_g_st[i].buffer(len(freq[i])).average())
            ).save(f"I_g_var{i}")
            (
                ((Q_g_st[i].buffer(len(freq[i])) * Q_g_st[i].buffer(len(freq[i]))).average())
                - (Q_g_st[i].buffer(len(freq[i])).average() * Q_g_st[i].buffer(len(freq[i])).average())
            ).save(f"Q_g_var{i}")
            (
                ((I_e_st[i].buffer(len(freq[i])) * I_e_st[i].buffer(len(freq[i]))).average())
                - (I_e_st[i].buffer(len(freq[i])).average() * I_e_st[i].buffer(len(freq[i])).average())
            ).save(f"I_e_var{i}")
            (
                ((Q_e_st[i].buffer(len(freq[i])) * Q_e_st[i].buffer(len(freq[i]))).average())
                - (Q_e_st[i].buffer(len(freq[i])).average() * Q_e_st[i].buffer(len(freq[i])).average())
            ).save(f"Q_e_var{i}")

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

    for q in range(len(qubit_list)):
        # Live plotting
        print("Qubit " + str(q))
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(
            job,
            [
                f"I_g_avg{i}",
                f"Q_g_avg{i}",
                f"I_e_avg{i}",
                f"Q_e_avg{i}",
                f"I_g_var{i}",
                f"Q_g_var{i}",
                f"I_e_var{i}",
                f"Q_e_var{i}",
                f"iteration{i}",
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
                qubit_data[q]["Q_e_avg"] - qubit_data[q]["I_e_avg"]
            )
            var = (
                qubit_data[q]["I_g_var"]
                + qubit_data[q]["Q_g_var"]
                + qubit_data[q]["I_e_var"]
                + qubit_data[q]["Q_e_var"]
            ) / 4
            SNR = ((np.abs(Z)) ** 2) / (2 * var)
            if debug:
                plt.cla()
                plt.plot(freq[q], SNR, ".-")
                plt.title(f"resonator optimization qubit {q}")
                plt.xlabel("Readout frequency [Hz]")
                plt.ylabel("SNR")
                plt.tight_layout()
                plt.pause(0.1)
        # Find the readout frequency that maximizes the SNR
        f_opt = freq[q][np.argmax(SNR)]
        SNR_opt = SNR[np.argmax(SNR)]
        print(
            f"Previous optimal readout frequency: {machine.readout_resonators[q].f_opt:.1f} Hz with SNR = {SNR[len(freq[q])//2+1]:.2f}"
        )
        machine.readout_resonators[q].f_opt = (
            f_opt + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq
        )
        print(f"New optimal readout frequency: {machine.readout_resonators[q].f_opt:.1f} Hz with SNR = {SNR_opt:.2f}")
    machine.save("./labnotebook/state_after_" + experiment + "_" + now + ".json")
    machine.save("latest_quam.json")
