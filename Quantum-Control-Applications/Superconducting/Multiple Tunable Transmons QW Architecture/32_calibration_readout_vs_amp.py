"""
Use IQ blobs measurements to find optimal readout amplitude
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
from macros import wait_cooldown_time

##################
# State and QuAM #
##################
experiment = "readout_amplitude_optimization"
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
n_avg = 4e3

# Amplitude scan
a_min = 0.2
a_max = 1.99
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)

with program() as readout_opt:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I_g = [declare(fixed) for _ in range(len(qubit_list))]
    Q_g = [declare(fixed) for _ in range(len(qubit_list))]
    I_g_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_g_st = [declare_stream() for _ in range(len(qubit_list))]
    I_e = [declare(fixed) for _ in range(len(qubit_list))]
    Q_e = [declare(fixed) for _ in range(len(qubit_list))]
    I_e_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_e_st = [declare_stream() for _ in range(len(qubit_list))]
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_g_st = [declare_stream() for _ in range(len(qubit_list))]
    state_e_st = [declare_stream() for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amps)):
                measure(
                    "readout" * amp(a),
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_g[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_g[i]),
                )
                wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                assign(state[i], I_g[i] > machine.readout_resonators[q].ge_threshold)
                save(I_g[i], I_g_st[i])
                save(Q_g[i], Q_g_st[i])
                save(state[i], state_g_st[i])

                align()  # global align

                play("x180", machine.qubits[q].name)
                align()
                measure(
                    "readout" * amp(a),
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I_e[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q_e[i]),
                )
                wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                assign(state[i], I_e[i] > machine.readout_resonators[q].ge_threshold)
                save(I_e[i], I_e_st[i])
                save(Q_e[i], Q_e_st[i])
                save(state[i], state_e_st[i])

            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            n_st[i].save(f"iteration{q}")
            # state
            state_g_st[i].boolean_to_int().buffer(len(amps)).average().save(f"state_g{q}")
            state_e_st[i].boolean_to_int().buffer(len(amps)).average().save(f"state_e{q}")
            # mean values
            I_g_st[i].buffer(len(amps)).average().save(f"I_g_avg{q}")
            Q_g_st[i].buffer(len(amps)).average().save(f"Q_g_avg{q}")
            I_e_st[i].buffer(len(amps)).average().save(f"I_e_avg{q}")
            Q_e_st[i].buffer(len(amps)).average().save(f"Q_e_avg{q}")
            # variances
            (
                ((I_g_st[i].buffer(len(amps)) * I_g_st[i].buffer(len(amps))).average())
                - (I_g_st[i].buffer(len(amps)).average() * I_g_st[i].buffer(len(amps)).average())
            ).save(f"I_g_var{q}")
            (
                ((Q_g_st[i].buffer(len(amps)) * Q_g_st[i].buffer(len(amps))).average())
                - (Q_g_st[i].buffer(len(amps)).average() * Q_g_st[i].buffer(len(amps)).average())
            ).save(f"Q_g_var{q}")
            (
                ((I_e_st[i].buffer(len(amps)) * I_e_st[i].buffer(len(amps))).average())
                - (I_e_st[i].buffer(len(amps)).average() * I_e_st[i].buffer(len(amps)).average())
            ).save(f"I_e_var{q}")
            (
                ((Q_e_st[i].buffer(len(amps)) * Q_e_st[i].buffer(len(amps))).average())
                - (Q_e_st[i].buffer(len(amps)).average() * Q_e_st[i].buffer(len(amps)).average())
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
    for i, q in enumerate(qubit_list):
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
                f"state_g{q}",
                f"state_e{q}",
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
            qubit_data[i]["state_g"] = data[8]
            qubit_data[i]["state_e"] = data[9]
            qubit_data[i]["iteration"] = data[10]
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
                plt.subplot(211)
                plt.cla()
                plt.plot(amps * machine.readout_resonators[q].readout_amplitude, SNR, ".-")
                plt.title(f"resonator optimization qubit {q}")
                plt.xlabel("Pulse amplitude [V]")
                plt.ylabel("SNR")
                plt.tight_layout()
                # plt.pause(0.1)
                plt.subplot(212)
                plt.cla()
                plt.plot(amps * machine.readout_resonators[q].readout_amplitude, qubit_data[i]["state_g"])
                plt.plot(amps * machine.readout_resonators[q].readout_amplitude, qubit_data[i]["state_e"])
                plt.title(f"resonator optimization qubit {q}")
                plt.xlabel("Pulse amplitude [V]")
                plt.ylabel("Probability")
                plt.yscale("log")
                plt.xscale("log")
                plt.tight_layout()
                plt.pause(0.1)
        # Find the readout amplitude that maximizes the SNR
        amp_opt = amps[np.argmax(SNR)] * machine.readout_resonators[q].readout_amplitude
        SNR_opt = SNR[np.argmax(SNR)]
        print(
            f"Previous optimal readout amplitude: {machine.readout_resonators[q].readout_amplitude:.1f} V with SNR = {SNR[len(amps)//2+1]:.2f}"
        )
        machine.readout_resonators[q].readout_amplitude = amp_opt
        print(
            f"New optimal readout amplitude: {machine.readout_resonators[q].readout_amplitude:.1f} V with SNR = {SNR_opt:.2f}"
        )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
