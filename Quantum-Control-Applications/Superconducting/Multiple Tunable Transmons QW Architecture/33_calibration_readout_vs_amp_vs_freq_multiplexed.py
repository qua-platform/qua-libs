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
from macros import *

##################
# State and QuAM #
##################
experiment = "readout_amplitude_frequency_optimization_multiplexed"
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
n_avg = 10

# Amplitude scan
a_min = 0.2
a_max = 1.00
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)

# Frequency scan
span = 2e6
df = 0.1e6
freq = [np.arange(machine.get_readout_IF(i) - span, machine.get_readout_IF(i) + span + df / 2, df) for i in qubit_list]
spans = np.arange(-span, span + df / 2, df)

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
    f = declare(int)
    f_res = [declare(int) for _ in range(len(qubit_list))]
    it = declare(int)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(machine.qubits[q].name + "_charge", "single",
                              machine.get_charge_bias_point(j, "working_point").value)

    with for_(it, 0, it < n_avg, it + 1):
        with for_(*from_array(a, amps)):
            with for_(*from_array(f, spans)):
                for i, q in enumerate(qubit_list):
                    assign(f_res[i], machine.get_readout_IF(i) + f)
                    update_frequency(machine.readout_resonators[q].name, f_res[i])
                    measure(
                        "readout" * amp(a),
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I_g[i], "out1"),
                        demod.full("sin", Q_g[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate)
                    assign(state[i], I_g[i] > machine.readout_resonators[q].ge_threshold)
                    save(I_g[i], I_g_st[i])
                    save(Q_g[i], Q_g_st[i])
                    save(state[i], state_g_st[i])

                align()  # global align

                for i, q in enumerate(qubit_list):
                    play("x180", machine.qubits[q].name)
                    # align() -- not needed because pairs of qb-rr share cores and will also mess up the multiplexing
                    measure(
                        "readout" * amp(a),
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I_e[i], "out1"),
                        demod.full("sin", Q_e[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate)
                    assign(state[i], I_e[i] > machine.readout_resonators[q].ge_threshold)
                    save(I_e[i], I_e_st[i])
                    save(Q_e[i], Q_e_st[i])
                    save(state[i], state_e_st[i])
                    save(n[i], n_st[i])

    with stream_processing():
        for i, q in enumerate(qubit_list):
            n_st[i].save(f"iteration{q}")
            # state
            state_g_st[i].boolean_to_int().buffer(len(freq[i])).buffer(len(amps)).average().save(f"state_g{q}")
            state_e_st[i].boolean_to_int().buffer(len(freq[i])).buffer(len(amps)).average().save(f"state_e{q}")
            # mean values
            I_g_st[i].buffer(len(freq[i])).buffer(len(amps)).average().save(f"I_g_avg{q}")
            Q_g_st[i].buffer(len(freq[i])).buffer(len(amps)).average().save(f"Q_g_avg{q}")
            I_e_st[i].buffer(len(freq[i])).buffer(len(amps)).average().save(f"I_e_avg{q}")
            Q_e_st[i].buffer(len(freq[i])).buffer(len(amps)).average().save(f"Q_e_avg{q}")

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
            qubit_data[i]["state_g"] = data[4]
            qubit_data[i]["state_e"] = data[5]
            qubit_data[i]["iteration"] = data[6]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
            # Derive SNR
            Z = (qubit_data[i]["I_e_avg"] - qubit_data[i]["I_g_avg"]) + 1j * (
                qubit_data[i]["Q_e_avg"] - qubit_data[i]["Q_g_avg"]
            )
            SNR = ((np.abs(Z)) ** 2)
            if debug:
                plt.cla()
                plt.pcolor((machine.get_readout_IF(i) + spans + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq) * 1e-9, amps * machine.readout_resonators[q].readout_amplitude, SNR, cmap='magma')
                plt.title(f"resonator optimization qubit {q}")
                plt.xlabel("Readout frequency [Hz]")
                plt.ylabel("Pulse amplitude [V]")
                plt.tight_layout()
                plt.pause(0.1)
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
