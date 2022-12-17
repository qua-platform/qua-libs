"""
Measure the IQ blobs of the qubit in the ground and excited states to estimate the readout fidelity
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
from qm import SimulationConfig
import matplotlib.pyplot as plt
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.analysis.discriminator import two_state_discriminator
from macros import *

##################
# State and QuAM #
##################
experiment = "IQ_blobs_multiplexed"
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
n_runs = 10e3

with program() as iq_blobs:
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
    it = declare(int)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(machine.qubits[q].name + "_charge", "single",
                              machine.get_charge_bias_point(j, "working_point").value)

    with for_(it, 0, it < n_runs, it + 1):
        for i, q in enumerate(qubit_list):
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                demod.full("rotated_cos", I_g[i], "out1"),
                demod.full("rotated_sin", Q_g[i], "out1"),
            )
            wait_cooldown_time_fivet1(q, machine, simulate)
            save(I_g[i], I_g_st[i])
            save(Q_g[i], Q_g_st[i])

        align()  # global align

        for i, q in enumerate(qubit_list):
            play("x180", machine.qubits[q].name)
            # align() -- no need bc qb-rr share core and mess up multiplexing
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                demod.full("rotated_cos", I_e[i], "out1"),
                demod.full("rotated_sin", Q_e[i], "out1"),
            )
            wait_cooldown_time_fivet1(q, machine, simulate)
            save(I_e[i], I_e_st[i])
            save(Q_e[i], Q_e_st[i])

            save(it, n_st[i])

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_g_st[i].save_all(f"Ig{q}")
            Q_g_st[i].save_all(f"Qg{q}")
            I_e_st[i].save_all(f"Ie{q}")
            Q_e_st[i].save_all(f"Qe{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(iq_blobs)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i, q in enumerate(qubit_list):
        # Live plotting
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"Ig{q}", f"Qg{q}", f"Ie{q}", f"Qe{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_runs:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["Ig"] = data[0]
            qubit_data[i]["Qg"] = data[1]
            qubit_data[i]["Ie"] = data[2]
            qubit_data[i]["Qe"] = data[3]
            qubit_data[i]["iteration"] = data[4]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_runs, start_time=my_results.start_time)
        # PLot the IQ blobs end derive the readout fidelity
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            qubit_data[i]["Ig"],
            qubit_data[i]["Qg"],
            qubit_data[i]["Ie"],
            qubit_data[i]["Qe"],
            b_print=True,
            b_plot=True,
        )
        machine.readout_resonators[q].readout_fidelity = fidelity
        machine.readout_resonators[q].ge_threshold = threshold
        machine.readout_resonators[q].rotation_angle = angle
        plt.suptitle(f"Qubit {q}")
        figures.append(plt.gcf())

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
