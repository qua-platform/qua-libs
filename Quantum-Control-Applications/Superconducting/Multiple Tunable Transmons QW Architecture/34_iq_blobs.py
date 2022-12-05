"""
Measure the IQ blobs of the qubit in the ground and excited states to estimate the readout fidelity
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.analysis.discriminator import two_state_discriminator
import matplotlib.pyplot as plt

##################
# State and QuAM #
##################
experiment = "IQ_blobs"
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

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * int(machine.qubits[q].t1 * 1e9) // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "working_point").value)

        with for_(n[q], 0, n[q] < n_runs, n[q] + 1):
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g[q]),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g[q]),
            )
            wait(cooldown_time, machine.readout_resonators[q].name)
            save(I_g[q], I_g_st[q])
            save(Q_g[q], Q_g_st[q])

            align()

            play("x180", machine.qubits[q].name)
            align()
            measure(
                "readout",
                machine.readout_resonators[q].name,
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e[q]),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e[q]),
            )
            wait(cooldown_time, machine.readout_resonators[q].name)
            save(I_e[q], I_e_st[q])
            save(Q_e[q], Q_e_st[q])

            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_g_st[q].save_all(f"Ig{q}")
            Q_g_st[q].save_all(f"Qg{q}")
            I_e_st[q].save_all(f"Ie{q}")
            Q_e_st[q].save_all(f"Qe{q}")
            n_st[q].save(f"iteration{q}")

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
    for q in range(len(qubit_list)):
        # Live plotting
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"Ig{q}", f"Qg{q}", f"Ie{q}", f"Qe{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_runs:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["Ig"] = data[0]
            qubit_data[q]["Qg"] = data[1]
            qubit_data[q]["Ie"] = data[2]
            qubit_data[q]["Qe"] = data[3]
            qubit_data[q]["iteration"] = data[4]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_runs, start_time=my_results.start_time)
        # PLot the IQ blobs end derive the readout fidelity
        angle, threshold, fidelity, gg, ge, eg, ee = two_state_discriminator(
            qubit_data[q]["Ig"],
            qubit_data[q]["Qg"],
            qubit_data[q]["Ie"],
            qubit_data[q]["Qe"],
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
