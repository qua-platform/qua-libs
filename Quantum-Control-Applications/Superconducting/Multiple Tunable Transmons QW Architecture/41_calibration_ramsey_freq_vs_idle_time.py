"""
2D Ramsey frequency versus dephasing time
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *
from config import NUMBER_OF_QUBITS_W_CHARGE

##################
# State and QuAM #
##################
experiment = "2D_ramsey_freq_vs_idle_time"
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
n_avg = 1e1

# Frequency scan
freq_span = 10e6
df = 0.1e6
freq = [
    np.arange(machine.get_qubit_IF(i) - freq_span, machine.get_qubit_IF(i) + freq_span + df / 2, df) for i in qubit_list
]
# Dephasing time scan
tau_min = 4  # in clock cycles
tau_max = 1000  # in clock cycles
d_tau = 20  # in clock cycles

taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus


# QUA program
with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    c = declare(fixed, value=1e-9)
    tau = declare(int)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        if q in qubit_w_charge_list:
            set_dc_offset(machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(q, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                if q in qubit_w_charge_list:
                    update_frequency(machine.qubits[q].name, f)
                else:
                    update_frequency(machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name, f)

                with for_(*from_array(tau, taus)):
                    if q in qubit_w_charge_list:
                        play("x90", machine.qubits[q].name)
                        wait(tau, machine.qubits[q].name)
                        play("x90", machine.qubits[q].name)
                    else:
                        play("x90", machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                        wait(tau, machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                        play("x90", machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("rotated_cos", I[i], "out1"),
                        demod.full("rotated_sin", Q[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate, qubit_w_charge_list)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(taus)).buffer(len(freq[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(taus)).buffer(len(freq[i])).average().save(f"Q{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for i, q in enumerate(qubit_list):
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["iteration"] = data[2]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                if q in qubit_w_charge_list:
                    plot_demodulated_data_2d(
                        taus * 4,
                        freq[i] + machine.drive_lines[machine.qubits[q].wiring.drive_line_index].lo_freq,
                        qubit_data[i]["I"],
                        qubit_data[i]["Q"],
                        "Dephasing time [ns]",
                        "drive frequency [Hz]",
                        f"{experiment} qubit {q}",
                        amp_and_phase=False,
                        fig=fig,
                        plot_options={"cmap": "magma"},
                    )
                else:
                    plot_demodulated_data_2d(
                        taus * 4,
                        freq[i] + machine.drive_lines[machine.qubits_wo_charge[q - NUMBER_OF_QUBITS_W_CHARGE].wiring.drive_line_index].lo_freq,
                        qubit_data[i]["I"],
                        qubit_data[i]["Q"],
                        "Dephasing time [ns]",
                        "drive frequency [Hz]",
                        f"{experiment} qubit {q}",
                        amp_and_phase=True,
                        fig=fig,
                        plot_options={"cmap": "magma"},
                    )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
