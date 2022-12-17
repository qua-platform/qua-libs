"""
Perform time Ramsey with frame rotation to get T2*
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *

##################
# State and QuAM #
##################
experiment = "Ramsey_w_charge"
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
n_avg = 1e2

# Dephasing time scan
tau_min = 16 // 4  # in clock cycles
tau_max = 24000 // 4  # in clock cycles
d_tau = 40 // 4  # in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus

# charge bias scan
bias_min = -0.4
bias_max = 0.4
dbias = 0.005
bias = np.arange(bias_min, bias_max + dbias / 2, dbias)


with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_and_charge_relation)
    tau = declare(int)
    state = [declare(bool) for _ in range(len(qubit_and_charge_relation))]
    state_st = [declare_stream() for _ in range(len(qubit_and_charge_relation))]
    b = declare(fixed)

    for i, q in enumerate(qubit_and_charge_relation):

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(b, bias)):
                set_dc_offset(machine.qubits[q].name + "_charge", "single", b)
                wait(2000 // 4)  # wait for 2 us
                with for_(*from_array(tau, taus)):
                    play("x90", machine.qubits[q].name)
                    wait(tau, machine.qubits[q].name)
                    frame_rotation_2pi(
                        Cast.mul_fixed_by_int(machine.qubits[q].ramsey_det * 1e-9, 4 * tau), machine.qubits[q].name
                    )  # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    play("x90", machine.qubits[q].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("rotated_cos", I[i], "out1"),
                        demod.full("rotated_sin", Q[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate)
                    assign(state[i], I[i] > machine.readout_resonators[q].ge_threshold)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(state[i], state_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_and_charge_relation):
            I_st[i].buffer(len(taus)).buffer(len(bias)).average().save(f"I{q}")
            Q_st[i].buffer(len(taus)).buffer(len(bias)).average().save(f"Q{q}")
            state_st[i].boolean_to_int().buffer(len(taus)).buffer(len(bias)).average().save(f"state{q}")
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
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_and_charge_relation))]
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

    for i, q in enumerate(qubit_and_charge_relation):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        print("Qubit " + str(q))
        qubit_data[i]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"state{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[i]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[i]["I"] = data[0]
            qubit_data[i]["Q"] = data[1]
            qubit_data[i]["state"] = data[2]
            qubit_data[i]["iteration"] = data[3]
            # Progress bar
            progress_counter(qubit_data[i]["iteration"], n_avg, start_time=my_results.start_time)

            # live plot
            if debug:
                plot_demodulated_data_2d(
                    4 * taus,
                    bias,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Dephasing time [ns]",
                    "charge bias [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
