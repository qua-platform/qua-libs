"""
Perform time Ramsey with frame rotation to get T2*
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from macros import *

##################
# State and QuAM #
##################
experiment = "Ramsey"
debug = True
simulate = False
fit_data = True
qubit_list = [1,0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 1e3

# Dephasing time scan
tau_min = 16 // 4 # in clock cycles
tau_max = 24000 // 4  # in clock cycles
d_tau = 40 // 4 # in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    tau = declare(int)
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
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
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[i]),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[i]),
                )
                wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                assign(state[i], I[i] > machine.readout_resonators[q].ge_threshold)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
                save(state[i], state_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(taus)).average().save(f"I{q}")
            Q_st[i].buffer(len(taus)).average().save(f"Q{q}")
            state_st[i].boolean_to_int().buffer(len(taus)).average().save(f"state{q}")
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
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

    for i, q in enumerate(qubit_list):
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
                plot_demodulated_data_1d(
                    4 * taus,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Dephasing time [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    Fit.ramsey(4 * taus, qubit_data[i]["I"])
                    plt.subplot(211)
                    plt.cla()
                    fit_I = Fit.ramsey(4 * taus, qubit_data[i]["I"], plot=debug)
                    plt.subplot(212)
                    plt.cla()
                    fit_Q = Fit.ramsey(4 * taus, qubit_data[i]["I"], plot=debug)
                    # plt.subplot(313)
                    # plt.cla()
                    # fit_state = Fit.ramsey(4 * taus, qubit_data[i]["state"], plot=debug)
                except (Exception,):
                    pass

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")
            machine.qubits[q].f_01 = machine.qubits[q].f_01 - (np.round(fit_I["f"][0] * 1e9) - machine.qubits[0].ramsey_det)
            machine.qubits[q].t2star = fit_I["T2"][0] * 1e-9
            print(f"New qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
