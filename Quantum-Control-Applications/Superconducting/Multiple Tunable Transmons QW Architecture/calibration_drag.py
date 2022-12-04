"""
Perform power drag calibration
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from datetime import datetime

##################
# State and QuAM #
##################
experiment = "drag_cal"
debug = True
simulate = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")


# machine.qubits[0].driving.drag_cosine.detuning = 2e6
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 1000

cooldown_time = 200 * u.us // 4

a_min = -1.9
a_max = 1.9
da = 0.05

alpha = np.arange(a_min, a_max + da / 2, da)

iter_min = 0
iter_max = 40
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

with program() as drag_cal:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    a = declare(fixed)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)
    it = declare(int)
    pulses = declare(int)
    state = [declare(bool) for _ in range(len(qubit_list))]
    state_st = [declare_stream() for _ in range(len(qubit_list))]

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, i)
        set_dc_offset(
            machine.qubits[i].name + "_flux", "single", machine.get_flux_bias_point(i, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, alpha)):
                with for_(*from_array(it, iters)):
                    with for_(pulses, iter_min, pulses <= it, pulses + d):
                        play("x180" * amp(1, 0, 0, a), machine.qubits[i].name)
                        play("x180" * amp(-1, 0, 0, -a), machine.qubits[i].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[i].name,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[i]),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[i]),
                    )
                    wait(cooldown_time, machine.readout_resonators[i].name)
                    assign(state[i], I[i] > machine.readout_resonators[i].ge_threshold)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(state[i], state_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_st[i].buffer(len(iters)).buffer(len(alpha)).average().save(f"I{i}")
            Q_st[i].buffer(len(iters)).buffer(len(alpha)).average().save(f"Q{i}")
            state_st[i].boolean_to_int().buffer(len(iters)).buffer(len(alpha)).average().save(f"state{i}")
            n_st[i].save(f"iteration{i}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, drag_cal, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(drag_cal)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]

    # Create the fitting object
    Fit = fitting.Fit()

    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"state{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["state"] = data[2]
            qubit_data[q]["iteration"] = data[3]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_2d(
                    iters,
                    alpha,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "Number of iterations",
                    "DRAG coefficient",
                    f"DRAG coefficient calibration for qubit {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
        fig = plt.figure()
        colors = ["b", "r", "g", "m", "c"]
        i = 0
        for it in [iter_min, int(0.25 * iter_max), int(0.5 * iter_max), int(0.75 * iter_max), iter_max]:
            z_I = np.polyfit(alpha, qubit_data[q]["I"][:, iters == it], 2)
            z_Q = np.polyfit(alpha, qubit_data[q]["Q"][:, iters == it], 2)
            plt.subplot(211)
            plt.plot(alpha, qubit_data[q]["I"][:, iters == it], colors[i] + "-", label=f"{it} iterations")
            # plt.plot(
            #     alpha, np.poly1d(np.squeeze(z_I))(alpha), colors[i] + ".", label=f"drag={(-z_I[1]/2/z_I[0])[0]:.3f}"
            # )
            plt.ylabel("I [a.u.]")
            plt.title(f"DRAG calibration for qubit {q}")
            plt.subplot(212)
            plt.plot(alpha, qubit_data[q]["Q"][:, iters == it], colors[i] + "-", label=f"{it} iterations")
            # plt.plot(
            #     alpha, np.poly1d(np.squeeze(z_Q))(alpha), colors[i] + ".", label=f"drag={(-z_Q[1]/2/z_Q[0])[0]:.3f}"
            # )
            plt.xlabel("DRAG coefficient alpha")
            plt.ylabel("Q [a.u.]")

            plt.tight_layout()
            i += 1
        plt.subplot(211)
        plt.legend(ncol=i)
        plt.subplot(212)
        plt.legend(ncol=i)
        # Update state with new DRAG coefficient
        print(f"Previous DRAG coefficient: {machine.qubits[q].driving.drag_cosine.alpha:.3f}")
        # Chose I, Q or amp...
        machine.qubits[q].driving.drag_cosine.alpha = (-z_I[1] / 2 / z_I[0])[0]
        print(f"New DRAG coefficient: {machine.qubits[q].driving.drag_cosine.alpha:.3f}")

machine.save("./lab_notebook/state_after_" + experiment + "_" + now + ".json")
# machine.save("latest_quam.json")
