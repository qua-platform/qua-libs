"""
T1 measurement
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from macros import *


##################
# State and QuAM #
##################
experiment = "T1"
debug = True
simulate = False
fit_data = True
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 5000

# Wait time duration
t_min = 16 // 4
t_max = 100000 // 4
dt = 300

# lengths = np.arange(t_min, t_max + dt / 2, dt)
lengths = np.logspace(np.log10(t_min), np.log10(t_max), 40)
# If logarithmic increment, then need to check that no items have the same integer part
assert len(np.where(np.diff(lengths.astype(int))==0)[0]) == 0

# QUA program
with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    t = declare(int)

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "near_anti_crossing").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            update_frequency(machine.qubits[q].name, int(machine.get_qubit_IF(0)))
            with for_(*from_array(t, lengths)):
                play("x180", machine.qubits[q].name)
                wait(t, machine.qubits[q].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                )
                wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(lengths)).average().save(f"I{q}")
            Q_st[i].buffer(len(lengths)).average().save(f"Q{q}")
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
    job = qmm.simulate(config, T1, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(T1)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    if np.isclose(np.std(lengths[1:] / lengths[:-1]), 0, atol=1e-3):
        lengths = get_equivalent_log_array(lengths)
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
            if debug and not fit_data:
                plot_demodulated_data_1d(
                    lengths * 4,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Wait time [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    plt.subplot(211)
                    plt.cla()
                    fit_I = Fit.T1(lengths * 4, qubit_data[i]["I"], plot=debug)
                    plt.subplot(212)
                    plt.cla()
                    fit_Q = Fit.T1(lengths * 4, qubit_data[i]["Q"], plot=debug)
                    plt.pause(0.01)
                except (Exception,):
                    pass

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous T1: {machine.qubits[q].t1:.3e} s")
            machine.qubits[q].t1 = fit_I["T1"][0] * 1e-9
            print(f"New T1: {machine.qubits[q].t1:.3e} s")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")