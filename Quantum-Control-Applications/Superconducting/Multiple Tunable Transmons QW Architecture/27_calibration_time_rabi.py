"""
time_rabi.py: performs time rabi
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

##################
# State and QuAM #
##################
experiment = "time_rabi"
debug = True
simulate = False
fit_data = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

gate_length = []
for q in qubit_list:
    gate_length.append(machine.get_qubit_gate(q, gate_shape).length)
    machine.get_qubit_gate(q, gate_shape).length = 16e-9

config = machine.build_config(digital, qubit_list, gate_shape)
for q in qubit_list:
    machine.get_qubit_gate(q, gate_shape).length = gate_length[q]

###################
# The QUA program #
###################
n_avg = 5000

# Gate length scan
t_min = 16 // 4
t_max = 1000 // 4
dt = 1
lengths = np.arange(t_min, t_max + dt / 2, dt)

with program() as time_rabi:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    t = declare(int)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * machine.qubits[q].t1 // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "near_anti_crossing").value
        )

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            update_frequency(machine.qubits[q].name, int(machine.get_qubit_IF(0)))
            with for_(*from_array(t, lengths)):
                play("x180", machine.qubits[q].name, duration=t)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[q]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[q]),
                )
                wait(cooldown_time, machine.readout_resonators[q].name)
                save(I[q], I_st[q])
                save(Q[q], Q_st[q])
            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_st[q].buffer(len(lengths)).average().save(f"I{q}")
            Q_st[q].buffer(len(lengths)).average().save(f"Q{q}")
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
    job = qmm.simulate(config, time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(time_rabi)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    # Create the fitting object
    Fit = fitting.Fit()

    for q in range(len(qubit_list)):
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Get results from QUA program
        my_results = fetching_tool(job, [f"I{q}", f"Q{q}", f"iteration{q}"], mode="live")
        while my_results.is_processing() and qubit_data[q]["iteration"] < n_avg - 1:
            # Fetch results
            data = my_results.fetch_all()
            qubit_data[q]["I"] = data[0]
            qubit_data[q]["Q"] = data[1]
            qubit_data[q]["iteration"] = data[2]
            # Progress bar
            progress_counter(qubit_data[q]["iteration"], n_avg, start_time=my_results.start_time)
            # live plot
            if debug:
                plot_demodulated_data_1d(
                    lengths * 4,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "x180 length [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    Fit.rabi(lengths * 4, qubit_data[q]["I"])
                    plt.subplot(211)
                    plt.cla()
                    fit_I = Fit.rabi(lengths * 4, qubit_data[q]["I"], plot=debug)
                    plt.pause(0.1)
                except (Exception,):
                    pass
                try:
                    Fit.rabi(lengths * 4, qubit_data[q]["Q"])
                    plt.subplot(211)
                    plt.cla()
                    fit_Q = Fit.rabi(lengths * 4, qubit_data[q]["Q"], plot=debug)
                    plt.pause(0.1)
                except (Exception,):
                    pass

        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous x180 length: {machine.get_qubit_gate(q, gate_shape).length*1e9:.0f} ns")
            machine.get_qubit_gate(q, gate_shape).length = max(
                16, np.round(0.5 / fit_I["f"][0]) - np.round(0.5 / fit_I["f"][0]) % 4
            )
            print(f"New x180 length: {machine.get_qubit_gate(q, gate_shape).length*1e9:.0f} ns")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
