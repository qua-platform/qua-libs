"""
performs 1D qubit spectroscopy
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
experiment = "1D_qubit_spectroscopy"
debug = True
simulate = False
fit_data = False
charge_lines = [0, 1]
injector_list = [0, 1]
digital = [1, 2, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment
amplitudes = [0.005319465640078984, 0.01, 0.03, 0.02, 0.08, 0.0005138336244760865]
fs = [4.28e9, 4.429e9, 4.56e9, 4.69e9, 4.76e9, 4.8e9]
lens = [50e-6, 50e-6, 50e-6, 50e-6, 50e-6, 50e-6]
machine.drive_lines[0].lo_freq = 4.5e9
machine.drive_lines[0].lo_power = 16
machine.drive_lines[1].lo_freq = 4.5e9
machine.drive_lines[1].lo_power = 16
# machine.get_qubit_gate(0, gate_shape).angle2volt.deg180 = 0.2
# machine.get_qubit_gate(0, gate_shape).length = 200e-9
populate_machine_qubits_saturation(machine, qubit_list, amplitudes, fs, lens)
config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e1

# Frequency scan
freq_span = 10e6
df = 0.1e6
freq = [
    np.arange(machine.get_qubit_IF(i) - freq_span, machine.get_qubit_IF(i) + freq_span + df / 2, df) for i in qubit_list
]

with program() as qubit_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    b = declare(fixed)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(machine.qubits[q].name + "_charge", "single",
                              machine.get_charge_bias_point(j, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(f, freq[i])):
                update_frequency(machine.qubits[q].name, f)
                play("saturation", machine.qubits[q].name)
                align()
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    demod.full("cos", I[i], "out1"),
                    demod.full("sin", Q[i], "out1"),
                )
                wait_cooldown_time_fivet1(q, machine, simulate)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(freq[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(freq[i])).average().save(f"Q{q}")
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
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    # Create the fitting object
    Fit = fitting.Fit()
    figures = []
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
            if debug:
                plot_demodulated_data_1d(
                    freq[i] + machine.drive_lines[machine.qubits[q].wiring.drive_line_index].lo_freq,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Microwave drive frequency [Hz]",
                    f"{experiment}_qubit{q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    fit = Fit.reflection_resonator_spectroscopy(
                        freq[i] + machine.drive_lines[machine.qubits[q].wiring.drive_line_index].lo_freq, np.sqrt(qubit_data[i]["I"]**2 + qubit_data[i]["Q"]**2),
                    )
                    plt.subplot(211)
                    plt.cla()
                    fit = Fit.reflection_resonator_spectroscopy(
                        freq[i] + machine.drive_lines[machine.qubits[q].wiring.drive_line_index].lo_freq, np.sqrt(qubit_data[i]["I"]**2 + qubit_data[i]["Q"]**2),
                        plot=debug)
                    plt.pause(0.1)
                except Exception as e:
                    pass

        # Update state with new resonance frequency
        if fit_data:
            print(f"Qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")
            machine.qubits[q].f_01 = np.round(fit["f"][0])
            print(f"New qubit frequency: {machine.qubits[q].f_01 * 1e-9:.6f} GHz")
            print(f"New resonance IF frequency: {machine.get_qubit_IF(q) * 1e-6:.3f} MHz")

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
