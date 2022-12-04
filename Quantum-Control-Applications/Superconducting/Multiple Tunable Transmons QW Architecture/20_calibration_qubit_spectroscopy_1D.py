"""
performs 1D qubit spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_1d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array

##################
# State and QuAM #
##################
experiment = "1D_qubit_spectroscopy"
debug = True
simulate = False
fit_data = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

machine.get_qubit_gate(0, gate_shape).angle2volt.deg180 = 0.2
machine.get_qubit_gate(0, gate_shape).length = 200e-9
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e2

# Frequency scan
freq_span = 10e6
df = 0.1e6
freq = [
    np.arange(machine.get_qubit_IF(i) - freq_span, machine.get_qubit_IF(i) + freq_span + df / 2, df) for i in qubit_list
]

with program() as qubit_spec:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)
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
        set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "readout").value)

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            update_frequency(machine.qubits[q].name, int(machine.get_qubit_IF(0)))
            with for_(*from_array(f, freq[q])):
                update_frequency(machine.qubits[q].name, f)
                play("x180", machine.qubits[q].name)
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
            I_st[q].buffer(len(freq[q])).average().save(f"I{q}")
            Q_st[q].buffer(len(freq[q])).average().save(f"Q{q}")
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
                    freq[q] + machine.drive_lines[q].lo_freq,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
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
                        freq[q] + machine.drive_lines[q].lo_freq,
                        np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2),
                    )
                    plt.subplot(211)
                    plt.cla()
                    fit = Fit.reflection_resonator_spectroscopy(
                        freq[q] + machine.drive_lines[q].lo_freq,
                        np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2),
                        plot=debug,
                    )
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
