"""
Perform the 1D resonator spectroscopy
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
from macros import *

##################
# State and QuAM #
##################
experiment = "1D_resonator_spectroscopy"
debug = True
simulate = False
fit_data = True
injector_list = [0, 1]
digital = [1, 2, 9]
charge_lines=[0, 1]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment
# amplitudes = [0.005, 0.005, 0.005, 0.005, 0.005, 0.005]
# f_opts = [6.231e9, 6.141e9, 6.231e9, 6.141e9, 6.231e9, 6.231e9]
# machine.readout_lines[0].lo_freq = 6.0e9
# machine.readout_lines[0].lo_power = 13
# machine.readout_lines[0].length = 3e-6
# populate_machine_resonators(machine, qubit_list, amplitudes, f_opts)
# machine.readout_resonators[0].readout_amplitude =0.005
# machine.readout_resonators[0].f_opt = 6.131e9
# machine.readout_resonators[0].f_opt = machine.readout_resonators[0].f_res
config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e3

span = 2e6
df = 0.1e6
freq = [np.arange(machine.get_readout_IF(i) - span, machine.get_readout_IF(i) + span + df / 2, df) for i in qubit_list]
spans = np.arange(-span, span + df / 2, df)

with program() as resonator_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    it = declare(int)
    it_st = declare_stream()
    f = declare(int)
    f_res = [declare(int) for _ in range(len(qubit_list))]

    with for_(it, 0, it < n_avg, it + 1):
        with for_(*from_array(f, spans)):
            for i, q in enumerate(qubit_list):
                assign(f_res[i], machine.get_readout_IF(i) + f)
                update_frequency(machine.readout_resonators[q].name, f_res[i])
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    demod.full("cos", I[i], "out1"),
                    demod.full("sin", Q[i], "out1"),
                )
                wait_cooldown_time(machine.readout_resonators[q].relaxation_time, simulate)
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
                save(it, n_st[i])

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
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    # Create the fitting object
    Fit = fitting.Fit()
    figures = []
    for i, q in enumerate(qubit_list):
        # Live plotting
        if debug:
            fig = plt.figure()
            figures.append(fig)
            interrupt_on_close(fig, job)
        print(f"Qubit {q}")
        qubit_data[i]["iteration"] = 0
        exit = False
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
                    (machine.get_readout_IF(i) + spans + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq) * 1e-9,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "frequency [GHz]",
                    f"resonator spectroscopy qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"marker": "."},
                )
            # Fitting
            if fit_data:
                try:
                    Fit.reflection_resonator_spectroscopy(
                        (machine.get_readout_IF(i) + spans + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq) * 1e-9,
                        np.sqrt(qubit_data[i]["I"] ** 2 + qubit_data[i]["Q"] ** 2),
                        plot=False,
                    )
                    plt.subplot(211)
                    plt.cla()
                    fit = Fit.reflection_resonator_spectroscopy(
                        (machine.get_readout_IF(i) + spans + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq) * 1e-9,
                        np.sqrt(qubit_data[i]["I"] ** 2 + qubit_data[i]["Q"] ** 2),
                        plot=True,
                    )
                    plt.pause(0.1)
                except (Exception,):
                    pass
            # Break the loop if interrupt on close
            if not i == (len(qubit_list)-1):
                if my_results.is_processing():
                    if not my_results.is_processing():
                        exit = True
                        break
        if exit:
            break
        # Update state with new resonance frequency
        if fit_data:
            print(f"Previous resonance frequency: {machine.readout_resonators[q].f_res * 1e-9:.6f} GHz")
            machine.readout_resonators[q].f_res = np.round(fit["f"][0])
            machine.readout_resonators[q].f_opt = machine.readout_resonators[q].f_res
            print(f"New resonance frequency: {machine.readout_resonators[q].f_res * 1e-9:.6f} GHz")
            print(f"New resonance IF frequency: {machine.get_readout_IF(q) * 1e-6:.3f} MHz")

    machine.save_results(experiment, figures)

# machine.save("latest_quam.json")


