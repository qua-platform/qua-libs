"""
Perform the 2D resonator spectroscopy frequency vs readout amplitude
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from macros import *
from qualang_tools.loops import from_array


##################
# State and QuAM #
##################
experiment = "2D_resonator_spectroscopy_vs_amp"
debug = True
simulate = False
qubit_w_charge_list = [0, 1]
qubit_wo_charge_list = [2, 3, 4, 5]
qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment
injector_list = [0, 1]
digital = [1, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

# machine.readout_resonators[0].f_opt = machine.readout_resonators[0].f_res
# machine.readout_resonators[0].readout_amplitude = 0.4
config = machine.build_config(digital, qubit_w_charge_list, qubit_wo_charge_list, injector_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 1000

# Frequency scan
freq_span = 2e6
df = 0.1e6
freq = [
    np.arange(machine.get_readout_IF(i) - freq_span, machine.get_readout_IF(i) + freq_span + df / 2, df)
    for i in qubit_list
]
# Bias scan
a_min = 0.1
a_max = 1
da = 0.01
amps = [np.arange(a_min, a_max + da/2, da) for i in range(len(qubit_list))]
# amps = [np.logspace(-2, np.log10(1.1), 31) for i in range(len(qubit_list))]

# QUA program
with program() as resonator_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    a = declare(fixed)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        if q == 0 or q == 1:
            set_dc_offset(machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(q, "working_point").value)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(a, amps[i])):
                with for_(*from_array(f, freq[i])):
                    update_frequency(machine.readout_resonators[q].name, f)
                    measure(
                        "readout"*amp(a),
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I[i], "out1"),
                        demod.full("sin", Q[i], "out1"),
                    )
                    wait_cooldown_time(machine.readout_resonators[q].relaxation_time, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(freq[i])).buffer(len(amps[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(freq[i])).buffer(len(amps[i])).average().save(f"Q{q}")
            n_st[i].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=10000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    # Live plotting
    figures = []

    for i, q in enumerate(qubit_list):
        scaling = np.array([amps[i] for _ in range(len(freq[i]))]).transpose()
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
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
                plot_demodulated_data_2d(
                    (freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq)*1e-6,
                    amps[i] * machine.readout_resonators[q].readout_amplitude,
                    qubit_data[i]["I"] / scaling,
                    qubit_data[i]["Q"] / scaling,
                    "Readout frequency [MHz]",
                    "Readout amplitude [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    plot_options={"cmap": "magma"},
                    fig=fig,
                )
            # Break the loop if interupt on close
            if my_results.is_processing():
                if not my_results.is_processing():
                    exit = True
                    break
        if exit:
            break

    # need to update quam with readout amplitude
    # machine.readout_resonators[q].readout_amplitude = 0.02
    # machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
