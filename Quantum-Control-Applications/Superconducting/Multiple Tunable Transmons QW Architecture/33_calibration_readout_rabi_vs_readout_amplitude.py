"""
performs time Rabi vs readout amplitude and look for the highest contrast
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

##################
# State and QuAM #
##################
experiment = "Rabi_vs_readout_amplitude"
debug = True
simulate = False
fit_data = True
qubit_list = [1, 0]
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
n_avg = 4e2

# Readout amplitude scan
a_min = 0.9
a_max = 1.1
da = 0.01
amps = np.arange(a_min, a_max + da / 2, da)

# Gate duration scan
t_min = 16 // 4
t_max = 2000 // 4
dt = 10
lengths = np.arange(t_min, t_max + dt / 2, dt)


# QUA program
with program() as readout_opt:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    a = declare(fixed)
    t = declare(int)

    for i, q in enumerate(qubit_list):
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "zero_frequency_point").value
        )

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(t, lengths)):
                with for_(*from_array(a, amps)):
                    play("x180", machine.qubits[q].name, duration=t)
                    align()
                    measure(
                        "readout" * amp(a),
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
            I_st[i].buffer(len(amps)).buffer(len(lengths)).average().save(f"I{q}")
            Q_st[i].buffer(len(amps)).buffer(len(lengths)).average().save(f"Q{q}")
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
    job = qmm.simulate(config, readout_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(readout_opt)

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
                plot_demodulated_data_2d(
                    amps * machine.readout_resonators[q].readout_amplitude,
                    lengths * 4,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Readout amplitude [V]",
                    "Gate length [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")