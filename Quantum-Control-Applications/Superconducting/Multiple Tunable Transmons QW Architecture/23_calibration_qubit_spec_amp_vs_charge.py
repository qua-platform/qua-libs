"""
Perform the 2D qubit spec as func of amp and charge
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
experiment = "2D_qubit_spectroscopy_amp_vs_charge"
debug = True
simulate = False
charge_lines = [0, 1]
injector_list = [0, 1]
digital = [1, 2, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
wait_time = 200
charge_point = "working_point"
qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment

# machine.get_qubit_gate(0, gate_shape).length = 1e-6
machine.get_sequence_state(0, "qubit_spectroscopy").length = (
    machine.get_qubit_gate(0, gate_shape).length + wait_time * 4e-9
)
config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e1

# Qubit pulse amplitude scan
a_min = 0.01
a_max = 0.99
na = 51
amps = np.linspace(a_min, a_max, na)
# charge bias scan
bias_min = [-0.1, -0.1]  # bias value specific to each qubit indeed
bias_max = [0.05, 0.05]  # each qubit can have a different working values
dbias = 0.005
bias = [np.arange(bias_min[i], bias_max[i] + dbias / 2, dbias) for i in range(len(qubit_and_charge_relation))]
# Ensure that charge biases remain in the [-0.5, 0.5) range
for i in qubit_and_charge_relation:
    assert np.all(bias[i] + machine.get_charge_bias_point(i, "working_point").value < 0.5)
    assert np.all(bias[i] + machine.get_charge_bias_point(i, "working_point").value >= -0.5)

with program() as qubit_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_and_charge_relation)
    a = declare(fixed)
    b = declare(fixed)

    for i, q in enumerate(qubit_and_charge_relation):
        set_dc_offset(
            machine.qubits[q].name + "_charge", "single", machine.get_charge_bias_point(i, "working_point").value
        )
        # Pre-factors to apply in order to get the bias scan
        pre_factors = bias[i] / machine.get_sequence_state(q, "qubit_spectroscopy").amplitude

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(b, pre_factors)):
                with for_(*from_array(a, amps)):
                    play("qubit_spectroscopy" * amp(b), machine.qubits[q].name + "_charge")
                    wait(wait_time, machine.qubits[q].name)
                    play("saturation" * amp(a), machine.qubits[q].name)
                    align()
                    wait(16, machine.readout_resonators[q].name)
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("cos", I[i], "out1"),
                        demod.full("sin", Q[i], "out1"),
                    )
                    wait_cooldown_time(5 * machine.qubits[q].t1, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i, q in enumerate(qubit_and_charge_relation):
            I_st[i].buffer(len(amps)).buffer(len(bias[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(amps)).buffer(len(bias[i])).average().save(f"Q{q}")
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
    job = qmm.simulate(config, qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_and_charge_relation))]
    figures = []
    for i, q in enumerate(qubit_and_charge_relation):
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
                    amps * machine.get_qubit_gate(q, gate_shape).angle2volt.deg180,
                    bias[i] + machine.get_charge_bias_point(q, charge_point).value,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Microwave drive amplitude [V]",
                    "charge bias [V]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
