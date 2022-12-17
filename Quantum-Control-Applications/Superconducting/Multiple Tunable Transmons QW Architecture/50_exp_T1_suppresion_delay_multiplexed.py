"""
T1 suppression experiment by qp injection (delay tuning)
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, fitting, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array, get_equivalent_log_array
from macros import *

##################
# State and QuAM #
##################
experiment = "T1_suppression_delay_multiplexed"
debug = True
simulate = False
charge_lines = [0, 1]
injector_list = [0, 1]
digital = [1, 2, 9]
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
qubit_list = [0, 1, 2, 3, 4, 5]  # you can shuffle the order at which you perform the experiment

config = machine.build_config(digital, qubit_list, injector_list, charge_lines, gate_shape)

###################
# The QUA program #
###################
n_avg = 5

# Wait time duration
t_min = 16 // 4
t_max = 100000 // 4
dt = 300

lengths = np.arange(t_min, t_max + dt / 2, dt)
# lengths = np.logspace(np.log10(t_min), np.log10(t_max), 40)
# If logarithmic increment, then need to check that no items have the same integer part
assert len(np.where(np.diff(lengths.astype(int)) == 0)[0]) == 0

# Delay between qp_injection and T1 experiment
delay_min = 16 // 4
delay_max = 200000 // 4
d_delay = 300

delays = np.arange(delay_min, delay_max + d_delay / 2, d_delay)

# QUA program
with program() as T1:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    t = declare(int)
    d = declare(int)
    iters = declare(int)

    for i, q in enumerate(qubit_list):
        # set qubit frequency to working point
        for j, z in enumerate(qubit_and_charge_relation):
            if q == z:
                set_dc_offset(
                    machine.qubits[q].name + "_charge",
                    "single",
                    machine.get_charge_bias_point(j, "working_point").value,
                )

    with for_(iters, 0, iters < n_avg, iters + 1):
        with for_(*from_array(d, delays)):
            with for_(*from_array(t, lengths)):
                play("injector", machine.qp_injectors[0].name)
                wait(d)
                align()
                for i, q in enumerate(qubit_list):
                    play("x180", machine.qubits[q].name)
                    wait(t, machine.qubits[q].name)
                    # align() -- no need bc qb-rr share same core and messes up multiplexing
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        demod.full("rotated_cos", I[i], "out1"),
                        demod.full("rotated_sin", Q[i], "out1"),
                    )
                    wait_cooldown_time_fivet1(q, machine, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
                    save(iters, n_st[i])

    with stream_processing():
        for i, q in enumerate(qubit_list):
            I_st[i].buffer(len(lengths)).buffer(len(delays)).average().save(f"I{q}")
            Q_st[i].buffer(len(lengths)).buffer(len(delays)).average().save(f"Q{q}")
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
            if debug:
                plot_demodulated_data_2d(
                    lengths * 4,
                    delays * 4,
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Wait time [ns]",
                    "Injector delay [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=False,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )

    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
