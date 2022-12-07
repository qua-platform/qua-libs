"""
Perform the 2D resonator spectroscopy frequency vs flux
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from macros import *
from qualang_tools.loops import from_array

##################
# State and QuAM #
##################
experiment = "2D_resonator_spectroscopy_vs_flux"
debug = True
simulate = False
qubit_list = [1,0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 1e3
# Frequency scan
freq_span = 5e6
df = 0.1e6
freq = [
    np.arange(machine.get_readout_IF(i) - freq_span, machine.get_readout_IF(i) + freq_span + df / 2, df)
    for i in qubit_list
]
# Bias scan
bias_span = 0.5
dbias = 0.05
bias = [
    np.arange(
        machine.get_flux_bias_point(i, "zero_frequency_point").value - bias_span,
        machine.get_flux_bias_point(i, "zero_frequency_point").value+ bias_span + dbias / 2,
        dbias,
    )
    for i in range(len(qubit_list))
]

# QUA program
with program() as resonator_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    b = declare(fixed)

    for i,q in enumerate(qubit_list):

        # bring other qubits than `q` to zero frequency
        machine.nullify_other_qubits(qubit_list, q)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(b, bias[i])):
                set_dc_offset(machine.qubits[q].name + "_flux", "single", b)
                wait(250)  # wait for 1 us
                with for_(*from_array(f, freq[i])):
                    update_frequency(machine.readout_resonators[q].name, f)
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                    )
                    wait_cooldown_time(machine.readout_resonators[q].relaxation_time, simulate)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i,q in enumerate(qubit_list):
            I_st[i].buffer(len(freq[i])).buffer(len(bias[i])).average().save(f"I{q}")
            Q_st[i].buffer(len(freq[i])).buffer(len(bias[i])).average().save(f"Q{q}")
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
    exit = False
    # Live plotting
    figures = []

    for i,q in enumerate(qubit_list):
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
                    (freq[i] + machine.readout_lines[machine.readout_resonators[q].wiring.readout_line_index].lo_freq)*1e-6,
                    bias[i],
                    qubit_data[i]["I"],
                    qubit_data[i]["Q"],
                    "Readout frequency [MHz]",
                    "Flux bias [V]",
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

    # need to update quam with important flux bias points in the console
    # machine.get_flux_bias_point(0, "zero_frequency_point").value = 0.115
    # And choose three points on the resonator frequency vs flux parabola to fit it and get the f_res vs flux correspondence
    # machine.set_f_res_vs_flux_vertex(0, [(-0.1, 4.46e9), (0, 4.45e9), (0.1, 4.465e9)])
    # machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
