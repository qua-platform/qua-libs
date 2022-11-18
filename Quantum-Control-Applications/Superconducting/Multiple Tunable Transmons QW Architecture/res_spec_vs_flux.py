"""
resonator_spec.py: performs the 2D resonator spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close, plot_demodulated_data_2d
from qualang_tools.results import progress_counter, fetching_tool
from macros import qua_declaration
from qualang_tools.loops import from_array


##################
# State and QuAM #
##################
debug = True
simulate = False
qubit_list = [0, 1]
digital = []
machine = QuAM("quam_bootstrap_state.json")
gate_shape = "drag_cosine"

machine.readout_lines[0].lo_freq = 6.5e9
machine.readout_resonators[0].f_res = 6.6457e9
machine.readout_resonators[1].f_res = 6.7057e9

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 1e3
cooldown_time = 5 * u.us // 4

# Frequency scan
freq_span = 25e6
df = 0.5e6
freq = [
    np.arange(machine.get_readout_IF(i) - freq_span, machine.get_readout_IF(i) + freq_span + df / 2, df)
    for i in qubit_list
]
# Bias scan
bias_span = 0.3
dbias = 0.05
bias = [
    np.arange(
        machine.get_flux_bias_point(i, "zero_frequency_point").value - bias_span,
        machine.get_flux_bias_point(i, "zero_frequency_point").value + bias_span + dbias / 2,
        dbias,
    )
    for i in range(len(qubit_list))
]

# QUA program
with program() as resonator_spec:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(qubit_list)
    f = declare(int)
    b = declare(fixed)

    for i in range(len(qubit_list)):
        # bring other qubits than `i` to zero frequency
        machine.nullify_qubits(True, qubit_list, i)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(b, bias[i])):
                set_dc_offset(machine.qubits[i].name + "_flux", "single", b)
                wait(250)  # wait for 1 us
                with for_(*from_array(f, freq[i])):
                    update_frequency(machine.readout_resonators[i].name, f)
                    measure(
                        "readout",
                        machine.readout_resonators[i].name,
                        None,
                        dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                        dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                    )
                    wait(cooldown_time, machine.readout_resonators[i].name)
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(len(qubit_list)):
            I_st[i].buffer(len(freq[i])).buffer(len(bias[i])).average().save(f"I{i}")
            Q_st[i].buffer(len(freq[i])).buffer(len(bias[i])).average().save(f"Q{i}")
            n_st[i].save(f"iteration{i}")

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
    # Live plotting
    # if debug:
    #     fig = [plt.figure() for q in range(len(qubit_list))]

    for q in range(len(qubit_list)):
        fig = plt.figure()
        interrupt_on_close(fig, job)
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
                plot_demodulated_data_2d(
                    freq[q] / u.MHz,
                    bias[q],
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "frequency [MHz]",
                    "Flux bias [V]",
                    f"resonator spectroscopy qubit {q}",
                    amp_and_phase=True,
                    plot_options={"cmap": "magma"},
                    fig=fig,
                )

    # need to update quam with important flux bias points in the console
