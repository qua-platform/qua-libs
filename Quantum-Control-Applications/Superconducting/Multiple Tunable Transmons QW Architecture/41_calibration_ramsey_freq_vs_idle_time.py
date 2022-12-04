"""
2D Ramsey frequency versus dephasing time
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


##################
# State and QuAM #
##################
experiment = "2D_ramsey_freq_vs_idle_time"
debug = True
simulate = False
fit_data = True
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"

config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
n_avg = 4e2

# Frequency scan
freq_span = 100e6
df = 1e6
freq = [np.arange(-freq_span, 0 + freq_span + df / 2, df) for i in qubit_list]
# Dephasing time scan
tau_min = 4  # in clock cycles
tau_max = 1000  # in clock cycles
d_tau = 20  # in clock cycles

taus = np.arange(tau_min, tau_max + 0.1, d_tau)  # + 0.1 to add tau_max to taus


# QUA program
with program() as ramsey:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)
    c = declare(fixed, value=1e-9)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    tau = declare(int)

    for q in range(len(qubit_list)):
        if not simulate:
            cooldown_time = 5 * machine.qubits[q].t1 // 4
        else:
            cooldown_time = 16
        # bring other qubits to zero frequency
        machine.nullify_other_qubits(qubit_list, q)
        set_dc_offset(
            machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "zero_frequency_point").value
        )

        with for_(n[q], 0, n[q] < n_avg, n[q] + 1):
            with for_(*from_array(f, freq[q])):
                update_frequency(machine.qubits[q].name, f)
                with for_(*from_array(tau, taus)):
                    play("x90", machine.qubits[q].name)
                    wait(tau, machine.qubits[q].name)
                    play("x90", machine.qubits[q].name)
                    align()
                    measure(
                        "readout",
                        machine.readout_resonators[q].name,
                        None,
                        dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I[q]),
                        dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q[q]),
                    )
                    wait(cooldown_time, machine.readout_resonators[q].name)
                    save(I[q], I_st[q])
                    save(Q[q], Q_st[q])
            save(n[q], n_st[q])

        align()

    with stream_processing():
        for q in range(len(qubit_list)):
            I_st[q].buffer(len(taus)).buffer(len(freq[q])).average().save(f"I{q}")
            Q_st[q].buffer(len(taus)).buffer(len(freq[q])).average().save(f"Q{q}")
            n_st[q].save(f"iteration{q}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)

#######################
# Simulate or execute #
#######################
if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey)

    # Initialize dataset
    qubit_data = [{} for _ in range(len(qubit_list))]
    figures = []
    for q in range(len(qubit_list)):
        print("Qubit " + str(q))
        qubit_data[q]["iteration"] = 0
        # Live plotting
        if debug:
            fig = plt.figure()
            interrupt_on_close(fig, job)
            figures.append(fig)
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
                    taus * 4,
                    freq[q] + machine.drive_lines[0].lo_freq,
                    qubit_data[q]["I"],
                    qubit_data[q]["Q"],
                    "Microwave drive frequency [Hz]",
                    "Dephasing time [ns]",
                    f"{experiment} qubit {q}",
                    amp_and_phase=True,
                    fig=fig,
                    plot_options={"cmap": "magma"},
                )
    machine.save_results(experiment, figures)
    # machine.save("latest_quam.json")
