"""
resonator_spec.py: performs the 1D resonator spectroscopy
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from quam import QuAM
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.units import unit
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter, fetching_tool
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
config = machine.build_config(digital, qubit_list, gate_shape)

###################
# The QUA program #
###################
u = unit()

n_avg = 4e2
cooldown_time = 5 * u.us // 4

# Frequency scan
freq_span = 25e6
df = 0.5e6
freq = [
    np.arange(machine.get_qubit_IF(i) - freq_span, machine.get_qubit_IF(i) + freq_span + df / 2, df) for i in qubit_list
]
# Bias scan
bias_span = 0.3
dbias = 0.05
bias = [
    np.arange(
        machine.get_flux_bias_point(i, "flux_zero_frequency_point").value - bias_span,
        machine.get_flux_bias_point(i, "flux_zero_frequency_point").value + bias_span + dbias / 2,
        dbias,
    )
    for i in range(len(qubit_list))
]
# QUA program
with program() as resonator_spec:
    n = [declare(int) for _ in range(len(qubit_list))]
    n_st = [declare_stream() for _ in range(len(qubit_list))]
    f = declare(int)
    I = [declare(fixed) for _ in range(len(qubit_list))]
    Q = [declare(fixed) for _ in range(len(qubit_list))]
    I_st = [declare_stream() for _ in range(len(qubit_list))]
    Q_st = [declare_stream() for _ in range(len(qubit_list))]
    b = declare(fixed)

    for i in range(len(qubit_list)):
        # bring other qubits to zero frequency
        machine.nullify_qubits(True, qubit_list, i)

        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(*from_array(b, bias[i])):
                set_dc_offset(machine.qubits[i].name + "_flux", "single", b)
                wait(250, machine.qubits[i].name)  # wait for 1 us
                with for_(*from_array(f, freq[i])):
                    update_frequency(machine.qubits[i].name, f)
                    play("const", machine.qubits[i].name)
                    align()
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
            I_st[i].buffer(len(freq)).buffer(len(bias[i])).average().save(f"I{i}")
            Q_st[i].buffer(len(freq)).buffer(len(bias[i])).average().save(f"Q{i}")
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
    if debug:
        fig = plt.figure()
        interrupt_on_close(fig, job)
    for q in range(len(qubit_list)):
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
                plt.subplot(2, len(qubit_list), 1 + q)
                plt.cla()
                plt.title(f"Qubit spectroscopy qubit {q}")
                plt.pcolor(
                    freq[q],
                    bias[q],
                    np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2),
                )
                plt.xlabel("Microwave drive amplitude [V]")
                plt.ylabel("Bias amplitude [V]")
                cbar = plt.colorbar()
                cbar.ax.set_ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]", rotation=270, labelpad=20)
                plt.subplot(2, len(qubit_list), len(qubit_list) + 1 + q)
                plt.cla()
                phase = signal.detrend(np.unwrap(np.angle(qubit_data[q]["I"] + 1j * qubit_data[q]["Q"])))
                plt.pcolor(freq[q], bias[q], phase)
                plt.xlabel("Microwave drive amplitude [V]")
                plt.ylabel("Bias amplitude [V]")
                cbar = plt.colorbar()
                cbar.set_label("Phase [rad]", rotation=270, labelpad=20)
                plt.pause(0.1)
                plt.tight_layout()
