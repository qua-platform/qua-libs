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

u = unit()

###################
# The QUA program #
###################
num_qubits = 2

n_avg = 4e3

cooldown_time = 5 * u.us // 4

f_min = [-70e6, -110e6, -170e6, -210e6]
f_max = [-40e6, -80e6, -120e6, -180e6]
df = 0.05e6

freqs = [np.arange(f_min[i], f_max[i] + 0.1, df) for i in range(num_qubits)]


with program() as resonator_spec:
    n = [declare(int) for _ in range(num_qubits)]
    n_st = [declare_stream() for _ in range(num_qubits)]
    f = declare(int)
    I = [declare(fixed) for _ in range(num_qubits)]
    Q = [declare(fixed) for _ in range(num_qubits)]
    I_st = [declare_stream() for _ in range(num_qubits)]
    Q_st = [declare_stream() for _ in range(num_qubits)]

    for i in range(num_qubits):
        with for_(n[i], 0, n[i] < n_avg, n[i] + 1):
            with for_(
                f, f_min[i], f <= f_max[i], f + df
            ):  # Notice it's <= to include f_max (This is only for integers!)
                update_frequency(f"rr{i}", f)
                measure(
                    "readout",
                    f"rr{i}",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[i]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[i]),
                )
                wait(cooldown_time, f"rr{i}")
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])
            save(n[i], n_st[i])

        align()

    with stream_processing():
        for i in range(num_qubits):
            I_st[i].buffer(len(freqs[i])).average().save(f"I{i}")
            Q_st[i].buffer(len(freqs[i])).average().save(f"Q{i}")
            n_st[i].save(f"iteration{i}")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host="172.16.2.103", port="85")

#######################
# Simulate or execute #
#######################

debug = False
simulate = False
qbts = [0, 1]
rrs = [0, 1]
digital = []
machine = QuAM("quam_bootstrap_state.json")
gate_shape = 'pulse1'
config = machine.build_config(digital, qbts, rrs, gate_shape)

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

    # Initialize dataset
    qubit_data = [{} for _ in range(num_qubits)]
    # Live plotting
    if debug:
        fig = plt.figure()
        interrupt_on_close(fig, job)
    for q in range(num_qubits):
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
                plt.subplot(2, num_qubits, 1 + q)
                plt.cla()
                plt.title(f"resonator spectroscopy qubit {q}")
                plt.plot(freqs[q] / u.MHz, np.sqrt(qubit_data[q]["I"] ** 2 + qubit_data[q]["Q"] ** 2), ".")
                plt.xlabel("frequency [MHz]")
                plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
                plt.subplot(2, num_qubits, num_qubits + 1 + q)
                plt.cla()
                phase = signal.detrend(np.unwrap(np.angle(qubit_data[q]["I"] + 1j * qubit_data[q]["Q"])))
                plt.plot(freqs[q] / u.MHz, phase, ".")
                plt.xlabel("frequency [MHz]")
                plt.ylabel("Phase [rad]")
                plt.pause(0.1)
                plt.tight_layout()
