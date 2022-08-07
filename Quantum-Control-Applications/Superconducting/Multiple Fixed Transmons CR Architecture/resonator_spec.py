"""
resonator_spec.py: performs 1D resonator spectroscopy for multiple qubits
"""

from state_and_config import build_config, state
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.simulate.credentials import create_credentials
from qm.simulate import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.units import unit

###################
# The QUA program #
###################

qubits = [0, 1]
n_avg = 1
u = unit()
cooldown_time = 10 * u.us // 4

f_min = 30e6
f_max = 70e6
df = 30e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

with program() as resonator_spec:
    n = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = [declare_stream() for q in qubits]
    Q_st = [declare_stream() for q in qubits]

    idx = 0
    for q in qubits:
        align()

        with for_(n, 0, n < n_avg, n + 1):
            with for_(
                f, f_min, f <= f_max, f + df
            ):  # Notice it's <= to include f_max (This is only for integers!)
                update_frequency(f"rr{q}", f)
                measure(
                    "readout",
                    f"rr{q}",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                wait(cooldown_time, f"rr{q}")
                save(I, I_st[idx])
                save(Q, Q_st[idx])

        idx += 1

    with stream_processing():
        for idx in range(len(qubits)):
            I_st[idx].buffer(len(freqs)).average().save(f"I_q{qubits[idx]}")
            Q_st[idx].buffer(len(freqs)).average().save(f"Q_q{qubits[idx]}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(
    host="nord-quantique-d14d58b1.quantum-machines.co",
    port=443,
    credentials=create_credentials(),
)

#######################
# Simulate or execute #
#######################

simulate = True
config = build_config(state)

if simulate:
    simulation_config = SimulationConfig(duration=20000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)

# Get results from QUA program
res_handles = job.result_handles
res_handles.wait_for_all_values()

for q in qubits:

    plt.figure()
    I = res_handles.get(f"I_q{q}").fetch_all()
    Q = res_handles.get(f"Q_q{q}").fetch_all()
    # Plot results
    x = 211
    plt.subplot(x)
    plt.cla()
    plt.title(f"rr{q} spectroscopy amplitude")
    plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plt.subplot(212)
    plt.cla()
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title(f"rr{q} spectroscopy phase")
    plt.plot(freqs / u.MHz, phase, ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.tight_layout()



