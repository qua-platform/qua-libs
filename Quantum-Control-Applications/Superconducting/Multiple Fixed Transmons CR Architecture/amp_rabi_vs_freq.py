"""
amp_rabi_vs_freq.py: performs 2D scans to all qubits of the driving amplitude vs the driving frequency.
The goal of this protocol is to find the resonance frequencies of the qubits.
"""
# todo: axis and axis labels
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

a_min = 0.1
a_max = 1.0
da = 0.4
amps = np.arange(a_min, a_min + da / 2, da)

with program() as rabi_chevron:

    n = declare(int)
    f = declare(int)
    a = declare(fixed)
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
                update_frequency(f"q{q}", f)
                with for_(a, a_min, a < a_max + da / 2, a + da):
                    play("x180" * amp(a), f"q{q}")
                    align()
                    wait(
                        4, f"rr{q}"
                    )  # to prevent simultaneous driving and readout
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
            I_st[idx].buffer(len(freqs), len(amps)).average().save(
                f"I_q{qubits[idx]}"
            )
            Q_st[idx].buffer(len(freqs), len(amps)).average().save(
                f"Q_q{qubits[idx]}"
            )

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
    simulation_config = SimulationConfig(duration=30000)
    job = qmm.simulate(config, rabi_chevron, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi_chevron)

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
    plt.title(f"q{q} amp rabi vs driving freq - amplitude")
    plt.pcolor(np.sqrt(I**2 + Q**2))
    # plt.xlabel("frequency [MHz]")
    # plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plt.subplot(212)
    plt.cla()
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title(f"q{q} amp rabi vs driving freq - phase")
    plt.pcolor(phase)
    # plt.xlabel("frequency [MHz]")
    # plt.ylabel("Phase [rad]")
    plt.tight_layout()
