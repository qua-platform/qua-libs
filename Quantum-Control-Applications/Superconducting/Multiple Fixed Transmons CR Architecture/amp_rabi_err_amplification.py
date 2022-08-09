"""
amp_spec.py: performs the 1D amp rabi for multiple qubits.
"""
# todo: fitting
import matplotlib
matplotlib.use('TKAgg')
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
from analysis_utils import _fit

###################
# The QUA program #
###################

qubits = [0, 1]
n_avg = 1
u = unit()
cooldown_time = 10 * u.us // 4

a_min = 0.1
a_max = 1.0
da = 0.4
amps = np.arange(a_min, a_min + da / 2, da)

repeated_pulses = 3

with program() as amp_rabi:
    n = declare(int)
    m = declare(int)
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
                a, a_min, a < a_max + da / 2, a + da
            ):  # Notice it's <= to include f_max (This is only for integers!)
                with for_(m, 0, m < repeated_pulses, m + 1):
                    play("x180" * amp(a), f"q{q}")
                align()
                wait(4, f"rr{q}")  # to prevent simultaneous driving and readout
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
            I_st[idx].buffer(len(amps)).average().save(f"I_q{qubits[idx]}")
            Q_st[idx].buffer(len(amps)).average().save(f"Q_q{qubits[idx]}")

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
    simulation_config = SimulationConfig(duration=10000)
    job = qmm.simulate(config, amp_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(amp_rabi)

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
    plt.title(
        f"q{q} amp rabi, X{repeated_pulses} ERR amplification - I"
    )
    plt.plot(amps, I, ".")
    plt.xlabel("amplitude [a.u]")
    plt.ylabel(r"I [a.u.]")
    plt.subplot(212)
    plt.cla()
    # detrend removes the linear increase of phase
    plt.title(f"q{q} amp rabi, X{repeated_pulses} ERR amplification - Q")
    plt.plot(amps, Q, ".")
    plt.xlabel("amplitude [a.u]")
    plt.ylabel("I [a.u]")
    plt.tight_layout()


# ======= analysis ======= #

############ fit ############
# plt.figure()
# plt.plot(a, I, '.')
# out = _fit(a, I)
# n = 2  # peak number
# print(out["f"])
# peak_location = (n - out["phase"] / (2 * np.pi)) / out["f"]
# plt.plot(peak_location, out["fit_func"](peak_location), "og")
# print(peak_location)

#### manually picking peak ####
# plt.figure()
# a_peaked = pick_sample(a, I)
# print(a_peaked)