"""
Performs a 1D frequency sweep on the resonator
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig

###################
# The QUA program #
###################

n_avg = 100

cooldown_time = 10000 // 4

f_min = 30e6
f_max = 70e6
df = 0.5e6
freqs = np.arange(f_min, f_max + 0.1, df)  # + 0.1 to add f_max to freqs

with program() as resonator_spec:
    n = declare(int)
    f = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            update_frequency("resonator", f)
            measure(
                "long_readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            wait(cooldown_time, "resonator")
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

#######################
# Simulate or execute #
#######################

simulate = True

if simulate:
    simulation_config = SimulationConfig(duration=1000)
    job = qmm.simulate(config, resonator_spec, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec)
    res_handles = job.result_handles
    res_handles.wait_for_all_values()
    I = res_handles.get("I").fetch_all()
    Q = res_handles.get("Q").fetch_all()

    plt.figure()
    plt.title("resonator spectroscopy power")
    plt.plot(freqs, np.sqrt(I**2 + Q**2), ".")
    # plt.plot(freqs + resonator_LO, np.sqrt(I**2 + Q**2), '.')
    plt.xlabel("freq")

    plt.figure()
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title("resonator spectroscopy phase")
    plt.plot(freqs, phase*(180/np.pi), ".")
    # plt.plot(freqs + resonator_LO, np.sqrt(I**2 + Q**2), '.')
    plt.ylabel('phase (degrees)')
    plt.xlabel("freq")
    plt.show()
