"""
resonator_spec.py: performs the 1D resonator spectroscopy
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

cooldown_time = 10 * u.us // 4

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

    # Get results from QUA program
    res_handles = job.result_handles
    I_handles = res_handles.get("I")
    Q_handles = res_handles.get("Q")
    I_handles.wait_for_values(1)
    Q_handles.wait_for_values(1)
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while job.result_handles.is_processing():
        # Fetch results
        I = res_handles.get("I").fetch_all()
        Q = res_handles.get("Q").fetch_all()
        # Plot results
        plt.subplot(211)
        plt.cla()
        plt.title("resonator spectroscopy amplitude")
        plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
        plt.xlabel("frequency [MHz]")
        plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
        plt.subplot(212)
        plt.cla()
        # detrend removes the linear increase of phase
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        plt.title("resonator spectroscopy phase")
        plt.plot(freqs / u.MHz, phase, ".")
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.pause(0.1)
        plt.tight_layout()

    # Fetch results
    I = res_handles.get("I").fetch_all()
    Q = res_handles.get("Q").fetch_all()
    # Convert I & Q to Volts
    I = u.demod2volts(I, readout_len)
    Q = u.demod2volts(Q, readout_len)
    # 1D spectroscopy plot
    plt.clf()
    plt.subplot(211)
    plt.title("resonator spectroscopy amplitude [V]")
    plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel(r"$\sqrt{I^2 + Q^2}$ [a.u.]")
    plt.subplot(212)
    # detrend removes the linear increase of phase
    phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
    plt.title("resonator spectroscopy phase [rad]")
    plt.plot(freqs / u.MHz, phase, ".")
    plt.xlabel("frequency [MHz]")
    plt.ylabel("Phase [rad]")
    plt.tight_layout()
