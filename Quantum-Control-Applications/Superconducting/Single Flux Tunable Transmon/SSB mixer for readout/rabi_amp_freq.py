"""
rabi_amp_freq.py: template for acquiring the 2D (pulse amplitude & frequency sweeps) Rabi oscillations
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

##############################
# Program-specific variables #
##############################
n_avg = 200  # Number of averaging loops

cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)

# Frequency sweep in Hz (Needs to be a list of int)
freq_span = 20e6
n_freq = 41
freq_array = (np.linspace(-freq_span / 2, freq_span / 2, n_freq) + qubit_IF).astype(int)

# Pulse amplitude sweep (as a prefactor of the flux amplitude)
a_min = 0
a_max = 1.99
n_a = 161
a_array = np.linspace(a_min, a_max, n_a)

###################
# The QUA program #
###################

with program() as rabi_amp_freq:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    a = declare(fixed)  # Pulse amplitude
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(a, a_array.tolist()):
            with for_each_(f, freq_array.tolist()):
                # Update the resonator frequency
                update_frequency("qubit", f)
                # Adjust the pulse amplitude
                play("pi" * amp(a), "qubit")
                align("qubit", "resonator")
                # Measure the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(cooldown_time, "resonator")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(n_a, n_freq).average().save("I")
        Q_st.buffer(n_a, n_freq).average().save("Q")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(rabi_amp_freq)
    res_handles = job.result_handles
    I_handles = res_handles.get("I")
    Q_handles = res_handles.get("Q")
    I_handles.wait_for_values(1)
    Q_handles.wait_for_values(1)

    # Live plotting
    fig = plt.figure(figsize=(15, 15))
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while res_handles.is_processing():

        I = I_handles.fetch_all()
        Q = Q_handles.fetch_all()

        plt.subplot(211)
        plt.cla()
        plt.title("resonator spectroscopy power")
        plt.pcolor((freq_array - qubit_IF) / 1e6, a_array * pi_amp, np.sqrt(I**2 + Q**2))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [a.u.]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor((freq_array - qubit_IF) / 1e6, a_array * pi_amp, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [a.u.]")
        plt.show()
        plt.pause(0.01)
