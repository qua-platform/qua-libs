"""
ramsey_freq_duration.py: template for acquiring the 2D (idle time & pulse frequency sweeps) Ramsey oscillations
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
n_avg = 1000  # Number of averaging loops

cooldown_time = 5 * qubit_T1 // 4  # Resonator cooldown time in clock cycles (4ns)

# Frequency sweep in Hz (Needs to be a list of int)
freq_span = 1e6
n_freq = 41
freq_array = (np.linspace(-freq_span / 2, freq_span / 2, n_freq) + qubit_IF).astype(int)

# Idle time sweep (Needs to be a list of int)
delay_max = 4e-6 / 1e-9
n_delay = 101
delay_array = np.round(np.linspace(0, delay_max, n_delay) / 4).astype(int)
if len(np.where((delay_array > 0) & (delay_array < 4))[0]) > 0:
    raise Exception("Delay must be either 0 or an integer larger than 4.")
###################
# The QUA program #
###################
with program() as ramsey_freq_duration:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    delay = declare(int)  # Idle time
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(delay, delay_array.tolist()):
            with for_each_(f, freq_array.tolist()):
                # Update the resonator frequency
                update_frequency("qubit", f)
                # Adjust the idle time
                with if_(delay >= 4):
                    play("pi_half", "qubit")
                    wait(delay, "qubit")
                    play("pi_half", "qubit")
                with else_():
                    play("pi_half", "qubit")
                    play("pi_half", "qubit")
                align("qubit", "resonator")
                # Measure the resonator
                measure(
                    "short_readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(cooldown_time, "resonator", "qubit")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(n_delay, n_freq).average().save("I")
        Q_st.buffer(n_delay, n_freq).average().save("Q")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, ramsey_freq_duration, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(ramsey_freq_duration)
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
        plt.pcolor((freq_array - qubit_IF) / 1e6, delay_array * 4, np.sqrt(I**2 + Q**2))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [a.u.]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor((freq_array - qubit_IF) / 1e6, delay_array * 4, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [a.u.]")
        plt.pause(0.1)
