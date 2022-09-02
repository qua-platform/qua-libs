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
from qualang_tools.loops import from_array

##############################
# Program-specific variables #
##############################
n_avg = 1000  # Number of averaging loops

cooldown_time = 5 * qubit_T1 // 4  # Resonator cooldown time in clock cycles (4ns)

# Frequency sweep in Hz (Needs to be a list of int)
freq_span = 10 * u.MHz
n_freq = 41
freq_array = (np.linspace(-freq_span / 2, freq_span / 2, n_freq) + qubit_IF).astype(int)

# Idle time sweep (Needs to be a list of int)
delay_max = 4 * u.us
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
    n_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(delay, delay_array)):
            with for_(*from_array(f, freq_array)):
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
                    "readout",
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
        save(n, n_st)

    with stream_processing():
        I_st.buffer(n_freq).buffer(n_delay).average().save("I")
        Q_st.buffer(n_freq).buffer(n_delay).average().save("Q")
        n_st.save("iteration")


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
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.subplot(211)
        plt.cla()
        plt.title("resonator spectroscopy amplitude")
        plt.pcolor((freq_array - qubit_IF) / u.MHz, delay_array * 4, np.sqrt(I**2 + Q**2))
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor((freq_array - qubit_IF) / u.MHz, delay_array * 4, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.xlabel("frequency [MHz]")
        plt.ylabel("Idle time [ns]")
        plt.tight_layout()
        plt.pause(0.01)
