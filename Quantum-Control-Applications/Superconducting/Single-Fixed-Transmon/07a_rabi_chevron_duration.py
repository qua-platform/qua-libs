"""
        RABI CHEVRON (DURATION VS FREQUENCY)
This sequence involves executing the qubit pulse (be it x180, square_pi, or another type) and measuring the state of
the resonator across various qubit intermediate frequencies and pulse durations.
Analyzing the results allows for determining the qubit and estimating the x180 pulse duration for a specified amplitude.

Prerequisites:
    -Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    -Calibration of the IQ mixer connected to the qubit drive line (whether external mixer or an Octave port).
    -Identification of the approximate qubit frequency ("qubit_spectroscopy").
    -Configuration of the qubit frequency and the desired pi pulse amplitude (labeled as "x180_amp").

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "qubit_IF", in the configuration.
    - Modify the qubit pulse duration setting, labeled as "x180_len", in the configuration.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

##############################
# Program-specific variables #
##############################

n_avg = 1000  # The number of averages
# The frequency sweep parameters
f_min = 30 * u.MHz
f_max = 70 * u.MHz
df = 500 * u.kHz
frequencies = np.arange(f_min, f_max + 0.1, df)  # The frequency vector (+ 0.1 to add f_max to frequencies)
# Pulse duration sweep (in clock cycles = 4ns)
t_min = 4
t_max = 250
dt = 4
durations = np.arange(t_min, t_max, dt)

###################
# The QUA program #
###################

with program() as rabi_amp_freq:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the qubit frequency
    t = declare(int)  # QUA variable for the qubit pulse duration
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(t, durations)):  # QUA for_ loop for sweeping the pulse duration
            with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency
                # Update the frequency of the digital oscillator linked to the qubit element
                update_frequency("qubit", f)
                # Play the qubit pulse with a variable duration (in clock cycles = 4ns)
                play("x180", "qubit", duration=t)
                # Align the two elements to measure after playing the qubit pulse.
                align("qubit", "resonator")
                # Measure the state of the resonator.
                # The integration weights have changed to maximize the SNR after having calibrated the IQ blobs.
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(frequencies)).buffer(len(durations)).average().save("I")
        Q_st.buffer(len(frequencies)).buffer(len(durations)).average().save("Q")
        n_st.save("iteration")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################

simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, rabi_amp_freq, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_amp_freq)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.subplot(211)
        plt.cla()
        plt.title(r"Rabi chevron $R=\sqrt{I^2 + Q^2}$")
        plt.pcolor((frequencies - qubit_IF) / u.MHz, durations * 4, R)
        plt.xlabel("Frequency detuning [MHz]")
        plt.ylabel("Pulse duration [ns]")
        plt.subplot(212)
        plt.cla()
        plt.title("Rabi chevron phase")
        plt.pcolor((frequencies - qubit_IF) / u.MHz, durations * 4, np.unwrap(phase))
        plt.xlabel("Frequency detuning [MHz]")
        plt.ylabel("Pulse duration [ns]")
        plt.tight_layout()
        plt.pause(0.1)
