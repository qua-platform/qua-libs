"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
The sequence consists in measuring the resonator (send a readout pulse and demodulate the signals to extract the 'I'
and 'Q' quadratures) for different readout intermediate frequencies and amplitudes.
From the results, one can check if a qubit is coupled to the resonator by observing the resonator frequency splitting
and adjust the readout amplitude to sit right before the splitting.

Prerequisites:
    - Having calibrated the time of flight, offsets and gains (time_of_flight).
    - Having calibrated the IQ mixer connected to the readout line (external mixer or Octave port).
    - Having found the resonance frequency of the resonator (resonator_spectroscopy).
    - Set the readout pulse amplitude (to 0.25V) and duration in the configuration.
    - Set the expected resonator depletion time in the configuration.

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF) in the configuration
    - Update the readout amplitude (readout_amp) in the configuration
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qm import SimulationConfig
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

n_avg = 1000  # The number of averages

# The frequency sweep
f_min = 50 * u.MHz
f_max = 52 * u.MHz
df = 50 * u.kHz
frequencies = np.arange(f_min, f_max + 0.1, df)  # The frequency vector (+ 0.1 to add f_max to frequencies)
# The readout amplitude sweep (as a pre-factor of the readout amplitude)
a_min = 0.001
a_max = 1.99
da = 0.01
amplitudes = np.arange(a_min, a_max + da / 2, da)  # The amplitude vector +da/2 to add a_max to the scan

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies)):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", f)
            with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the readout amplitude
                # Measure the resonator (send a readout pulse whose amplitude is rescaled by the pre-factor 'a' [-2, 2)
                # and demodulate the signals to get the 'I' & 'Q' quadratures)
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to deplete
                wait(depletion_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(amplitudes)).buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(frequencies)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, resonator_spec_2D, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    qm = qmm.open_qm(config)
    job = qm.execute(resonator_spec_2D)
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
        # Convert results into Volts and normalize
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        row_sums = R.sum(axis=0)
        R = R / row_sums[np.newaxis, :]
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.title(r"resonator spectroscopy $R=\sqrt{I^2 + Q^2}$ (normalized)")
        plt.pcolor(amplitudes * readout_amp, frequencies / u.MHz, R)
        plt.ylabel("Frequency [MHz]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor(amplitudes * readout_amp, frequencies / u.MHz, signal.detrend(np.unwrap(phase)))
        plt.ylabel("Frequency [MHz]")
        plt.xlabel("Readout amplitude [V]")
        plt.pause(0.1)
        plt.tight_layout()
