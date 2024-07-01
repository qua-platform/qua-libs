"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
    - Adjust the readout amplitude, labeled as "readout_amp", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal


###################
# The QUA program #
###################
n_avg = 1000  # The number of averages
# The frequency sweep around the resonator frequency "resonator_IF"
span = 10 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
a_min = 0.001
a_max = 1.99
amplitudes = np.geomspace(a_min, a_max, 20)

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", df + resonator_IF)
            with for_each_(a, amplitudes):
                # Measure the resonator (send a readout pulse whose amplitude is rescaled by the pre-factor 'a' [-2, 2)
                # and demodulate the signals to get the 'I' & 'Q' quadratures)
                measure(
                    "readout" * amp(a),
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
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
        # Note that the buffering goes from the most inner loop (left) to the most outer one (right)
        I_st.buffer(len(amplitudes)).buffer(len(dfs)).average().save("I")
        Q_st.buffer(len(amplitudes)).buffer(len(dfs)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
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
        # Normalize data
        row_sums = R.sum(axis=0)
        R /= row_sums[np.newaxis, :]
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz & IF = {resonator_IF / u.MHz} MHz")
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ (normalized)")
        plt.pcolor(amplitudes * readout_amp, dfs / u.MHz, R)
        plt.xscale("log")
        plt.xlim(amplitudes[0] * readout_amp, amplitudes[-1] * readout_amp)
        plt.ylabel("Readout detuning [MHz]")
        plt.subplot(212)
        plt.cla()
        plt.title("Phase")
        plt.pcolor(amplitudes * readout_amp, dfs / u.MHz, signal.detrend(np.unwrap(phase)))
        plt.ylabel("Readout detuning [MHz]")
        plt.xlabel("Readout amplitude [V]")
        plt.xscale("log")
        plt.xlim(amplitudes[0] * readout_amp, amplitudes[-1] * readout_amp)
        plt.pause(0.1)
        plt.tight_layout()
