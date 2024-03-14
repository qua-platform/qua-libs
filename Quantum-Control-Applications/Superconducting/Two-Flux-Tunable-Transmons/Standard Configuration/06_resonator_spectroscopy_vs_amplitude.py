"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures.
This is done across various readout intermediate frequencies and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy_multiplexed").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF_q", in the configuration.
    - Adjust the readout amplitude, labeled as "readout_amp_q", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration
import matplotlib.pyplot as plt
from scipy import signal


###################
# The QUA program #
###################
n_avg = 1000  # The number of averages
# The frequency sweep around the resonators' frequency "resonator_IF_q"
span = 10 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
a_min = 0.001
a_max = 1.99
da = 0.01
amplitudes = np.arange(a_min, a_max + da / 2, da)  # The amplitude vector +da/2 to add a_max to the scan

with program() as multi_res_spec_vs_amp:
    # QUA macro to declare the measurement variables and their corresponding streams for a given number of resonators
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for sweeping the readout frequency detuning around the resonance
    a = declare(fixed)  # QUA variable for sweeping the readout amplitude pre-factor

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the two resonator elements
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the readout amplitude
                # resonator 1
                wait(depletion_time * u.ns, "rr1")  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    "rr1",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[0]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[0]),
                )
                # Save the 'I' & 'Q' quadratures for rr1 to their respective streams
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

                # align("rr1", "rr2")  # Uncomment to measure sequentially

                # resonator 2
                wait(depletion_time * u.ns, "rr2")  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    "rr2",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[1]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[1]),
                )
                # Save the 'I' & 'Q' quadratures for rr2 to their respective streams
                save(I[1], I_st[1])
                save(Q[1], Q_st[1])
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        # Note that the buffering goes from the most inner loop (left) to the most outer one (right)
        # resonator 1
        I_st[0].buffer(len(amplitudes)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(amplitudes)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amplitudes)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(amplitudes)).buffer(len(dfs)).average().save("Q2")

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
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_res_spec_vs_amp)
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Data analysis
        S1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        S2 = u.demod2volts(I2 + 1j * Q2, readout_len)
        R1 = np.abs(S1)
        phase1 = np.angle(S1)
        R2 = np.abs(S2)
        phase2 = np.angle(S2)
        # Normalize data
        row_sums = R1.sum(axis=0)
        R1 /= row_sums[np.newaxis, :]
        row_sums = R2.sum(axis=0)
        R2 /= row_sums[np.newaxis, :]
        # Plot
        plt.suptitle("Resonator spectroscopy")
        plt.subplot(221)
        plt.cla()
        plt.title(f"Resonator 1 - LO: {resonator_LO / u.GHz} GHz")
        plt.ylabel("Readout IF [MHz]")
        plt.pcolor(amplitudes * readout_amp_q1, (dfs + resonator_IF_q1) / u.MHz, R1)
        plt.axhline(resonator_IF_q1 / u.MHz, color="k")
        plt.subplot(222)
        plt.cla()
        plt.title(f"Resonator 2 - LO: {resonator_LO / u.GHz} GHz")
        plt.pcolor(amplitudes * readout_amp_q2, (dfs + resonator_IF_q2) / u.MHz, R2)
        plt.axhline(resonator_IF_q2 / u.MHz, color="k")
        plt.subplot(223)
        plt.cla()
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout IF [MHz]")
        plt.pcolor(amplitudes * readout_amp_q1, (dfs + resonator_IF_q1) / u.MHz, signal.detrend(np.unwrap(phase1)))
        plt.axhline(resonator_IF_q1 / u.MHz, color="k")
        plt.subplot(224)
        plt.cla()
        plt.xlabel("Readout amplitude [V]")
        plt.pcolor(amplitudes * readout_amp_q2, (dfs + resonator_IF_q2) / u.MHz, signal.detrend(np.unwrap(phase2)))
        plt.axhline(resonator_IF_q2 / u.MHz, color="k")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
