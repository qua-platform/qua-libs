"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures.
This is done across various readout intermediate frequencies and flux biases.
Based on the results, one can determine the resonator frequency as a function of flux bias.

This information can then be used to adjust the readout frequency for the maximum frequency point.

Prerequisites:

    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy_multiplexed").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the configuration.
Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
    - Adjust the flux bias to the maximum frequency point, labeled as "max_frequency_point", in the configuration.
    - Update the resonator frequency versus flux fit parameters (amplitude_fit, frequency_fit, phase_fit, offset_fit) in the configuration
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout
import warnings

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
n_avg = 2000
# The flux bias sweep
flux_pts = 50
dcs = np.linspace(-0.49, 0.49, flux_pts)
# The frequency sweep for both resonators around their respective resonance
dfs = np.arange(-5e6, 5e6, 0.05e6)


with program() as multi_res_spec_vs_flux:
    # QUA macro to declare the measurement variables and their corresponding streams for a given number of resonators
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for sweeping the readout frequency detuning around the resonance
    dc = declare(fixed)  # QUA variable for sweeping the flux bias

    set_dc_offset("q2_z", "single", 0)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two resonator elements
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset("q1_z", "single", dc)
                # set_dc_offset("q2_z", "single", dc)
                wait(flux_settle_time * u.ns)  # Wait for the flux to settle
                # Macro to perform multiplexed readout on the specified resonators
                # It also save the 'I' and 'Q' quadratures into their respective streams
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], sequential=False)
                # wait for the resonators to relax
                wait(depletion_time * u.ns, "rr1", "rr2")
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("Q2")

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
    job = qmm.simulate(config, multi_res_spec_vs_flux, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_res_spec_vs_flux)
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
        s1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        s2 = u.demod2volts(I2 + 1j * Q2, readout_len)

        R1 = np.abs(s1)
        R2 = np.abs(s2)
        # Plots
        plt.suptitle("Resonator spectroscopy")
        plt.subplot(121)
        plt.cla()
        plt.title(f"rr1 (LO: {resonator_LO / u.MHz} MHz)")
        plt.xlabel("Flux (V)")
        plt.ylabel("Freq (MHz)")
        plt.pcolor(dcs, resonator_IF_q1 / u.MHz + dfs / u.MHz, R1)
        plt.subplot(122)
        plt.cla()
        plt.title(f"rr2 (LO: {resonator_LO / u.MHz} MHz)")
        plt.xlabel("Flux (V)")
        plt.ylabel("Freq (MHz)")
        plt.pcolor(dcs, resonator_IF_q2 / u.MHz + dfs / u.MHz, R2)
        plt.tight_layout()
        plt.pause(0.1)

    # Fitting to cosine resonator frequency response
    def cosine_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

    # Array for the flux minima
    minima = np.zeros(len(dcs))
    # Frequency range for the 2 resonators
    frequencies = [dfs + resonator_IF_q1, dfs + resonator_IF_q1]
    # Amplitude for the 2 resonators
    R = [R1, R2]
    plt.figure()
    for rr in range(2):
        print(f"Resonator rr{rr+1}")
        # Find the resonator frequency vs flux minima
        for i in range(len(dcs)):
            minima[i] = frequencies[rr][np.argmin(R[rr].T[i])] / u.MHz

        # Cosine fit
        initial_guess = [1, 0.5, 0, 0]  # Initial guess for the parameters
        fit_params, _ = curve_fit(cosine_func, dcs, minima, p0=initial_guess)

        # Get the fitted values
        amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
        print("fitting parameters", fit_params)

        # Generate the fitted curve using the fitted parameters
        fitted_curve = cosine_func(dcs, amplitude_fit, frequency_fit, phase_fit, offset_fit)
        plt.subplot(2, 1, rr + 1)
        plt.pcolor(dcs, frequencies[rr] / u.MHz, R1)
        plt.plot(dcs, minima, "x-", color="red", label="Flux minima")
        plt.plot(dcs, fitted_curve, label="Fitted Cosine", color="orange")
        plt.xlabel("Flux level [V]")
        plt.ylabel("Readout frequency [MHz]")
        plt.title(f"Resonator rr{rr+1}")
        plt.legend()
        plt.tight_layout()

        print("DC flux value corresponding to the maximum frequency point", dcs[np.argmax(fitted_curve)])
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
