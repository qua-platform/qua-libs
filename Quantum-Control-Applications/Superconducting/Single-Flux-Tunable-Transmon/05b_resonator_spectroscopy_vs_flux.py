"""
        RESONATOR SPECTROSCOPY VERSUS FLUX
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures. This is done across various readout intermediate dfs and flux biases.
The resonator frequency as a function of flux bias is then extracted and fitted so that the parameters can be stored in the configuration.

This information can then be used to adjust the readout frequency for the maximum frequency point.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude and duration.
    - Specification of the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF", in the configuration.
    - Adjust the flux bias to the maximum frequency point, labeled as "max_frequency_point", in the configuration.
    - Update the resonator frequency versus flux fit parameters (amplitude_fit, frequency_fit, phase_fit, offset_fit) in the configuration
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
from scipy.optimize import curve_fit


##############################
# Program-specific variables #
##############################
n_avg = 6000  # Number of averaging loops
# The frequency sweep around the resonator frequency "resonator_IF"
span = 10 * u.MHz
df = 100 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)
# Flux bias sweep in V
flux_min = -0.49
flux_max = 0.49
step = 0.01
flux = np.arange(flux_min, flux_max + step / 2, step)

###################
# The QUA program #
###################

with program() as resonator_spec_2D:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    dc = declare(fixed)  # QUA variable for the flux bias
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, dfs)):
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", f + resonator_IF)
            with for_(*from_array(dc, flux)):
                # Flux sweeping by tuning the OPX dc offset associated to the flux_line element
                set_dc_offset("flux_line", "single", dc)
                wait(flux_settle_time * u.ns, "resonator", "qubit")
                # Measure the resonator (send a readout pulse whose amplitude is rescaled by the pre-factor 'a' [-2, 2)
                # and demodulate the signals to get the 'I' & 'Q' quadratures)
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
                )
                # Wait for the resonator to cooldown
                wait(depletion_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(flux)).buffer(len(dfs)).average().save("I")
        Q_st.buffer(len(flux)).buffer(len(dfs)).average().save("Q")
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
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
        plt.title(r"$R=\sqrt{I^2 + Q^2}$")
        plt.pcolor(flux, (dfs + resonator_IF) / u.MHz, R)
        plt.ylabel("Readout IF [MHz]")
        plt.subplot(212)
        plt.cla()
        plt.title("Phase")
        plt.pcolor(flux, (dfs + resonator_IF) / u.MHz, signal.detrend(np.unwrap(phase)))
        plt.xlabel("Flux bias [V]")
        plt.ylabel("Readout IF [MHz]")
        plt.pause(0.1)
        plt.tight_layout()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Fitting to cosine resonator frequency response
    def cosine_func(x, amplitude, frequency, phi, offset):
        return amplitude * np.cos(2 * np.pi * frequency * x + phi) + offset

    # Find the resonator frequency vs flux minima
    minima = np.zeros(len(flux))
    for i in range(len(flux)):
        minima[i] = dfs[np.argmin(R.T[i])] + resonator_IF
    # Cosine fit
    initial_guess = [1, 1 / 0.4, 0, 0]  # Initial guess for the parameters
    fit_params, _ = curve_fit(cosine_func, flux, minima, p0=initial_guess)
    # Get the fitted values
    amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
    print("fitting parameters", fit_params)
    # Generate the fitted curve using the fitted parameters
    fitted_curve = cosine_func(flux, amplitude_fit, frequency_fit, phase_fit, offset_fit)

    plt.figure()
    plt.suptitle(f"Resonator spectroscopy - LO = {resonator_LO / u.GHz} GHz")
    plt.pcolor(flux, (dfs + resonator_IF) / u.MHz, R)
    plt.plot(flux, minima / u.MHz, "x-", color="red", label="Flux minima")
    plt.plot(flux, fitted_curve / u.MHz, label="Fitted Cosine", color="orange")
    plt.xlabel("Flux bias [V]")
    plt.ylabel("Readout IF [MHz]")
    plt.legend()

    print("DC flux value corresponding to the maximum frequency point", flux[np.argmax(fitted_curve)])
