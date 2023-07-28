from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from qualang_tools.loops import from_array
from scipy.optimize import curve_fit

##############################
# Program-specific variables #
##############################

n_avg = 6000  # Number of averaging loops

cooldown_time = 20 * u.us  # Resonator cooldown time in ns
flux_settle_time = 100 * u.ns // 4  # Flux settle time in ns

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 500 * u.kHz
freqs = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
dc_min = -0.49
dc_max = 0.49
step = 0.01
flux = np.arange(dc_min, dc_max + step / 2, step)  # +da/2 to add a_max to the scan

###################
# The QUA program #
###################

with program() as resonator_spec_2D:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    dc = declare(fixed)  # flux dc level
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            # Update the resonator frequency
            update_frequency("resonator", f)
            with for_(*from_array(dc, flux)):
                # Flux sweeping
                set_dc_offset("flux_line", "single", dc)
                wait(flux_settle_time * u.ns, "resonator", "qubit")
                # Measure the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(cooldown_time * u.ns, "resonator")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(flux)).buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(flux)).buffer(len(freqs)).average().save("Q")
        n_st.save("iteration")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, qop_port, octave=octave_config)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=8000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
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
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.cla()
        plt.title("resonator spectroscopy amplitude")
        plt.pcolor(flux, freqs / u.MHz, np.sqrt(I**2 + Q**2))
        plt.xlabel("flux level [V]")
        plt.ylabel("frequency [MHz]")
        plt.subplot(212)
        plt.cla()
        plt.title("resonator spectroscopy phase")
        plt.pcolor(flux, freqs / u.MHz, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.xlabel("flux level [V]")
        plt.ylabel("frequency [MHz]")
        plt.pause(0.1)
        plt.tight_layout()
    plt.show()

    # Fitting to cosine resonator frequency response
    def cosine_func(x, amplitude, frequency, phase, offset):
        return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset

    Z = I + 1j * Q
    mag = np.abs(Z)
    minima = np.zeros(len(flux))
    for i in range(len(flux)):
        minima[i] = freqs[np.argmin(mag.T[i])] / u.MHz

    initial_guess = [1, 0.5, 0, 0]  # Initial guess for the parameters
    fit_params, _ = curve_fit(cosine_func, flux, minima, p0=initial_guess)

    # Get the fitted values
    amplitude_fit, frequency_fit, phase_fit, offset_fit = fit_params
    print("fitting parameters", fit_params)

    # Generate the fitted curve using the fitted parameters
    fitted_curve = cosine_func(flux, amplitude_fit, frequency_fit, phase_fit, offset_fit)

    plt.figure()
    plt.pcolor(flux, freqs / u.MHz, np.abs(Z))
    plt.plot(flux, minima, "x-", color="red", label="Flux minima")
    plt.plot(flux, fitted_curve, label="Fitted Cosine", color="orange")
    plt.xlabel("flux level [V]")
    plt.ylabel("frequency [MHz]")
    plt.legend()

    print("DC flux value corresponding to the maximum frequency point", flux[np.argmax(fitted_curve)])
