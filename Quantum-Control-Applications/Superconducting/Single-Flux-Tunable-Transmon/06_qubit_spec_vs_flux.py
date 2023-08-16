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

n_avg = 3000  # Number of averaging loops

cooldown_time = 20 * u.us  # Resonator cooldown time in ns
flux_settle_time = 100 * u.ns  # Flux settle time in ns

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 50 * u.kHz
freqs = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
dc_min = -0.49
dc_max = 0.49
step = 0.01
flux = np.arange(dc_min, dc_max + step / 2, step)  # +da/2 to add a_max to the scan


# Get the resonator frequency vs flux trend from the node 05_resonator_spec_vs_flux.py in order to always measure on
# resonance while sweeping the flux
def cosine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset


amplitude_fit, frequency_fit, phase_fit, offset_fit = [0, 0, 0, 0]
fitted_curve = cosine_func(flux, amplitude_fit, frequency_fit, phase_fit, offset_fit) * u.MHz
fitted_curve = fitted_curve.astype(int)

###################
# The QUA program #
###################

with program() as qubit_spec_2D:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    dc = declare(fixed)  # flux dc level
    I = declare(fixed)
    Q = declare(fixed)
    resonator_freq = declare(int, value=fitted_curve.tolist())  # res freq vs flux table
    index = declare(int, value=0)  # index to get the right resonator freq for a given flux
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, freqs)):
            # Update the qubit frequency
            update_frequency("qubit", f)
            assign(index, 0)
            with for_(*from_array(dc, flux)):
                # Update the resonator frequency to always measure on resonance
                update_frequency("resonator", resonator_freq[index] + resonator_IF)
                # Flux sweeping
                set_dc_offset("flux_line", "single", dc)
                wait(flux_settle_time * u.ns, "resonator", "qubit")
                # Play a saturation pulse on the qubit
                play("cw", "qubit")
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
                wait(cooldown_time * u.ns, "resonator")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)
                assign(index, index + 1)
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
    job = qmm.simulate(config, qubit_spec_2D, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(qubit_spec_2D)
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
        plt.title("qubit spectroscopy amplitude")
        plt.pcolor(flux, freqs / u.MHz, np.sqrt(I**2 + Q**2))
        plt.ylabel("qubit frequency [MHz]")
        plt.xlabel("flux level [V]")
        plt.subplot(212)
        plt.cla()
        plt.title("qubit spectroscopy phase")
        plt.pcolor(flux, freqs / u.MHz, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.ylabel("qubit frequency [MHz]")
        plt.xlabel("flux level [V]")
        plt.pause(0.1)
        plt.tight_layout()
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
