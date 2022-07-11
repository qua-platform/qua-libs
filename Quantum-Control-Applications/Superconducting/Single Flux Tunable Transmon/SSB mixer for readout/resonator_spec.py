"""
resonator_spec.py: performs the 1D and 2D (with flux amplitude sweep) resonator spectroscopy
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

n_avg = 30  # Number of averaging loops

cooldown_time = u.to_clock_cycles(2 * u.us)  # Resonator cooldown time in clock cycles (4ns)
flux_settle_time = u.to_clock_cycles(4 * u.us)  # Flux settle time in clock cycles (4ns)

# Frequency sweep in Hz
f_min = 55 * u.MHz
f_max = 65 * u.MHz
df = 50 * u.kHz
freqs = np.arange(f_min, f_max + df / 2, df)  # +df/2 to add f_max to the scan
# Flux amplitude sweep (as a prefactor of the flux amplitude)
a_min = -1
a_max = 0
da = 0.01
flux = np.arange(a_min, a_max + da / 2, da)  # +da/2 to add a_max to the scan

###################
# The QUA program #
###################

with program() as resonator_spec_1D:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(f, f_min, f <= f_max, f + df):  # Notice it's <= to include f_max (This is only for integers!)
            # Adjust the flux line if needed
            # play("const" * amp(0), "flux_line")
            wait(flux_settle_time, "resonator")
            # Update the resonator frequency
            update_frequency("resonator", f)
            # Measure the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("cos", "out1", "sin", "out2", I),
                dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
            )
            # Wait for the resonator to cooldown
            wait(cooldown_time, "resonator", "flux_line")
            # Save data to the stream processing
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).average().save("I")
        Q_st.buffer(len(freqs)).average().save("Q")

with program() as resonator_spec_2D:
    n = declare(int)  # Averaging index
    f = declare(int)  # Resonator frequency
    a = declare(fixed)  # Flux amplitude prefactor
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(a, a_min, a < a_max + da / 2, a + da):  # Notice it's < a_max + da/2 to include a_max
            with for_(*from_array(f, freqs)):
                # Update the resonator frequency
                update_frequency("resonator", f)
                # Adjust the flux line
                play("const" * amp(a), "flux_line")
                wait(flux_settle_time, "resonator", "qubit")
                # Measure the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait for the resonator to cooldown
                wait(cooldown_time, "resonator")
                # Save data to the stream processing
                save(I, I_st)
                save(Q, Q_st)

    with stream_processing():
        I_st.buffer(len(freqs)).buffer(len(flux)).average().save("I")
        Q_st.buffer(len(freqs)).buffer(len(flux)).average().save("Q")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
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
    results = fetching_tool(job, data_list=["I", "Q"], mode="live")
    # Live plotting
    fig = plt.figure(figsize=(15, 15))
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while job.result_handles.is_processing():
        try:
            # Fetch results
            I, Q = results.fetch_all()
            # 1D spectroscopy plot
            if len(I.shape) == 1:
                plt.subplot(211)
                plt.cla()
                plt.title("resonator spectroscopy amplitude")
                plt.plot(freqs / u.MHz, np.sqrt(I**2 + Q**2), ".")
                plt.xlabel("freq [MHz]")
                plt.subplot(212)
                plt.cla()
                # detrend removes the linear increase of phase
                phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
                plt.title("resonator spectroscopy phase")
                plt.plot(freqs / u.MHz, phase, ".")
                plt.xlabel("freq [MHz]")
            # 2D spectroscopy plot
            elif len(I.shape) == 2:
                plt.subplot(211)
                plt.cla()
                plt.title("resonator spectroscopy amplitude")
                plt.pcolor(freqs / u.MHz, flux * const_flux_amp, np.sqrt(I**2 + Q**2))
                plt.xlabel("freq [MHz]")
                plt.ylabel("flux amplitude [V]")
                plt.subplot(212)
                plt.cla()
                plt.title("resonator spectroscopy phase")
                plt.pcolor(freqs / u.MHz, flux * const_flux_amp, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
                plt.xlabel("freq [MHz]")
                plt.ylabel("flux amplitude [V]")
                plt.pause(0.1)
        except (Exception,):
            pass

    I, Q = results.fetch_all()
    # Convert I & Q to Volts
    I = u.demod2volts(I, readout_len)
    Q = u.demod2volts(Q, readout_len)
    # 1D spectroscopy plot
    plt.clf()
    if len(I.shape) == 1:
        plt.subplot(211)
        plt.title("resonator spectroscopy amplitude")
        plt.plot(freqs / u.MHz, np.sqrt(I ** 2 + Q ** 2), ".")
        plt.xlabel("freq [MHz]")
        plt.subplot(212)
        # detrend removes the linear increase of phase
        phase = signal.detrend(np.unwrap(np.angle(I + 1j * Q)))
        plt.title("resonator spectroscopy phase")
        plt.plot(freqs / u.MHz, phase, ".")
        plt.xlabel("freq [MHz]")
    # 2D spectroscopy plot
    elif len(I.shape) == 2:
        plt.subplot(211)
        plt.title("resonator spectroscopy amplitude")
        plt.pcolor(freqs / u.MHz, flux * const_flux_amp, np.sqrt(I ** 2 + Q ** 2))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [V]")
        plt.subplot(212)
        plt.title("resonator spectroscopy phase")
        plt.pcolor(freqs / u.MHz, flux * const_flux_amp, signal.detrend(np.unwrap(np.angle(I + 1j * Q))))
        plt.xlabel("freq [MHz]")
        plt.ylabel("flux amplitude [V]")
        plt.pause(0.1)