"""
cryoscope.py: template for performing the cryoscope protocol.
"""
import scipy.io
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import ge_averaged_measurement
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery import baking
from scipy import signal

###################
# The QUA program #
###################

n_avg = 10000  # Number of averages
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)

# FLux pulse waveform generation
flux_amp = -0.1
flux_waveform = np.array([flux_amp] * const_flux_len)
# signal.triang(const_flux_len)
# np.cos(2 * np.pi * 10e6 * np.arange(0,const_flux_len)*1e-9)**2


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", "flux_line", wf)
            b.play("flux_pulse", "flux_line")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, const_flux_len)

# Expected averaged frequency based on the detuning versus flux pulse amplitude calibration
poly_fit = [2.44840253e04, 2.96184426e02, -1.72597554e-02]
theory = np.polyval(poly_fit, flux_waveform)

with program() as cryoscope:
    n = declare(int)  # Average index
    segment = declare(int)  # FLux pulse segment index
    flag = declare(bool)  # X/2 or Y/2 flag
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()
    n_st = declare_stream()

    # Calibrate the ground and excited states readout for deriving the Bloch vector
    Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(cooldown_time, n_avg)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        with for_(segment, 0, segment <= const_flux_len, segment + 1):
            # Alternate between X/2 and Y/2 pulses
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("x90", "qubit")
                # Play truncated flux pulse
                align("qubit", "flux_line")
                with switch_(segment):
                    for j in range(0, const_flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Play second X/2 or Y/2
                wait(const_flux_len // 4, "qubit")
                with if_(flag):
                    play("x90", "qubit")
                with else_():
                    play("y90", "qubit")
                # Measure resonator state after the sequence
                wait(int(const_len * 2 + const_flux_len) // 4, "resonator")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q),
                )
                # Wait cooldown time and save the results
                wait(cooldown_time, "resonator", "qubit")
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        I_st.buffer(const_flux_len + 1, 2).average().save("I")
        Q_st.buffer(const_flux_len + 1, 2).average().save("Q")
        Ig_st.average().save("Ig")
        Qg_st.average().save("Qg")
        Ie_st.average().save("Ie")
        Qe_st.average().save("Qe")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=30000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, cryoscope, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    xplot = range(const_flux_len + 1)
    while results.is_processing():
        # Fetch results
        I, Q, Ie, Qe, Ig, Qg, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Phase of ground and excited states
        phase_g = np.angle(Ig + 1j * Qg)
        phase_e = np.angle(Ie + 1j * Qe)
        # Phase of cryoscope measurement
        phase = np.unwrap(np.angle(I + 1j * Q))
        # Population in excited state
        pop = (phase - phase_g) / (phase_e - phase_g)
        # Bloch vector Sx + iSy
        qubit_state = (pop[:, 0] * 2 - 1) + 1j * (pop[:, 1] * 2 - 1)
        # Accumulated phase: angle between Sx and Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[-1]
        # Filtering and derivative of the phase to get the averaged frequency
        d_qubit_phase = scipy.signal.savgol_filter(qubit_phase / 2 / np.pi, 13, 3, deriv=1, delta=0.001)
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)

        # Plots
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, np.sqrt(I**2 + Q**2))
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Readout amplitude [a.u.]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, phase)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Readout phase [rad]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, pop)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, d_qubit_phase, "b.")
        plt.plot(xplot, theory, "r--", lw=3)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.legend(("exp", "theory"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)
