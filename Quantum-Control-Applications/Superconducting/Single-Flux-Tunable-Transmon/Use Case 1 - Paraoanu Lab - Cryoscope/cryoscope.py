"""
cryoscope.py: template for performing the cryoscope protocol.
"""

import scipy.io
from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration_SSB import *
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.bakery import baking
from qualang_tools.plot import interrupt_on_close
from scipy import signal

###################
# The QUA program #
###################

n_avg = 10000  # Number of averages
cooldown_time = 50000 // 4 * 0 + 4  # Cooldown time between sequences in clock cycles

# FLux pulse waveform generation
flux_eleph_amp = 0.07
Data = scipy.io.loadmat(
    "C:\\Users\\TheoQM\\Documents\\2_Customers\\6_Paraoanu\\2_Paraoanu_QUA\\OnSite\\cryoscope image\\elephant.mat"
)
flux_waveform = np.transpose(Data["sig2"])
flux_waveform = np.concatenate(flux_waveform)
flux_waveform = flux_waveform / np.max(flux_waveform)
flux_waveform = np.sqrt(flux_waveform)
flux_waveform = flux_waveform * -flux_eleph_amp
flux_waveform = flux_waveform[25:]
for k in range(const_flux_len - len(flux_waveform)):
    flux_waveform = np.append(flux_waveform, -0.25)
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


# Macro for measuring the averaged ground and excited states for calibration
def ge_calibration(n_avg_cal):
    Ical = declare(fixed)
    Qcal = declare(fixed)
    Igcal_st = declare_stream()
    Qgcal_st = declare_stream()
    Iecal_st = declare_stream()
    Qecal_st = declare_stream()
    with for_(n, 0, n < n_avg_cal, n + 1):
        # Ground state calibration
        align("qubit", "resonator")
        measure(
            "short_readout",
            "resonator",
            None,
            dual_demod.full("cos", "sin", Ical),
            dual_demod.full("minus_sin", "cos", Qcal),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(Ical, Igcal_st)
        save(Qcal, Qgcal_st)

        # Excited state calibration
        align("qubit", "resonator")
        play("pi", "qubit")
        align("qubit", "resonator")
        measure(
            "short_readout",
            "resonator",
            None,
            dual_demod.full("cos", "sin", Ical),
            dual_demod.full("minus_sin", "cos", Qcal),
        )
        wait(cooldown_time, "resonator", "qubit")
        save(Ical, Iecal_st)
        save(Qcal, Qecal_st)

        return Igcal_st, Qgcal_st, Iecal_st, Qecal_st


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

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        with for_(segment, 0, segment <= const_flux_len, segment + 10):
            # Alternate between X/2 and Y/2 pulses
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("X90", "qubit")
                # Play truncated flux pulse
                align("qubit", "flux_line")
                with switch_(segment):
                    for j in range(0, const_flux_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Play second X/2 or Y/2
                wait(const_flux_len // 4, "qubit")
                with if_(flag):
                    play("X90", "qubit")
                with else_():
                    play("Y90", "qubit")
                # Measure resonator state after the sequence
                wait(int(const_len * 2 + const_flux_len) // 4, "resonator")
                measure(
                    "short_readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
                )
                # Wait cooldown time and save the results
                wait(cooldown_time, "resonator", "qubit")
                save(I, I_st)
                save(Q, Q_st)

    # Calibrate the ground and excited states readout for deriving the Bloch vector
    Ig_st, Qg_st, Ie_st, Qe_st = ge_calibration(n_avg)

    with stream_processing():
        I_st.buffer(const_flux_len + 1, 2).average().save("I")
        Q_st.buffer(const_flux_len + 1, 2).average().save("Q")
        Ig_st.average().save("Ig")
        Qg_st.average().save("Qg")
        Ie_st.average().save("Ie")
        Qe_st.average().save("Qe")

#####################################
#  Open Communication with the QOP  #
#####################################

qmm = QuantumMachinesManager(host=qop_ip, port=qop_port)

simulation = False
if simulation:
    simulation_config = SimulationConfig(
        duration=3000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, cryoscope, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope)
    res_handles = job.result_handles
    I_handles = res_handles.get("I")
    Q_handles = res_handles.get("Q")
    I_handles.wait_for_values(1)
    Q_handles.wait_for_values(1)
    Ie_handles = res_handles.get("Ie")
    Qe_handles = res_handles.get("Qe")
    Ie_handles.wait_for_values(1)
    Qe_handles.wait_for_values(1)
    Ig_handles = res_handles.get("Ig")
    Qg_handles = res_handles.get("Qg")
    Ig_handles.wait_for_values(1)
    Qg_handles.wait_for_values(1)

    # Live plotting
    fig = plt.figure(figsize=(15, 15))
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    xplot = range(const_flux_len)
    while res_handles.is_processing():
        I = I_handles.fetch_all()
        Q = Q_handles.fetch_all()
        Ie = Ie_handles.fetch_all()
        Qe = Qe_handles.fetch_all()
        Ig = Ig_handles.fetch_all()
        Qg = Qg_handles.fetch_all()

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
        plt.plot(xplot, np.sqrt(I**2 + Q**2)[0])
        plt.plot(xplot, np.sqrt(I**2 + Q**2)[1])
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Readout amplitude [a.u.]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, phase[0])
        plt.plot(xplot, phase[1])
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Readout phase [rad]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, pop[0])
        plt.plot(xplot, pop[1])
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
        plt.show()
        plt.pause(0.1)
