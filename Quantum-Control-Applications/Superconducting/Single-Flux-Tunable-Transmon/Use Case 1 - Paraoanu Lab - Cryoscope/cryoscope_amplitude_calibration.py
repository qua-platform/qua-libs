"""
cryoscope_amplitude_calibration.py: template for performing the detuning vs flux pulse amplitude calibration prior to the cryoscope
protocol.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration_SSB import *
import matplotlib.pyplot as plt
import numpy as np


###################
# The QUA program #
###################

n_avg = 10000  # Number of averages
cooldown_time = 50000 // 4  # Cooldown time between sequences in clock cycles
# Flux amplitude sweep (as a prefactor of the flux amplitude)
n_flux_amp = 401
flux_amp_array = np.linspace(0, -0.2, n_flux_amp)


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


with program() as cryoscope_amp:
    n = declare(int)  # Averaging index
    flag = declare(bool)  # X/2 or Y/2 flag
    flux_amp = declare(fixed)  # Flux amplitude prefactor
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # Calibrate the ground and excited states readout for deriving the Bloch vector
    Ig_st, Qg_st, Ie_st, Qe_st = ge_calibration(n_avg)

    with for_(n, 0, n < n_avg, n + 1):
        with for_each_(flux_amp, flux_amp_array.tolist()):
            with for_each_(flag, [True, False]):
                wait(int(const_len / 4 * 2 + const_flux_len / 4), "resonator")
                # Play first X/2
                play("X90", "qubit")
                # Play truncated flux pulse with scanning amplitude
                align("qubit", "flux_line")
                play("cw" * amp(flux_amp), "flux_line")
                # Play second X/2 or Y/2
                align("qubit", "flux_line")
                with if_(flag):
                    play("X90", "qubit")
                with else_():
                    play("Y90", "qubit")
                # Measure resonator state after the sequence
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

    with stream_processing():
        I_st.buffer(n_flux_amp, 2).average().save("I")
        Q_st.buffer(n_flux_amp, 2).average().save("Q")
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
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, cryoscope_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope_amp)
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
    def on_close(event):
        print("Execution stopped by user!")
        job.halt()
        event.canvas.stop_event_loop()

    fig = plt.figure(figsize=(15, 15))
    fig.canvas.mpl_connect("close_event", on_close)
    xplot = flux_amp_array * const_flux_amp
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
        # qubit_phase = qubit_phase - qubit_phase[-1]
        detuning = qubit_phase / (2 * np.pi * const_flux_len) * 1000
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)
        # Quadratic fit of detuning versus flux pulse amplitude
        pol = np.polyfit(xplot, qubit_phase, deg=2)

        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, np.sqrt(I**2 + Q**2))
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Readout amplitude [a.u.]")
        plt.legend("X", "Y", loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, phase)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Readout phase [rad]")
        plt.legend("X", "Y", loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, pop)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("X and Y")
        plt.legend("X", "Y", loc="lower right")

        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, detuning, "bo")
        plt.plot(xplot, np.polyval(pol, xplot), "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [Hz]")
        plt.legend("data", "Fit", loc="upper right")
        plt.show()
        plt.pause(0.1)
