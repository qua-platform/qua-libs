"""
cryoscope_amplitude_calibration.py: template for performing the detuning vs flux pulse amplitude calibration prior to Cryoscope
protocol.
"""
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig, LoopbackInterface
from configuration import *
from macros import ge_averaged_measurement
import matplotlib.pyplot as plt
import numpy as np
from qualang_tools.loops import from_array


###################
# The QUA program #
###################

n_avg = 10000  # Number of averages
cooldown_time = 5 * qubit_T1 // 4  # Cooldown time in clock cycles (4ns)
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
n_flux_amp = 401
flux_amp_array = np.linspace(0, -0.2, n_flux_amp)

with program() as cryoscope_amp:
    n = declare(int)  # Averaging index
    flag = declare(bool)  # X/2 or Y/2 flag
    flux_amp = declare(fixed)  # Flux amplitude pre-factor
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    # Calibrate the ground and excited states' readout for deriving the Bloch vector
    Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(cooldown_time, n_avg)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(flux_amp, flux_amp_array)):
            with for_each_(flag, [True, False]):
                wait(int(const_len / 4 * 2 + const_flux_len / 4), "resonator")
                # Play first X/2
                play("x90", "qubit")
                # Play truncated flux pulse with scanning amplitude
                align("qubit", "flux_line")
                play("const" * amp(flux_amp), "flux_line")
                # Play second X/2 or Y/2
                align("qubit", "flux_line")
                with if_(flag):
                    play("x90", "qubit")
                with else_():
                    play("y90", "qubit")
                # Measure resonator state after the sequence
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

    with stream_processing():
        I_st.buffer(2).buffer(n_flux_amp).average().save("I")
        Q_st.buffer(2).buffer(n_flux_amp).average().save("Q")
        Ig_st.average().save("Ig")
        Qg_st.average().save("Qg")
        Ie_st.average().save("Ie")
        Qe_st.average().save("Qe")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip)

simulation = True
if simulation:
    simulation_config = SimulationConfig(
        duration=28000, simulation_interface=LoopbackInterface([("con1", 3, "con1", 1)])
    )
    job = qmm.simulate(config, cryoscope_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope_amp)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg"], mode="live")

    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    xplot = flux_amp_array * const_flux_amp
    while results.is_processing():
        # Fetch results
        I, Q, Ie, Qe, Ig, Qg = results.fetch_all()

        # Phase of ground and excited states
        phase_g = np.angle(Ig + 1j * Qg)
        phase_e = np.angle(Ie + 1j * Qe)
        # Phase of Cryoscope measurement
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
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, phase)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Readout phase [rad]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, pop)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, detuning, "bo")
        plt.plot(xplot, np.polyval(pol, xplot), "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [Hz]")
        plt.legend(("data", "Fit"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)
