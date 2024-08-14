"""
        CRYOSCOPE
The goal of this protocol is to measure the frequency shift induced by a flux pulse of a given duration.

Since the flux line ends on the qubit chip, it is not possible to measure the flux pulse after propagation through the
fridge.

The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a fixed dephasing time.
A flux pulse with varying duration is played during the idle time. The Sx and Sy components of the Bloch vector are
measured by alternatively closing the Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing
 as a function of the flux pulse duration.

The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more details about the sequence and
the post-processing of the data.


Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import ge_averaged_measurement
import matplotlib.pyplot as plt


###################
# The QUA program #
###################
n_avg = 10_000  # Number of averages
# Flag to set to True if state discrimination is calibrated (where the qubit state is inferred from the 'I' quadrature).
# Otherwise, a preliminary sequence will be played to measure the averaged I and Q values when the qubit is in |g> and |e>.
state_discrimination = False
# Flux amplitude sweep (as a pre-factor of the flux amplitude) - must be within [-2; 2)
flux_amp_array = np.linspace(0, -0.2, 101)

with program() as cryoscope_amp:
    n = declare(int)  # QUA variable for the averaging loop
    flux_amp = declare(fixed)  # Flux amplitude pre-factor
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    if state_discrimination:
        state = declare(bool)
        state_st = declare_stream()
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    if not state_discrimination:
        # Calibrate the ground and excited states readout for deriving the Bloch vector
        # The ge_averaged_measurement() function is defined in macros.py
        # Note that if you have calibrated the readout to perform state discrimination, then the QUA program below can
        # be modified to directly fetch the qubit state.
        Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(thermalization_time, n_avg)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(flux_amp, flux_amp_array)):
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("x90", "qubit")
                # Play truncated flux pulse with varying amplitude
                align("qubit", "flux_line")
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                play("const" * amp(flux_amp), "flux_line")
                align("qubit", "flux_line")
                # Wait some time to ensure that the 2nd x90 pulse will arrive after the flux pulse
                wait(20 * u.ns)
                # Play second X/2 or Y/2
                align("qubit", "flux_line")
                with if_(flag):
                    play("x90", "qubit")
                with else_():
                    play("y90", "qubit")
                # Measure resonator state after the sequence
                align("resonator", "qubit")
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
                )
                # State discrimination if the readout has been calibrated
                if state_discrimination:
                    assign(state, I > ge_threshold)
                    save(state, state_st)
                # Wait cooldown time and save the results
                wait(thermalization_time * u.ns, "resonator", "qubit")
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix (x90/y90, flux amplitudes), average the 2D matrices together and store the
        # results on the OPX processor
        I_st.buffer(2).buffer(len(flux_amp_array)).average().save("I")
        Q_st.buffer(2).buffer(len(flux_amp_array)).average().save("Q")
        if state_discrimination:
            # Also save the qubit state
            state_st.boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state")
        else:
            # Also save the averaged I/Q values for the qubit in |g> and |e>
            Ig_st.average().save("Ig")
            Qg_st.average().save("Qg")
            Ie_st.average().save("Ie")
            Qe_st.average().save("Qe")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cryoscope_amp, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cryoscope_amp)
    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    else:
        results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    xplot = flux_amp_array * const_flux_amp
    while results.is_processing():
        # Fetch results
        if state_discrimination:
            I, Q, state, iteration = results.fetch_all()
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Bloch vector Sx + iSy
            qubit_state = (state[:, 0] * 2 - 1) + 1j * (state[:, 1] * 2 - 1)
        else:
            I, Q, Ie, Qe, Ig, Qg, iteration = results.fetch_all()
            # Phase of ground and excited states
            phase_g = np.angle(Ig + 1j * Qg)
            phase_e = np.angle(Ie + 1j * Qe)
            # Phase of cryoscope measurement
            phase = np.unwrap(np.angle(I + 1j * Q))
            # Population in excited state
            state = (phase - phase_g) / (phase_e - phase_g)
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Convert the results into Volts
            I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
            # Bloch vector Sx + iSy
            qubit_state = (state[:, 0] * 2 - 1) + 1j * (state[:, 1] * 2 - 1)

        # Accumulated phase: angle between Sx and Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        # qubit_phase = qubit_phase - qubit_phase[-1]
        detuning = qubit_phase / (2 * np.pi * const_flux_len / u.s)
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)
        # Quadratic fit of detuning versus flux pulse amplitude
        pol = np.polyfit(xplot, detuning, deg=2)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plots
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, I)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("I quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, Q)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Q quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, state)
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, detuning / u.MHz, "bo")
        plt.plot(xplot, np.polyval(pol, xplot) / u.MHz, "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [MHz]")
        plt.legend(("data", "Fit"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
