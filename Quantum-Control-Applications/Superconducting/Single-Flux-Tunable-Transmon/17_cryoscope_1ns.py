"""
        CRYOSCOPE
The goal of this protocol is to measure the step response of the flux line and design proper FIR and IIR filters
(implemented on the OPX) to pre-distort the flux pulses and improve the two-qubit gates fidelity.
Since the flux line ends on the qubit chip, it is not possible to measure the flux pulse after propagation through the
fridge. The idea is to exploit the flux dependency of the qubit frequency, measured with a modified Ramsey sequence, to
estimate the flux amplitude received by the qubit as a function of time.

The sequence consists of a Ramsey sequence ("x90" - idle time - "x90" or "y90") with a fixed dephasing time.
A flux pulse with varying duration is played during the idle time. The Sx and Sy components of the Bloch vector are
measured by alternatively closing the Ramsey sequence with a "x90" or "y90" gate in order to extract the qubit dephasing
 as a function of the flux pulse duration.

The results are then post-processed to retrieve the step function of the flux line which is fitted with an exponential
function. The corresponding exponential parameters are then used to derive the FIR and IIR filter taps that will
compensate for the distortions introduced by the flux line (wiring, bias-tee...).
Such digital filters are then implemented on the OPX. Note that these filters will introduce a global delay on all the
output channels that may rotate the IQ blobs so that you may need to recalibrate them for state discrimination or
active reset protocols for instance. You can read more about these filters here:
https://docs.quantum-machines.co/0.1/qm-qua-sdk/docs/Guides/output_filter/?h=filter#hardware-implementation

The protocol is inspired from https://doi.org/10.1063/1.5133894, which contains more details about the sequence and
the post-processing of the data.

This version sweeps the flux pulse duration using the baking tool, which means that the flux pulse can be scanned with
a 1ns resolution, but must be shorter than ~260ns. If you want to measure longer flux pulse, you can either reduce the
resolution (do 2ns steps instead of 1ns) or use the 4ns version (cryoscope_4ns.py).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.

Next steps before going to the next node:
    - Update the FIR and IIR filter taps in the configuration (config/controllers/con1/analog_outputs/"filter": {"feedforward": fir, "feedback": iir}).
    - WARNING: the digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold).
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from macros import ge_averaged_measurement
from scipy import signal, optimize
import matplotlib.pyplot as plt


####################
# Helper functions #
####################
def exponential_decay(x, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).

    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return 1 + a * np.exp(-x / t)


def exponential_correction(A, tau, Ts=1e-9):
    """Derive FIR and IIR filter taps based on the exponential coefficients A and tau from 1 + a * np.exp(-x / t).

    :param A: amplitude of the exponential decay.
    :param tau: decay time of the exponential decay
    :param Ts: sampling period. Default is 1e-9
    :return: FIR and IIR taps
    """
    tau = tau * Ts
    k1 = Ts + 2 * tau * (A + 1)
    k2 = Ts - 2 * tau * (A + 1)
    c1 = Ts + 2 * tau
    c2 = Ts - 2 * tau
    feedback_tap = k2 / k1
    feedforward_taps = np.array([c1, c2]) / k1
    return feedforward_taps, feedback_tap


def filter_calc(exponential):
    """Derive FIR and IIR filter taps based on a list of exponential coefficients.

    :param exponential: exponential coefficients defined as [(A1, tau1), (A2, tau2)]
    :return: FIR and IIR taps as [fir], [iir]
    """
    # Initialization based on the number of exponential coefficients
    b = np.zeros((2, len(exponential)))
    feedback_taps = np.zeros(len(exponential))
    # Derive feedback tap for each set of exponential coefficients
    for i, (A, tau) in enumerate(exponential):
        b[:, i], feedback_taps[i] = exponential_correction(A, tau)
    # Derive feedback tap for each set of exponential coefficients
    feedforward_taps = b[:, 0]
    for i in range(len(exponential) - 1):
        feedforward_taps = np.convolve(feedforward_taps, b[:, i + 1])
    # feedforward taps are bounded to +/- 2
    if np.abs(max(feedforward_taps)) >= 2:
        feedforward_taps = 2 * feedforward_taps / max(feedforward_taps)

    return feedforward_taps, feedback_taps


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


###################
# The QUA program #
###################
n_avg = 10_000  # Number of averages
# Flag to set to True if state discrimination is calibrated (where the qubit state is inferred from the 'I' quadrature).
# Otherwise, a preliminary sequence will be played to measure the averaged I and Q values when the qubit is in |g> and |e>.
state_discrimination = False
# FLux pulse waveform generation
# The zeros are just here to visualize the rising and falling times of the flux pulse. they need to be set to 0 before
# fitting the step response with an exponential.
zeros_before_pulse = 20  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 20  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
flux_waveform = np.array([0.0] * zeros_before_pulse + [const_flux_amp] * const_flux_len + [0.0] * zeros_after_pulse)

# Baked flux pulse segments with 1ns resolution
square_pulse_segments = baked_waveform(flux_waveform, len(flux_waveform))
step_response_th = (
    [0.0] * zeros_before_pulse + [1.0] * (const_flux_len + 1) + [0.0] * zeros_after_pulse
)  # Perfect step response (square)
xplot = np.arange(0, len(flux_waveform) + 1, 1)  # x-axis for plotting - Must be in ns.

with program() as cryoscope:
    n = declare(int)  # QUA variable for the averaging loop
    segment = declare(int)  # QUA variable for the flux pulse segment index
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

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        with for_(segment, 0, segment <= const_flux_len + total_zeros, segment + 1):
            # Alternate between X/2 and Y/2 pulses
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("x90", "qubit")
                # Play truncated flux pulse
                align("qubit", "flux_line")
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                with switch_(segment):
                    for j in range(0, len(flux_waveform) + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                wait((len(flux_waveform) + 20) * u.ns, "qubit")
                # Play second X/2 or Y/2
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
                    dual_demod.full("rotated_cos", "rotated_sin", I),
                    dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
                )
                # State discrimination if the readout has been calibrated
                if state_discrimination:
                    assign(state, I > ge_threshold)
                    save(state, state_st)

                # Wait cooldown time and save the results
                wait(thermalization_time * u.ns)
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix (x90/y90, flux pulse length), average the 2D matrices together and store the
        # results on the OPX processor
        I_st.buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("I")
        Q_st.buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("Q")
        if state_discrimination:
            # Also save the qubit state
            state_st.boolean_to_int().buffer(2).buffer(const_flux_len + total_zeros + 1).average().save("state")
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
    job = qmm.simulate(config, cryoscope, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cryoscope)
    # Get results from QUA program
    if state_discrimination:
        results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    else:
        results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        if state_discrimination:
            I, Q, state, iteration = results.fetch_all()
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
            # Bloch vector Sx + iSy
            qubit_state = (state[:, 0] * 2 - 1) + 1j * (state[:, 1] * 2 - 1)

        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Accumulated phase: angle between Sx and Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[-1]
        # Filtering and derivative of the phase to get the averaged frequency
        detuning = signal.savgol_filter(qubit_phase / 2 / np.pi, 13, 3, deriv=1, delta=0.001)
        # Flux line step response in freq domain and voltage domain
        step_response_freq = detuning / np.average(detuning[-int(const_flux_len / 2) :])
        step_response_volt = np.sqrt(step_response_freq)
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)

        # Plots
        plt.suptitle("Cryoscope with 1ns resolution")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, I)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(222)
        plt.cla()
        plt.plot(xplot, Q)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, state)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")

        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, step_response_freq, label="Frequency")
        plt.plot(xplot, step_response_volt, label=r"Voltage ($\sqrt{freq}$)")
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Step response")
        plt.legend()
        plt.tight_layout()
        plt.pause(0.1)

    ## Fit step response with exponential
    [A, tau], _ = optimize.curve_fit(
        exponential_decay,
        xplot,
        step_response_volt,
    )
    print(f"A: {A}\ntau: {tau}")

    ## Derive IIR and FIR corrections
    fir, iir = filter_calc(exponential=[(A, tau)])
    print(f"FIR: {fir}\nIIR: {iir}")

    ## Derive responses and plots
    # Response without filter
    no_filter = exponential_decay(xplot, A, tau)
    # Response with filters
    with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], step_response_th)  # Output filter , DAC Output

    # Plot all data
    plt.rcParams.update({"font.size": 13})
    plt.figure()
    plt.suptitle("Cryoscope with filter implementation")
    plt.plot(xplot, step_response_volt, "o-", label="Experimental data")
    plt.plot(xplot, no_filter, label="Fitted response without filter")
    plt.plot(xplot, with_filter, label="Fitted response with filter")
    plt.plot(xplot, step_response_th, label="Ideal WF")  # pulse
    plt.text(
        max(xplot) // 2,
        max(step_response_volt) / 2,
        f"IIR = {iir}\nFIR = {fir}",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.text(
        max(xplot) // 4,
        max(step_response_volt) / 2,
        f"A = {A:.2f}\ntau = {tau:.2f}",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.legend(loc="upper right")
    plt.tight_layout()
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
