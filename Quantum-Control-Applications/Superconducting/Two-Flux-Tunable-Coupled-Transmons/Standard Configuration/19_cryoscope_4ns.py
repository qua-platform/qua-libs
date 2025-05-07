"""
        CRYOSCOPE - 4ns
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

This version sweeps the flux pulse duration using real-time QUA, which means that the flux pulse can be arbitrarily long
but the step must be larger than 1 clock cycle (4ns) and the minimum pulse duration must be 4 clock cycles (16ns).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the configuration.

Next steps before going to the next node:
    - Update the FIR and IIR filter taps in the configuration:
        - For OPX+: (config/controllers/con1/analog_outputs/"filter": {"feedforward": fir, "feedback": iir}).
        - For OPX1000: (config/controllers/con1/analog_outputs/"filter": {"feedforward": [], "exponential": [(A, tau)]}).
    - WARNING: the digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold).
"""

import matplotlib.pyplot as plt
import numpy as np
from configuration import *
from macros import multiplexed_readout, qua_declaration
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.results.data_handler import DataHandler
from scipy import optimize, signal


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


##################
#   Parameters   #
##################
# Parameters Definition
# Index of the qubit to measure
qubit = 1
n_avg = 10_000  # Number of averages
# Flux pulse durations in clock cycles (4ns) - must be > 4 or the pulse won't be played.
durations = np.arange(3, const_flux_len // 4, 1)  # Starts at 3 clock-cycles to have the first point without pulse.
flux_waveform = np.array([const_flux_amp] * max(durations))
xplot = durations * 4  # x-axis for plotting and deriving the filter taps - must be in ns.
step_response_th = [1.0] * len(xplot)  # Perfect step response (square)


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "durations": durations,
    "flux_waveform": flux_waveform,
    "config": config,
}

###################
# The QUA program #
###################
with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    t = declare(int)  # QUA variable for the flux pulse duration
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        # Loop over the truncated flux pulse
        with for_(*from_array(t, durations)):
            # Alternate between X/2 and Y/2 pulses
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("x90", f"q{qubit}_xy")
                # Play truncated flux pulse
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play the flux pulse only if t is larger than the minimum of 4 clock cycles (16ns)
                with if_(t > 3):
                    play("const", f"q{qubit}_z", duration=t)
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                wait((len(flux_waveform) + 20) * u.ns, f"q{qubit}_xy")
                # Play second X/2 or Y/2
                with if_(flag):
                    play("x90", f"q{qubit}_xy")
                with else_():
                    play("y90", f"q{qubit}_xy")
                # Measure resonator state after the sequence
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # State discrimination
                assign(state[0], I[0] > ge_threshold_q1)
                assign(state[1], I[1] > ge_threshold_q2)
                save(state[0], state_st[0])
                save(state[1], state_st[1])
                # Wait cooldown time and save the results
                wait(thermalization_time * u.ns)
        save(n, n_st)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(len(durations)).average().save("I1")
        Q_st[0].buffer(2).buffer(len(durations)).average().save("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(len(durations)).average().save("state1")
        # resonator 2
        I_st[1].buffer(2).buffer(len(durations)).average().save("I2")
        Q_st[1].buffer(2).buffer(len(durations)).average().save("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(len(durations)).average().save("state2")

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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, cryoscope, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cryoscope)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "state1", "I2", "Q2", "state2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, state1, I2, Q2, state2 = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
        I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Bloch vector Sx + iSy
        if qubit == 1:
            Sxx = state1[:, 0] * 2 - 1
            Syy = state1[:, 1] * 2 - 1
        elif qubit == 2:
            Sxx = state2[:, 0] * 2 - 1
            Syy = state2[:, 1] * 2 - 1
        else:
            Sxx = 0
            Syy = 0
        S = Sxx + 1j * Syy
        # Accumulated phase: angle between Sx and Sy
        phase = np.unwrap(np.angle(S))
        phase = phase - phase[-1]
        # Filtering and derivative of the phase to get the averaged frequency
        detuning = signal.savgol_filter(phase / 2 / np.pi, 21, 2, deriv=1, delta=0.001)
        # Flux line step response in freq domain and voltage domain
        step_response_freq = detuning / np.average(detuning[-int(const_flux_len / 2) :])
        step_response_volt = np.sqrt(step_response_freq)
        # Plots
        plt.suptitle(f"Cryoscope for qubit {qubit} (qubit 1 (2) displayed on top (bottom))")
        plt.subplot(241)
        plt.cla()
        plt.plot(xplot, I1)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(242)
        plt.cla()
        plt.plot(xplot, Q1)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(243)
        plt.cla()
        plt.plot(xplot, state1)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(244)
        plt.cla()
        plt.plot(xplot, step_response_freq, label="Frequency")
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Step response")
        plt.title(f"Qubit {qubit}")
        plt.legend()

        plt.subplot(245)
        plt.cla()
        plt.plot(xplot, I2)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("I quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(246)
        plt.cla()
        plt.plot(xplot, Q2)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(247)
        plt.cla()
        plt.plot(xplot, state2)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Excited state population")
        plt.legend(("X", "Y"), loc="lower right")
        plt.subplot(248)
        plt.cla()
        plt.plot(xplot, step_response_volt, label=r"Voltage ($\sqrt{freq}$)")
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Step response")
        plt.legend()
        plt.title(f"Qubit {qubit}")

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
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I1_data": I1})
    save_data_dict.update({"Q1_data": Q1})
    save_data_dict.update({"state1_data": state1})
    save_data_dict.update({"I2_data": I2})
    save_data_dict.update({"Q2_data": Q2})
    save_data_dict.update({"state2_data": state2})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
