"""
Perform Cryoscope to measure the flux line step response and derive the pre-distortion filter taps
"""
from qm.qua import *
from quam import QuAM
from qm.QuantumMachinesManager import QuantumMachinesManager
import matplotlib.pyplot as plt
from qm import SimulationConfig
import numpy as np
from scipy import signal
import scipy.optimize
from datetime import datetime
from qualang_tools.bakery import baking
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.units import unit
from macros import ge_averaged_measurement

##################
# State and QuAM #
##################
u = unit()
debug = False
simulate = False
qubit_list = [0]
digital = []
machine = QuAM("latest_quam.json")
gate_shape = "drag_cosine"
now = datetime.now()
now = now.strftime("%m%d%Y_%H%M%S")

for q in qubit_list:
    machine.qubits[q].flux_bias_points.append({"name": "jump", "value": 0.1})
config = machine.build_config(digital, qubit_list, gate_shape)
q = 0
##############################
# Program-specific variables #
##############################
cooldown_time = 200 * u.us

# FLux pulse waveform generation
flux_len = 160
flux_pulse = np.array([machine.get_flux_bias_point(q, "jump").value] * flux_len)  # flux_len = 200 ns
zeros_before_pulse = 0  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 0  # End of the flux pulse (after we put zeros to see the falling time)
flux_waveform = np.array([0.0] * zeros_before_pulse + list(flux_pulse) + [0.0] * zeros_after_pulse)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", machine.qubits[q].name + "_flux", wf)
            b.play("flux_pulse", machine.qubits[q].name + "_flux")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
total_len = flux_len + zeros_before_pulse + zeros_after_pulse
square_pulse_segments = baked_waveform(flux_waveform, total_len)
step_response = [1.0] * flux_len
xplot = np.arange(0, total_len, 1)
# Number of averages
n_avg = 1e3

###################
# The QUA program #
###################

with program() as cryoscope:
    n = declare(int)  # Variable for averaging
    n_st = declare_stream()
    I = declare(fixed)  # I quadrature for state measurement
    Q = declare(fixed)  # Q quadrature for state measurement
    state = declare(bool)  # Qubit state
    state_st = declare_stream()
    I_st = declare_stream()
    Q_st = declare_stream()
    I_g = declare(fixed)  # I quadrature for qubit cooldown
    segment = declare(int)  # Flux pulse segment
    flag = declare(bool)  # Boolean flag to switch between x90 and y90 for state measurement

    # Set the flux line offset of the other qubit to 0
    # machine.nullify_other_qubits(qubit_list, q)
    # set_dc_offset(machine.qubits[q].name + "_flux", "single", machine.get_flux_bias_point(q, "insensitive_point").value)

    Ig_st, Qg_st, Ie_st, Qe_st = ge_averaged_measurement(machine, q, cooldown_time, 10 * n_avg)

    with for_(n, 0, n < n_avg, n + 1):
        # Notice it's <= to include t_max (This is only for integers!)
        with for_(segment, 0, segment < total_len, segment + 1):

            with for_each_(flag, [True, False]):
                # Cooldown
                wait(cooldown_time, machine.qubits[q].name)
                align()
                # wait(500)
                # Cryoscope protocol
                # Play the first pi/2 pulse
                play("x90", machine.qubits[q].name)
                align(machine.qubits[q].name, machine.qubits[q].name + "_flux")
                # Play truncated flux pulse with 1ns resolution
                with switch_(segment):
                    for j in range(0, total_len + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Wait some fixed time so that the whole protocol duration is constant
                wait(total_len // 4, machine.qubits[q].name)
                # Play the second pi/2 pulse along x and y successively
                with if_(flag):
                    play("x90", machine.qubits[q].name)
                with else_():
                    play("y90", machine.qubits[q].name)
                # State readout
                align(machine.qubits[q].name, machine.readout_resonators[q].name)
                measure(
                    "readout",
                    machine.readout_resonators[q].name,
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )
                # State discrimination
                assign(state, I > machine.readout_resonators[q].ge_threshold)
                save(state, state_st)
                save(I, I_st)
                save(Q, Q_st)
        save(n, n_st)

    with stream_processing():
        state_st.boolean_to_int().buffer(2).buffer(total_len).average().save("state")
        n_st.save("iteration")
        I_st.buffer(total_len, 2).average().save("I")
        Q_st.buffer(total_len, 2).average().save("Q")
        Ig_st.average().save("Ig")
        Qg_st.average().save("Qg")
        Ie_st.average().save("Ie")
        Qe_st.average().save("Qe")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.port)
# Open quantum machine
qm = qmm.open_qm(config)
if simulate:
    simulation_config = SimulationConfig(duration=50000)  # in clock cycles
    job = qmm.simulate(config, cryoscope, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Execute QUA program
    job = qm.execute(cryoscope)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "Ie", "Qe", "Ig", "Qg", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
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
        d_qubit_phase = scipy.signal.savgol_filter(qubit_phase / 2 / np.pi, 15, 3, deriv=1, delta=0.001)
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
        # plt.plot(xplot, theory, "r--", lw=3)
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.legend(("exp", "theory"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)
        # # Derive results
        # Sxx = state[:, 0] * 2 - 1  # Bloch vector projection along X
        # Syy = state[:, 1] * 2 - 1  # Bloch vector projection along Y
        # S = Sxx + 1j * Syy  # Bloch vector
        # # Qubit phase
        # phase = np.unwrap(np.angle(S))
        # phase = phase - phase[-1]
        # # Qubit detuning
        # detuning = signal.savgol_filter(
        #     phase[zeros_before_pulse : flux_len + zeros_before_pulse] / 2 / np.pi, 21, 2, deriv=1, delta=0.001
        # )
        # # Step response
        # step_response = detuning / np.average(detuning[-int(flux_len / 4)])
        # # plot results
        # plt.subplot(121)
        # plt.cla()
        # plt.plot(xplot, Sxx, ".-", label="Sxx")
        # plt.plot(xplot, Syy, ".-", label="Syy")
        # plt.xlabel("Flux pulse duration [ns]")
        # plt.ylabel("Bloch vector components")
        # plt.title("Cryoscope")
        # plt.legend()
        # plt.subplot(122)
        # plt.cla()
        # plt.plot(xplot[zeros_before_pulse : flux_len + zeros_before_pulse], detuning, ".-", label="Pulse")
        # plt.title("Square pulse response")
        # plt.xlabel("Flux pulse duration [ns]")
        # plt.ylabel("Qubit detuning [MHz]")
        # plt.legend()
        # plt.tight_layout()
        # plt.pause(0.01)


# Exponential decay
def expdecay(x, a, t):
    """Exponential decay defined as 1 + a * np.exp(-x / t).
    :param x: numpy array for the time vector in ns
    :param a: float for the exponential amplitude
    :param t: float for the exponential decay time in ns
    :return: numpy array for the exponential decay
    """
    return 1 + a * np.exp(-x / t)


# Theoretical IIR and FIR taps based on exponential decay coefficients
def exponential_correction(A, tau, Ts=1e-9):
    """Derive FIR and IIR filter taps based on a the exponential coefficients A and tau from 1 + a * np.exp(-x / t).
    :param A: amplitude of the exponential decay
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


# FIR and IIR taps calculation
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
    # Derive feddback tap for each set of exponential coefficients
    feedforward_taps = b[:, 0]
    for i in range(len(exponential) - 1):
        feedforward_taps = np.convolve(feedforward_taps, b[:, i + 1])
    # feedforward taps are bounded to +/- 2
    if np.abs(max(feedforward_taps)) >= 2:
        feedforward_taps = 2 * feedforward_taps / max(feedforward_taps)

    return feedforward_taps, feedback_taps


step_response = d_qubit_phase
## Fit step response with exponential
[A, tau], _ = scipy.optimize.curve_fit(
    expdecay, xplot[zeros_before_pulse : flux_len + zeros_before_pulse], step_response
)
print(f"A: {A}\ntau: {tau}")

## Derive IIR and FIR corrections
fir, iir = filter_calc(exponential=[(A, tau)])
print(f"FIR: {fir}\nIIR: {iir}")

## Derive responses and plots
# Ideal response
pulse = np.array([0.0] * zeros_before_pulse + [1.0] * flux_len + [0.0] * zeros_after_pulse)
# Response without filter
no_filter = expdecay(xplot, a=A, t=tau)
# Response with filters
with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], pulse)  # Output filter , DAC Output

# Plot all data
plt.rcParams.update({"font.size": 13})
plt.figure()
plt.suptitle("Cryoscope with filter implementation")
plt.subplot(121)
plt.plot(xplot, pulse, "o-", label="Data")
plt.plot(xplot, expdecay(xplot, A, tau), label="Fit")
plt.text(100, 0.95, f"A = {A:.2f}\ntau = {tau:.2f}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.axhline(y=1.01)
plt.axhline(y=0.99)
plt.xlabel("Flux pulse duration [ns]")
plt.ylabel("Step response")
plt.legend()

plt.subplot(122)
plt.plot()
plt.plot(no_filter, label="After Bias-T without filter")
plt.plot(with_filter, label="After Bias-T with filter")
plt.plot(pulse, label="Ideal WF")  # pulse
plt.plot(list(step_response), label="Experimental data")
plt.text(40, 0.93, f"IIR = {iir}\nFIR = {fir}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
plt.xlabel("Flux pulse duration [ns]")
plt.ylabel("Step response")
plt.legend(loc="upper right")
plt.tight_layout()
