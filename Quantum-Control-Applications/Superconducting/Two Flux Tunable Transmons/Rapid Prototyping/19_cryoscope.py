from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from scipy import signal
import scipy.optimize
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery import baking
from macros import expdecay, filter_calc
from quam import QuAM
from configuration import build_config, u

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("quam_bootstrap_state.json", flat_data=False)
config = build_config(machine)

qubit_index = 0

##########
# baking #
##########
# FLux pulse waveform generation
zeros_before_pulse = 20  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 20  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
flux_waveform = np.array(
    [0.0] * zeros_before_pulse
    + [machine.qubits[qubit_index].z.flux_pulse_amp] * machine.qubits[qubit_index].z.flux_pulse_length
    + [0.0] * zeros_after_pulse
)


def baked_waveform(waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", f"q{qubit_index}_z", wf)
            b.play("flux_pulse", f"q{qubit_index}_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, len(flux_waveform))
step_response = [1.0] * machine.qubits[qubit_index].z.flux_pulse_length
xplot = np.arange(0, len(flux_waveform) + 0.1, 1)

###################
# The QUA program #
###################
cooldown_time = 5 * 50 * u.us
n_avg = 1000

with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    segment = declare(int)  # Flux pulse segment
    flag = declare(bool)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(segment, 0, segment <= machine.qubits[qubit_index].z.flux_pulse_length + total_zeros, segment + 1):
            with for_each_(flag, [True, False]):
                play("x90", "q1_xy")

                align()
                # TODO: this wait() creates time between pi and flux pulses, it is possible to calibrate the delay
                wait(20 * u.ns)

                with switch_(segment):
                    for j in range(0, len(flux_waveform) + 1):
                        with case_(j):
                            square_pulse_segments[j].run()

                wait((machine.qubits[qubit_index].z.flux_pulse_length + 100) * u.ns, f"q{qubit_index}_xy")

                with if_(flag):
                    play("x90", f"q{qubit_index}_xy")
                with else_():
                    play("y90", f"q{qubit_index}_xy")

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[0, 1], weights="rotated_")
                assign(state[qubit_index], I[qubit_index] > machine.qubits[qubit_index].ge_threshold)
                save(state[qubit_index], state_st[qubit_index])
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(machine.qubits[qubit_index].z.flux_pulse_length + total_zeros + 1).average().save("I1")
        Q_st[0].buffer(2).buffer(machine.qubits[qubit_index].z.flux_pulse_length + total_zeros + 1).average().save("Q1")
        # resonator 2
        I_st[1].buffer(2).buffer(machine.qubits[qubit_index].z.flux_pulse_length + total_zeros + 1).average().save("I2")
        Q_st[1].buffer(2).buffer(machine.qubits[qubit_index].z.flux_pulse_length + total_zeros + 1).average().save("Q2")
        # Qubit state
        state_st[qubit_index].boolean_to_int().buffer(2).buffer(
            machine.qubits[qubit_index].z.flux_pulse_length + total_zeros + 1
        ).average().save("state")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, machine.network.qop_port)

simulate = False
if simulate:
    job = qmm.simulate(config, cryoscope, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2", "state"], mode="live")
    while results.is_processing():
        n, I1, Q1, I2, Q2, state = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)

        plt.subplot(231)
        plt.cla()
        plt.plot(xplot, I1, ".-")
        plt.title("q1 - I")
        plt.ylabel("Interaction time (ns)")
        plt.subplot(232)
        plt.cla()
        plt.plot(xplot, Q1, ".-")
        plt.title("q1 - Q")
        plt.xlabel("Interaction time (ns)")
        plt.subplot(233)
        plt.cla()
        plt.title("qubit - state")
        plt.plot(xplot, state, ".-")
        plt.xlabel("Interaction time (ns)")
        plt.subplot(234)
        plt.cla()
        plt.plot(xplot, I2, ".-")
        plt.title("q2 - I")
        plt.xlabel("Interaction time (ns)")
        plt.subplot(235)
        plt.cla()
        plt.plot(xplot, Q2, ".-")
        plt.title("q2 - Q")
        plt.xlabel("Interaction time (ns)")
        plt.tight_layout()
        plt.pause(0.1)

    Sxx = state[:, 0] * 2 - 1
    Syy = state[:, 1] * 2 - 1
    S = Sxx + 1j * Syy

    phase = np.unwrap(np.angle(S))
    phase = phase - phase[-1]
    detuning = signal.savgol_filter(phase / 2 / np.pi, 21, 2, deriv=1, delta=0.001)
    step_response = detuning / np.average(detuning[-int(machine.qubits[qubit_index].z.flux_pulse_length / 2) :])

    plt.figure()
    plt.subplot(121)
    plt.plot(xplot, Sxx, label="Sxx")
    plt.plot(xplot, Syy, label="Sxx")
    plt.xlabel("Interaction time (ns)")
    plt.ylabel("Spin vector component")
    plt.legend()
    plt.subplot(122)
    plt.plot(xplot, step_response, label="freq")
    plt.plot(xplot, np.sqrt(step_response), label="sqrt(freq)")
    plt.xlabel("Interaction time (ns)")
    plt.legend()
    plt.tight_layout()

    ## Fit step response with exponential
    [A, tau], _ = scipy.optimize.curve_fit(
        expdecay,
        xplot,
        np.sqrt(step_response),
    )
    print(f"A: {A}\ntau: {tau}")

    ## Derive IIR and FIR corrections
    fir, iir = filter_calc(exponential=[(A, tau)])
    print(f"FIR: {fir}\nIIR: {iir}")

    ## Derive responses and plots
    # Ideal response
    pulse = np.array([1.0] * (machine.qubits[qubit_index].z.flux_pulse_length + 1))
    # Response without filter
    no_filter = expdecay(xplot, a=A, t=tau)
    # Response with filters
    with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], pulse)  # Output filter , DAC Output

    # Plot all data
    plt.rcParams.update({"font.size": 13})
    plt.figure()
    plt.suptitle("Cryoscope with filter implementation")
    plt.subplot(121)
    plt.plot(xplot, np.sqrt(step_response), "o-", label="Data")
    plt.plot(xplot, expdecay(xplot, A, tau), label="Fit")
    plt.text(100, 0.95, f"A = {A:.2f}\ntau = {tau:.2f}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.axhline(y=1.01)
    plt.axhline(y=0.99)
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    # plt.legend()

    plt.subplot(122)
    plt.plot()
    plt.plot(no_filter, label="After Bias-T without filter")
    plt.plot(with_filter, label="After Bias-T with filter")
    plt.plot(pulse, label="Ideal WF")  # pulse
    plt.plot(list(np.sqrt(step_response)), label="Experimental data")
    plt.text(40, 0.93, f"IIR = {iir}\nFIR = {fir}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.tight_layout()
    plt.legend(loc="upper right")

# machine.qubits[qubit_index].z.wiring.filter.fir_taps = fir
# machine.qubits[qubit_index].z.wiring.filter.iir_taps = iir
# machine._save("quam_bootstrap_state.json")
