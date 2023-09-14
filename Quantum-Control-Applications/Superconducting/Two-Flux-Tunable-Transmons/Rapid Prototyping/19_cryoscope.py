"""
WARNING: the digital filters will add a global delay --> need to recalibrate readout !!
"""

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
from configuration import *

#########################################
# Set-up the machine and get the config #
#########################################
machine = QuAM("current_state.json", flat_data=False)
config = build_config(machine)


qb1 = machine.qubits[active_qubits[0]]
qb2 = machine.qubits[active_qubits[1]]
q1_z = machine.qubits[active_qubits[0]].name + "_z"
q2_z = machine.qubits[active_qubits[1]].name + "_z"
rr1 = machine.resonators[active_qubits[0]]
rr2 = machine.resonators[active_qubits[1]]
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq

qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2


qb = qb2


cooldown_time = 5 * max(qb1.T1, qb2.T1)
n_avg = 1000

##########
# baking #
##########
# Flux pulse waveform generation
zeros_before_pulse = 0  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 0  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
flux_waveform = np.array(
    [0.0] * zeros_before_pulse
    + [qb.z.flux_pulse_amp] * qb.z.flux_pulse_length
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

            b.add_op("flux_pulse", qb.name+"_z", wf)
            b.play("flux_pulse", qb.name+"_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, len(flux_waveform))
step_response = [1.0] * qb.z.flux_pulse_length
xplot = np.arange(0, len(flux_waveform) + 0.1, 1)

###################
# The QUA program #
###################


with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)
    segment = declare(int)  # Flux pulse segment
    flag = declare(bool)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(segment, 0, segment <= qb.z.flux_pulse_length + total_zeros, segment + 1):
            with for_each_(flag, [True, False]):
                play("x90", qb.name + "_xy")

                align()
                wait(20 * u.ns)

                with switch_(segment):
                    for j in range(0, len(flux_waveform) + 1):
                        with case_(j):
                            square_pulse_segments[j].run()

                wait((qb.z.flux_pulse_length + 100) * u.ns, qb.name + "_xy")

                with if_(flag):
                    play("x90", qb.name + "_xy")
                with else_():
                    play("y90", qb.name + "_xy")

                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                assign(state[0], I[0] > qb1.ge_threshold)
                assign(state[1], I[1] > qb2.ge_threshold)
                save(state[0], state_st[0])
                save(state[1], state_st[1])
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # Qubit state
        state_st[0].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("state1")
        state_st[1].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("state2")
        # I_st[0].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("I1")
        # I_st[1].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("I2")
        # Q_st[0].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("Q1")
        # Q_st[1].boolean_to_int().buffer(2).buffer(qb.z.flux_pulse_length + total_zeros + 1).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name)

simulate = False
if simulate:
    job = qmm.simulate(config, cryoscope, SimulationConfig(11000))
    job.get_simulated_samples().con1.plot()
else:
    qm = qmm.open_qm(config)
    job = qm.execute(cryoscope)
    fig = plt.figure()
    interrupt_on_close(fig, job)
    results = fetching_tool(job, ["n", "state1", "state2"], mode="live")
    while results.is_processing():
        n, state1, state2 = results.fetch_all()
        progress_counter(n, n_avg, start_time=results.start_time)
        # Accumulated phase: angle between Sx and Sy
        if qb == qb1:
            state = state1
        else:
            state = state2
        Sx = state[:, 0] * 2 - 1
        Sy = state[:, 1] * 2 - 1
        qubit_state = Sx + 1j * Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[-1]
        # Filtering and derivative of the phase to get the averaged frequency
        coarse_detuning = np.gradient(qubit_phase / 2 / np.pi, (xplot[1]-xplot[0])/u.s)
        detuning = signal.savgol_filter(qubit_phase / 2 / np.pi, 13, 3, deriv=1, delta=0.001)
        # Flux line step response in freq domain and voltage domain
        step_response_freq = detuning / np.average(detuning[-int(qb.z.flux_pulse_length / 2) :])
        step_response_volt = np.sqrt(step_response_freq)
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)
        plt.suptitle("Cryoscope")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, state1, ".-")
        plt.title(f"{qb1.name}")
        plt.xlabel("Interaction time [ns]")
        plt.ylabel("State")
        plt.legend(("Sx", "Sy"))
        plt.subplot(222)
        plt.cla()
        plt.title(f"{qb2.name}")
        plt.plot(xplot, state2, ".-")
        plt.xlabel("Interaction time [ns]")
        plt.legend(("Sx", "Sy"))
        plt.subplot(223)
        plt.cla()
        plt.plot(xplot, coarse_detuning / u.MHz, "-")
        plt.xlabel("Interaction time [ns]")
        plt.ylabel("Induced detuning [MHz]")
        plt.title(f"{qb.name}")
        plt.subplot(224)
        plt.cla()
        plt.plot(xplot, step_response_freq, label="Frequency")
        plt.plot(xplot, step_response_volt, label=r"Voltage ($\sqrt{freq}$)")
        plt.xlabel("Pulse duration [ns]")
        plt.ylabel("Step response")
        plt.title(f"{qb.name}")
        plt.legend()
        plt.tight_layout()
        plt.pause(5)

    ## Fit step response with exponential
    [A, tau], _ = scipy.optimize.curve_fit(
        expdecay,
        xplot,
        step_response_volt,
    )
    print(f"A: {A}\ntau: {tau}")

    ## Derive IIR and FIR corrections
    fir, iir = filter_calc(exponential=[(A, tau)])
    print(f"FIR: {fir}\nIIR: {iir}")

    ## Derive responses and plots
    # Ideal response
    pulse = np.array([1.0] * (qb.z.flux_pulse_length + 1))
    # Response without filter
    no_filter = expdecay(xplot, a=A, t=tau)
    # Response with filters
    with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], pulse)  # Output filter , DAC Output

    # Plot all data
    plt.figure()
    plt.suptitle("Cryoscope with filter implementation")
    plt.subplot(121)
    plt.plot(xplot, step_response_volt, "o-", label="Data")
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
    plt.plot(list(step_response_volt), label="Experimental data")
    plt.text(40, 0.93, f"IIR = {-iir}\nFIR = {fir}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.tight_layout()
    plt.legend(loc="upper right")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
# qb.z.wiring.filter.fir_taps = list(fir)
# qb.z.wiring.filter.iir_taps = list(-iir)
# machine._save("current_state.json")
