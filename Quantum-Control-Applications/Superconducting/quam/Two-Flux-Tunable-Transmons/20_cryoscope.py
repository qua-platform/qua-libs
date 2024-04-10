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
a 1ns resolution, but must be shorter than ~260ns.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - Having calibrated the IQ blobs for state discrimination.

Next steps before going to the next node:
    - Update the FIR and IIR filter taps in the state (qubits[].z.wiring.filter.fir_taps & qubits[].z.wiring.filter.iir_taps).
    - Save the current state by calling machine.save("quam")
    - WARNING: the digital filters will add a global delay --> need to recalibrate IQ blobs (rotation_angle & ge_threshold).
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery import baking
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize, signal

from components import QuAM
from macros import qua_declaration, multiplexed_readout
from digital_filters import exponential_decay, single_exponential_correction


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("quam")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]


####################
# Helper functions #
####################
def baked_waveform(qubit, waveform, pulse_duration):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()

            b.add_op("flux_pulse", qubit.z.name, wf)
            b.play("flux_pulse", qubit.z.name)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


###################
# The QUA program #
###################
qb = q2  # Qubit under study
flux_operation = "const"
flux_pulse_len = q1.z.operations[flux_operation].length
flux_pulse_amp = q1.z.operations[flux_operation].amplitude
n_avg = 1000  # Number of averages

# FLux pulse waveform generation
# The zeros are just here to visualize the rising and falling times of the flux pulse. they need to be set to 0 before
# fitting the step response with an exponential.
zeros_before_pulse = 0  # Beginning of the flux pulse (before we put zeros to see the rising time)
zeros_after_pulse = 0  # End of the flux pulse (after we put zeros to see the falling time)
total_zeros = zeros_after_pulse + zeros_before_pulse
flux_waveform = np.array([0.0] * zeros_before_pulse + [flux_pulse_amp] * flux_pulse_len + [0.0] * zeros_after_pulse)
# Baked flux pulse segments with 1ns resolution
square_pulse_segments = baked_waveform(qb, flux_waveform, len(flux_waveform))
step_response = [1.0] * flux_pulse_len
xplot = np.arange(0, len(flux_waveform) + 0.1, 1)

with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    segment = declare(int)  # QUA variable for the flux pulse segment index
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    state = [declare(bool) for _ in range(2)]  # State of the qubits
    state_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        # Loop over the truncated flux pulse
        with for_(segment, 0, segment <= flux_pulse_len + total_zeros, segment + 1):
            # Alternate between X/2 and Y/2 pulses
            with for_each_(flag, [True, False]):
                # Play first X/2
                qb.xy.play("x90")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play truncated flux pulse
                with switch_(segment):
                    for j in range(0, len(flux_waveform) + 1):
                        with case_(j):
                            square_pulse_segments[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                qb.xy.wait((flux_pulse_len + 100) * u.ns)
                # Play second X/2 or Y/2
                with if_(flag):
                    qb.xy.play("x90")
                with else_():
                    qb.xy.play("y90")
                # Measure resonators state after the sequence
                align()
                multiplexed_readout(machine, I, I_st, Q, Q_st)
                # State discrimination
                assign(state[0], I[0] > q1.resonator.operations["readout"].threshold)
                assign(state[1], I[1] > q2.resonator.operations["readout"].threshold)
                # Wait cooldown time and save the results
                wait(machine.get_thermalization_time * u.ns)
                save(state[0], state_st[0])
                save(state[1], state_st[1])

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # Qubit state
        state_st[0].boolean_to_int().buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("state1")
        state_st[1].boolean_to_int().buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("state2")
        # I_st[0].buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("I1")
        # I_st[1].buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("I2")
        # Q_st[0].buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("Q1")
        # Q_st[1].buffer(2).buffer(flux_pulse_len + total_zeros + 1).average().save("Q2")


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
    results = fetching_tool(job, ["n", "state1", "state2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, state1, state2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Get the state of the qubit under study
        if qb == q1:
            state = state1
        else:
            state = state2
        # Derive the Bloch vector components from the two projections
        Sx = state[:, 0] * 2 - 1
        Sy = state[:, 1] * 2 - 1
        qubit_state = Sx + 1j * Sy
        # Accumulated phase: angle between Sx and Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[-1]
        # Filtering and derivative of the phase to get the averaged frequency
        coarse_detuning = np.gradient(qubit_phase / 2 / np.pi, (xplot[1] - xplot[0]) / u.s)
        detuning = signal.savgol_filter(qubit_phase / 2 / np.pi, 13, 3, deriv=1, delta=0.001)
        # Flux line step response in freq domain and voltage domain
        step_response_freq = detuning / np.average(detuning[-int(flux_pulse_len / 2) :])
        step_response_volt = np.sqrt(step_response_freq)
        # Qubit coherence: |Sx+iSy|
        qubit_coherence = np.abs(qubit_state)

        # Plot results
        plt.suptitle("Cryoscope")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, state1, ".-")
        plt.title(f"{q1.name}")
        plt.xlabel("Interaction time [ns]")
        plt.ylabel("State")
        plt.legend(("Sx", "Sy"))
        plt.subplot(222)
        plt.cla()
        plt.title(f"{q2.name}")
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
    [A, tau], _ = optimize.curve_fit(
        exponential_decay,
        xplot,
        step_response_volt,
    )
    print(f"A: {A}\ntau: {tau}")

    ## Derive IIR and FIR corrections
    fir, iir = single_exponential_correction(A, tau)
    print(f"FIR: {fir}\nIIR: {iir}")

    ## Derive responses and plots
    # Ideal response
    pulse = np.array([1.0] * (flux_pulse_len + 1))
    # Response without filter
    no_filter = exponential_decay(xplot, a=A, t=tau)
    # Response with filters
    with_filter = no_filter * signal.lfilter(fir, [1, iir[0]], pulse)  # Output filter , DAC Output

    # Plot all data
    plt.figure()
    plt.suptitle("Cryoscope with filter implementation")
    plt.subplot(121)
    plt.plot(xplot, step_response_volt, "o-", label="Data")
    plt.plot(xplot, exponential_decay(xplot, A, tau), label="Fit")
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
    plt.plot(list(step_response_volt), label="Experimental data")
    plt.text(40, 0.93, f"IIR = {iir}\nFIR = {fir}", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    plt.xlabel("Flux pulse duration [ns]")
    plt.ylabel("Step response")
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    qb.z.filter_fir_taps = list(fir)
    qb.z.filter_iir_taps = list(iir)
# machine.save("quam")
