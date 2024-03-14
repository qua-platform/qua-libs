"""
        CRYOSCOPE AMPLITUDE
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
    - Having calibrated qubit gates (x90 and y90) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - Having calibrated the IQ blobs for state discrimination.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt


#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# The qubit under study
qb = qb1
# Adjust the flux pulse amplitude if needed
qb.z.flux_pulse_amp = 0.1
# Build the config
config = build_config(machine)


###################
# The QUA program #
###################
n_avg = 500
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# Flux amplitude sweep (as a pre-factor of the flux amplitude) - must be within [-2; 2)
flux_amp_array = np.linspace(0, 0.45, 1001)

with program() as cryoscope:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    flux_amp = declare(fixed)  # Flux amplitude pre-factor
    flag = declare(bool)  # QUA boolean to switch between x90 and y90
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(flux_amp, flux_amp_array)):
            with for_each_(flag, [True, False]):
                # Play first X/2
                play("x90", qb.name + "_xy")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play the flux pulse
                play("const" * amp(flux_amp), qb.name + "_z")
                align(qb.name + "_xy", qb.name + "_z")
                # Wait some time to ensure that the 2nd x90 pulse will arrive after the flux pulse
                wait(20 * u.ns)
                align()
                with if_(flag):
                    play("x90", qb.name + "_xy")
                with else_():
                    play("y90", qb.name + "_xy")

                # Measure resonators state after the sequence
                align()
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                # State discrimination
                assign(state[0], I[0] > qb1.ge_threshold)
                assign(state[1], I[1] > qb2.ge_threshold)
                # Wait cooldown time and save the results
                wait(cooldown_time * u.ns)
                save(state[0], state_st[0])
                save(state[1], state_st[1])

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # Qubit state
        state_st[0].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state1")
        state_st[1].boolean_to_int().buffer(2).buffer(len(flux_amp_array)).average().save("state2")
        # I_st[0].buffer(2).buffer(len(flux_amp_array)).average().save("I1")
        # I_st[1].buffer(2).buffer(len(flux_amp_array)).average().save("I2")
        # Q_st[0].buffer(2).buffer(len(flux_amp_array)).average().save("Q1")
        # Q_st[1].buffer(2).buffer(len(flux_amp_array)).average().save("Q2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

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
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    xplot = flux_amp_array * qb.z.flux_pulse_amp
    while results.is_processing():
        # Fetch results
        n, state1, state2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Get the state of the qubit under study
        if qb == qb1:
            state = state1
        else:
            state = state2
        # Derive the Bloch vector components from the two projections
        Sx = state[:, 0] * 2 - 1
        Sy = state[:, 1] * 2 - 1
        qubit_state = Sx + 1j * Sy
        qubit_phase = np.unwrap(np.angle(qubit_state))
        qubit_phase = qubit_phase - qubit_phase[0]
        # Filtering and derivative of the phase to get the averaged frequency
        coarse_detuning = qubit_phase / (2 * np.pi * qb.z.flux_pulse_length / u.s)
        # Quadratic fit of detuning versus flux pulse amplitude
        pol = np.polyfit(xplot, coarse_detuning, deg=2)

        # Plot the results
        plt.suptitle("Cryoscope")
        plt.subplot(221)
        plt.cla()
        plt.plot(xplot, state1, ".-")
        plt.title(f"{qb1.name}")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("State")
        plt.legend(("Sx", "Sy"))
        plt.subplot(222)
        plt.cla()
        plt.title(f"{qb2.name}")
        plt.plot(xplot, state2, ".-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.legend(("Sx", "Sy"))
        plt.subplot(212)
        plt.cla()
        plt.plot(xplot, coarse_detuning / u.MHz, ".")
        plt.plot(xplot, np.polyval(pol, xplot) / u.MHz, "r-")
        plt.xlabel("Flux pulse amplitude [V]")
        plt.ylabel("Averaged detuning [MHz]")
        plt.title(f"{qb.name}")
        plt.legend(("data", "Fit"), loc="upper right")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

# machine._save("current_state.json")
