"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the qubit pulse amplitude (pi_amp) in the state.
    - Save the current state by calling machine._save("current_state.json")
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

# Build the config
config = build_config(machine)

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(0.6, 1.4, 0.01)
# Number of applied Rabi pulses sweep
N_pi = 100  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")[::2]

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(npi, N_pi_vec)):
            with for_(*from_array(a, amps)):
                # Loop for error amplification (perform many qubit pulses)
                with for_(count, 0, count < npi, count + 1):
                    play("x180" * amp(a), qb1.name + "_xy")
                    play("x180" * amp(a), qb2.name + "_xy")
                # Align all elements to measure after playing the qubit pulse.
                align()
                # Start using Rotated-Readout:
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q2")

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
    job = qmm.simulate(config, rabi, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        if I1.shape[0] > 1:
            plt.suptitle("Power Rabi with error amplification")
            plt.subplot(321)
            plt.cla()
            plt.pcolor(amps * qb1.xy.pi_amp, N_pi_vec, I1)
            plt.title(f"{qb1.name} - I")
            plt.subplot(323)
            plt.cla()
            plt.pcolor(amps * qb1.xy.pi_amp, N_pi_vec, Q1)
            plt.title(f"{qb1.name} - Q")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Number of Rabi pulses")
            plt.subplot(322)
            plt.cla()
            plt.pcolor(amps * qb2.xy.pi_amp, N_pi_vec, I2)
            plt.title(f"{qb2.name} - I")
            plt.subplot(324)
            plt.cla()
            plt.pcolor(amps * qb2.xy.pi_amp, N_pi_vec, Q2)
            plt.title(f"{qb2.name} - Q")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Number of Rabi pulses")
            plt.subplot(325)
            plt.cla()
            plt.plot(amps * qb1.xy.pi_amp, np.sum(I1, axis=0))
            plt.axvline(qb1.xy.pi_amp, color="k")
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.ylabel(r"$\Sigma$ of Rabi pulses")
            plt.subplot(326)
            plt.cla()
            plt.plot(amps * qb2.xy.pi_amp, np.sum(I2, axis=0))
            plt.axvline(qb2.xy.pi_amp, color="k")
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.tight_layout()

        else:
            plt.suptitle("Power Rabi")
            plt.subplot(221)
            plt.cla()
            plt.plot(amps * qb1.xy.pi_amp, I1[0])
            plt.title(f"{qb1.name}")
            plt.ylabel("I quadrature [V]")
            plt.subplot(223)
            plt.cla()
            plt.plot(amps * qb1.xy.pi_amp, Q1[0])
            plt.xlabel("qubit pulse amplitudre [V]")
            plt.ylabel("Q quadrature [V]")
            plt.subplot(222)
            plt.cla()
            plt.plot(amps * qb2.xy.pi_amp, I2[0])
            plt.title(f"{qb2.name}")
            plt.subplot(224)
            plt.cla()
            plt.plot(amps * qb2.xy.pi_amp, Q2[0])
            plt.xlabel("qubit pulse amplitude [V]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Get the optimal pi pulse amplitude when doing error amplification
    try:
        qb1.xy.pi_amp = amps[np.argmax(np.sum(I1, axis=0))] * qb1.xy.pi_amp
        qb2.xy.pi_amp = amps[np.argmax(np.sum(I2, axis=0))] * qb2.xy.pi_amp
    except (Exception,):
        pass

# qb1.xy.pi_amp =
# qb2.xy.pi_amp =
# machine._save("current_state.json")
