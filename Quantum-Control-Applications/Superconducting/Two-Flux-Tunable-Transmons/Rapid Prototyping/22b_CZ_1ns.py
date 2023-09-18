"""
        CZ CHEVRON - 1ns granularity
The goal of this protocol is to find the parameters of the CZ gate between two flux-tunable qubits.
The protocol consists in flux tuning one qubit to bring the |11> state on resonance with |20>.
The two qubits must start in their excited states so that, when |11> and |20> are on resonance, the state |11> will
start acquiring a global phase when varying the flux pulse duration.

By scanning the flux pulse amplitude and duration, the CZ chevron can be obtained and post-processed to extract the
CZ gate parameters corresponding to a single oscillation period such that |11> pick up an overall phase of pi (flux
pulse amplitude and interation time).

This version sweeps the flux pulse duration using the baking tool, which means that the flux pulse can be scanned with
a 1ns resolution, but must be shorter than ~260ns. If you want to measure longer flux pulse, you can either reduce the
resolution (do 2ns steps instead of 1ns) or use the 4ns version (CZ.py).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having found the qubits maximum frequency point (qubit_spectroscopy_vs_flux).
    - Having calibrated qubit gates (x180) by running qubit spectroscopy, rabi_chevron, power_rabi, Ramsey and updated the state.
    - (Optional) having corrected the flux line distortions by running the Cryoscope protocol and updating the filter taps in the state.

Next steps before going to the next node:
    - Update the CZ gate parameters in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
import numpy as np
from macros import qua_declaration, multiplexed_readout
from qualang_tools.bakery import baking
import warnings

warnings.filterwarnings("ignore")


##########
# baking #
##########
def baked_waveform(waveform, pulse_duration, flux_qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    for i in range(0, pulse_duration + 1):
        with baking(config, padding_method="right") as b:
            if i == 0:  # Otherwise, the baking will be empty and will not be created
                wf = [0.0] * 16
            else:
                wf = waveform[:i].tolist()
            b.add_op("flux_pulse", flux_qubit.name + "_z", wf)
            b.play("flux_pulse", flux_qubit.name + "_z")
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
    return pulse_segments


#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Adjust the flux pulse amplitude and duration
machine.qubits[active_qubits[1]].z.flux_pulse_amp = -0.104
machine.qubits[active_qubits[1]].z.flux_pulse_length = 52

# Build the config
config = build_config(machine)


###################
# The QUA program #
###################
qb = qb2  # The qubit whose flux will be swept
n_avg = 40
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# The flux amplitude pre-factor
amps = np.arange(0.85, 1.2, 0.001)
# Flux pulse waveform generation
# The variable machine.qubits[qubit_index].z.flux_pulse_length is defined in the configuration
flux_waveform = np.array([qb.z.flux_pulse_amp] * qb.z.flux_pulse_length)
# Baked flux pulse segments
square_pulse_segments = baked_waveform(flux_waveform, qb.z.flux_pulse_length, qb)

with program() as cz:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the flux pulse amplitude pre-factor.
    segment = declare(int)  # QUA variable for the flux pulse segment index

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amps)):
            with for_(segment, 0, segment <= qb.z.flux_pulse_length, segment + 1):
                # Put the two qubits in their excited states
                play("x180", qb1.name + "_xy")
                play("x180", qb2.name + "_xy")
                align()
                # Wait some time to ensure that the flux pulse will arrive after the x90 pulse
                wait(20 * u.ns)
                # Play a flux pulse on the qubit with the highest frequency to bring it close to the excited qubit while
                # varying its amplitude and duration in order to observe the SWAP chevron with 1ns resolution.
                with switch_(segment):
                    for j in range(0, qb.z.flux_pulse_length + 1):
                        with case_(j):
                            square_pulse_segments[j].run(amp_array=[(qb.name + "_z", a)])
                align()
                # Wait some time to ensure that the flux pulse will end before the readout pulse
                wait(20 * u.ns)
                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits, weights="rotated_")
                # Wait for the qubits to decay to the ground state
                wait(cooldown_time * u.ns)

    with stream_processing():
        # for the progress counter
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("I1")
        Q_st[0].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("I2")
        Q_st[1].buffer(qb.z.flux_pulse_length + 1).buffer(len(amps)).average().save("Q2")

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
    job = qmm.simulate(config, cz, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cz)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Time axis for plotting
    xplot = np.arange(0, qb.z.flux_pulse_length + 0.1, 1)
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot the results
        plt.suptitle("CZ chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, I1.T)
        # plt.plot(qb.z.cz.level, qb.z.cz.length, "r*")
        plt.title(f"{qb1.name} - I, f_01={int(qb1.xy.f_01 / u.MHz)} MHz")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, Q1.T)
        # plt.plot(qb.z.cz.level, qb.z.cz.length, "r*")
        plt.title(f"{qb1.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.ylabel("Interaction time [ns]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, I2.T)
        # plt.plot(qb.z.cz.level, qb.z.cz.length, "r*")
        plt.title(f"{qb2.name} - I, f_01={int(qb2.xy.f_01 / u.MHz)} MHz")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * qb.z.flux_pulse_amp, xplot, Q2.T)
        # plt.plot(qb.z.cz.level, qb.z.cz.length, "r*")
        plt.title(f"{qb2.name} - Q")
        plt.xlabel("Flux amplitude [V]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    # qb.z.cz.length =
    # qb.z.cz.level =
# machine._save("current_state.json")
