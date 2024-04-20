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
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit

import matplotlib.pyplot as plt
import numpy as np

from components import QuAM
from macros import qua_declaration, multiplexed_readout, node_save


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("state.json")
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.octave.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
q1 = machine.active_qubits[0]
q2 = machine.active_qubits[1]

###################
# The QUA program #
###################

operation = "x180"  # The qubit operation to play
n_avg = 100  # The number of averages

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(0.6, 1.4, 0.01)
# Number of applied Rabi pulses sweep
N_pi = 1  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")[::2]

with program() as rabi:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(npi, N_pi_vec)):
            with for_(*from_array(a, amps)):
                # Loop for error amplification (perform many qubit pulses)
                with for_(count, 0, count < npi, count + 1):
                    q1.xy.play("x180", amplitude_scale=a)
                    q2.xy.play("x180", amplitude_scale=a)
                # Align all elements to measure after playing the qubit pulse.
                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(machine, I, I_st, Q, Q_st)
                # Wait for the qubit to decay to the ground state
                wait(machine.get_thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(np.ceil(N_pi / 2)).average().save("Q2")


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
    # Calibrate the active qubits
    # machine.calibrate_active_qubits(qm)
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
        I1 = u.demod2volts(I1, q1.resonator.operations["readout"].length)
        Q1 = u.demod2volts(Q1, q1.resonator.operations["readout"].length)
        I2 = u.demod2volts(I2, q2.resonator.operations["readout"].length)
        Q2 = u.demod2volts(Q2, q2.resonator.operations["readout"].length)
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot results
        if I1.shape[0] > 1:
            plt.suptitle("Power Rabi with error amplification")
            plt.subplot(321)
            plt.cla()
            plt.pcolor(amps * q1.xy.operations[operation].amplitude, N_pi_vec, I1)
            plt.title(f"{q1.name} - I")
            plt.subplot(323)
            plt.cla()
            plt.pcolor(amps * q1.xy.operations[operation].amplitude, N_pi_vec, Q1)
            plt.title(f"{q1.name} - Q")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Number of Rabi pulses")
            plt.subplot(322)
            plt.cla()
            plt.pcolor(amps * q2.xy.operations[operation].amplitude, N_pi_vec, I2)
            plt.title(f"{q2.name} - I")
            plt.subplot(324)
            plt.cla()
            plt.pcolor(amps * q2.xy.operations[operation].amplitude, N_pi_vec, Q2)
            plt.title(f"{q2.name} - Q")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Number of Rabi pulses")
            plt.subplot(325)
            plt.cla()
            plt.plot(amps * q1.xy.operations[operation].amplitude, np.sum(I1, axis=0))
            plt.axvline(q1.xy.operations[operation].amplitude, color="k")
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.ylabel(r"$\Sigma$ of Rabi pulses")
            plt.subplot(326)
            plt.cla()
            plt.plot(amps * q2.xy.operations[operation].amplitude, np.sum(I2, axis=0))
            plt.axvline(q2.xy.operations[operation].amplitude, color="k")
            plt.xlabel("Rabi pulse amplitude [V]")
            plt.tight_layout()

        else:
            plt.suptitle("Power Rabi")
            plt.subplot(221)
            plt.cla()
            plt.plot(amps * q1.xy.operations[operation].amplitude, I1[0])
            plt.title(f"{q1.name}")
            plt.ylabel("I quadrature [V]")
            plt.subplot(223)
            plt.cla()
            plt.plot(amps * q1.xy.operations[operation].amplitude, Q1[0])
            plt.xlabel("qubit pulse amplitudre [V]")
            plt.ylabel("Q quadrature [V]")
            plt.subplot(222)
            plt.cla()
            plt.plot(amps * q2.xy.operations[operation].amplitude, I2[0])
            plt.title(f"{q2.name}")
            plt.subplot(224)
            plt.cla()
            plt.plot(amps * q2.xy.operations[operation].amplitude, Q2[0])
            plt.xlabel("qubit pulse amplitude [V]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Save data from the node
    data = {
        f"{q1.name}_amplitude": amps * q1.xy.operations[operation].amplitude,
        f"{q1.name}_I": np.abs(I1),
        f"{q1.name}_Q": np.angle(Q1),
        f"{q2.name}_amplitude": amps * q2.xy.operations[operation].amplitude,
        f"{q2.name}_I": np.abs(I2),
        f"{q2.name}_Q": np.angle(Q2),
        "figure": fig,
    }

    # Get the optimal pi pulse amplitude when doing error amplification
    try:
        q1.xy.operations[operation].amplitude = (
            amps[np.argmax(np.sum(I1, axis=0))] * q1.xy.operations[operation].amplitude
        )
        q2.xy.operations[operation].amplitude = (
            amps[np.argmax(np.sum(I2, axis=0))] * q2.xy.operations[operation].amplitude
        )

        data[f"{q1.name}"] = {"x180_amplitude": q1.xy.operations[operation].amplitude, "fit_successful": True}
        data[f"{q2.name}"] = {"x180_amplitude": q2.xy.operations[operation].amplitude, "fit_successful": True}
    except (Exception,):
        pass
    # Save data from the node
    node_save("power_rabi", data, machine)
