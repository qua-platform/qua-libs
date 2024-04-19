"""
        IQ BLOBS
This sequence involves measuring the state of the resonator 'N' times, first after thermalization (with the qubit
in the |g> state) and then after applying a pi pulse to the qubit (bringing the qubit to the |e> state) successively.
The resulting IQ blobs are displayed, and the data is processed to determine:
    - The rotation angle required for the integration weights, ensuring that the separation between |g> and |e> states
      aligns with the 'I' quadrature.
    - The threshold along the 'I' quadrature for effective qubit state discrimination.
    - The readout fidelity matrix, which is also influenced by the pi pulse fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the state.
    - Update the g -> e threshold (ge_threshold) in the state.
    - Save the current state by calling machine.save("quam")
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.units import unit
from qualang_tools.analysis.discriminator import two_state_discriminator

import matplotlib.pyplot as plt

from components import QuAM
from macros import qua_declaration, multiplexed_readout


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

###################
# The QUA program #
###################
n_runs = 10000  # Number of runs


with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)

    # Bring the active qubits to the minimum frequency point
    machine.apply_all_flux_to_min()

    with for_(n, 0, n < n_runs, n + 1):
        # ground iq blobs for both qubits
        wait(machine.get_thermalization_time * u.ns)
        align()
        multiplexed_readout(machine, I_g, I_g_st, Q_g, Q_g_st)

        align()
        # Wait for the qubit to decay to the ground state in the case of measurement induced transitions
        wait(machine.get_thermalization_time * u.ns)
        # excited iq blobs for both qubits
        q1.xy.play("x180")
        q2.xy.play("x180")
        align()
        multiplexed_readout(machine, I_e, I_e_st, Q_e, Q_e_st)

    with stream_processing():
        for i in range(2):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")


###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(iq_blobs)
    # fetch data
    results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
    I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()
    # Plot the IQ blobs, rotate them to get the separation along the 'I' quadrature, estimate a threshold between them
    # for state discrimination and derive the fidelity matrix
    angle1, threshold1, fidelity1, _, _, _, _ = two_state_discriminator(I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, True, True)
    plt.suptitle(f"{q1.name}")
    angle2, threshold2, fidelity2, _, _, _, _ = two_state_discriminator(I_g_q2, Q_g_q2, I_e_q2, Q_e_q2, True, True)
    plt.suptitle(f"{q2.name}")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    q1.resonator.operations["readout"].integration_weights_angle += angle1
    # rr1.readout_fidelity = fidelity1
    q1.resonator.operations["readout"].threshold = threshold1
    q2.resonator.operations["readout"].integration_weights_angle += angle2
    # rr2.readout_fidelity = fidelity2
    q2.resonator.operations["readout"].threshold = threshold2
    # machine.save("quam")
