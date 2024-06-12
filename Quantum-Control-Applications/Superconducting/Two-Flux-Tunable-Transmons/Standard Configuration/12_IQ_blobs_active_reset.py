from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import qua_declaration, multiplexed_readout, reset_qubit


###################
# The QUA program #
###################
n_runs = 10000  # Number of runs

with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)

    with for_(n, 0, n < n_runs, n + 1):
        # ground iq blobs
        reset_qubit("cooldown", "q1_xy", "rr1", cooldown_time=thermalization_time)
        reset_qubit("cooldown", "q2_xy", "rr2", cooldown_time=thermalization_time)
        # reset_qubit("active", "q1_xy", "resonator", threshold=ge_threshold_q1, max_tries=10, Ig=I_g)
        align()
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=[1, 2], weights="rotated_")

        # excited iq blobs
        align()
        # Reset the qubits to the ground state in the case of measurement induced transitions
        reset_qubit("cooldown", "q1_xy", "rr1", cooldown_time=thermalization_time)
        reset_qubit("cooldown", "q2_xy", "rr2", cooldown_time=thermalization_time)
        # Play the qubit pi pulses
        play("x180", "q1_xy")
        # play("x180", "q2_xy")
        align()
        multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=[1, 2], weights="rotated_")

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        for i in range(2):
            I_g_st[i].save_all(f"I_g_q{i}")
            Q_g_st[i].save_all(f"Q_g_q{i}")
            I_e_st[i].save_all(f"I_e_q{i}")
            Q_e_st[i].save_all(f"Q_e_q{i}")

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
    two_state_discriminator(I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, True, True)
    plt.suptitle("qubit 1")
    two_state_discriminator(I_g_q2, Q_g_q2, I_e_q2, Q_e_q2, True, True)
    plt.suptitle("qubit 2")
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
