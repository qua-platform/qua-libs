"""
        READOUT OPTIMISATION: AMPLITUDE
The sequence consists in measuring the state of the resonator after thermalization (qubit in |g>) and after
playing a pi pulse to the qubit (qubit in |e>) successively while sweeping the readout amplitude.
The 'I' & 'Q' quadratures when the qubit is in |g> and |e> are extracted to derive the readout fidelity.
The optimal readout amplitude is chosen as to maximize the readout fidelity.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the readout frequency and updated the configuration.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout amplitude (readout_amp_q) in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.analysis import two_state_discriminator
from macros import multiplexed_readout, qua_declaration


###################
# The QUA program #
###################

n_runs = 1
# The readout amplitude sweep (as a pre-factor of the readout amplitude)
a_min = 0.9
a_max = 1.1
da = 0.01
amplitudes = np.arange(a_min, a_max + da / 2, da)  # The amplitude vector +da/2 to add a_max to the scan


with program() as ro_amp_opt:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)
    counter = declare(int, value=0)  # Counter for the progress bar
    counter_st = declare_stream()  # Stream for the counter variable
    a = declare(fixed)  # QUA variable for the readout amplitude

    with for_(*from_array(a, amplitudes)):
        # Save the counter to get the progress bar
        save(counter, counter_st)
        with for_(n, 0, n < n_runs, n + 1):
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the ground IQ blobs
            multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=[1, 2], weights="rotated_", amplitude=a)

            align()
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the excited IQ blobs
            play("x180", "q1_xy")
            play("x180", "q2_xy")
            align()
            multiplexed_readout(I_e, I_e_st, Q_e, Q_e_st, resonators=[1, 2], weights="rotated_", amplitude=a)
        # Increment the counter
        assign(counter, counter + 1)

    with stream_processing():
        # Save all streamed points for plotting the IQ blobs
        for i in range(2):
            I_g_st[i].buffer(n_runs).buffer(len(amplitudes)).save(f"I_g_q{i}")
            Q_g_st[i].buffer(n_runs).buffer(len(amplitudes)).save(f"Q_g_q{i}")
            I_e_st[i].buffer(n_runs).buffer(len(amplitudes)).save(f"I_e_q{i}")
            Q_e_st[i].buffer(n_runs).buffer(len(amplitudes)).save(f"Q_e_q{i}")
        counter_st.save("iteration")

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
    job = qmm.simulate(config, ro_amp_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_amp_opt)  # execute QUA program
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration"], mode="live")
    # Get progress counter to monitor runtime of the program
    while results.is_processing():
        # Fetch results
        iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration[0], len(amplitudes), start_time=results.get_start_time())

    # Fetch the results at the end
    results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
    I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()
    # Process the data
    fidelity_vec = [[], []]
    ground_fidelity_vec = [[], []]
    for i in range(len(amplitudes)):
        _, _, fidelity_q1, gg_q1, _, _, _ = two_state_discriminator(
            I_g_q1[i], Q_g_q1[i], I_e_q1[i], Q_e_q1[i], b_print=False, b_plot=False
        )
        _, _, fidelity_q2, gg_q2, _, _, _ = two_state_discriminator(
            I_g_q2[i], Q_g_q2[i], I_e_q2[i], Q_e_q2[i], b_print=False, b_plot=False
        )
        fidelity_vec[0].append(fidelity_q1)
        fidelity_vec[1].append(fidelity_q2)
        ground_fidelity_vec[0].append(gg_q1)
        ground_fidelity_vec[1].append(gg_q2)

    # Plot the data
    plt.figure()
    plt.suptitle("Readout amplitude optimization")
    plt.subplot(121)
    plt.plot(amplitudes * readout_amp_q1, fidelity_vec[0], "b.-", label="averaged fidelity")
    plt.plot(amplitudes * readout_amp_q1, ground_fidelity_vec[0], "r.-", label="ground fidelity")
    plt.title("Qubit 1")
    plt.xlabel("Readout amplitude [V]")
    plt.ylabel("Fidelity [%]")
    plt.legend()
    plt.subplot(122)
    plt.title("Qubit 2")
    plt.plot(amplitudes * readout_amp_q2, fidelity_vec[1], "b.-", label="averaged fidelity")
    plt.plot(amplitudes * readout_amp_q2, ground_fidelity_vec[1], "r.-", label="ground fidelity")
    plt.xlabel("Readout amplitude [V]")
    plt.ylabel("Fidelity [%]")
    plt.legend()
    plt.tight_layout()

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
