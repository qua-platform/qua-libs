"""
        READOUT OPTIMISATION: FREQUENCY & AMP
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF_q) in the configuration.
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from macros import multiplexed_readout, qua_declaration
from qualang_tools.results.data_handler import DataHandler
from qualang_tools.analysis import two_state_discriminator

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1000  # Number of runs
scaling_max = 1.99
scaling_min = 0.0
scaling_step = 0.025
scalings = np.arange(scaling_min, scaling_max, scaling_step)
iq_blobs_analysis_method = "snr"  # "snr" "fidelity" or "overlap"

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "scalings": scalings,
    "config": config,
}

###################
#   QUA Program   #
###################

with program() as PROGRAM:

    Ig, I_g_st, Qg, Q_g_st, n, n_st = qua_declaration(nb_of_qubits=2)
    Ie, I_e_st, Qe, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the readout frequency
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(a, scalings)):

            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the ground IQ blobs
            multiplexed_readout(Ig, I_g_st, Qg, Q_g_st, resonators=[1, 2], amplitude=a, weights="rotated_")

            align()
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the excited IQ blobs
            play("x180", "q1_xy")
            play("x180", "q2_xy")
            align()
            # Measure the state of the resonators
            multiplexed_readout(Ie, I_e_st, Qe, Q_e_st, resonators=[1, 2], amplitude=a, weights="rotated_")

    with stream_processing():
        n_st.save("iteration")
        for i in range(2):
            I_g_st[i].buffer(n_avg).buffer(len(scalings)).save(f"I_g_q{i}")
            Q_g_st[i].buffer(n_avg).buffer(len(scalings)).save(f"Q_g_q{i}")
            I_e_st[i].buffer(n_avg).buffer(len(scalings)).save(f"I_e_q{i}")
            Q_e_st[i].buffer(n_avg).buffer(len(scalings)).save(f"Q_e_q{i}")

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
    job = qmm.simulate(config, PROGRAM, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        results = fetching_tool(job, data_list=["iteration"], mode="live")
        # Get progress counter to monitor runtime of the program
        while results.is_processing():
            # Fetch results
            iteration = results.fetch_all()
            # Progress bar
            progress_counter(iteration[0], len(scalings), start_time=results.get_start_time())

        # Fetch the results at the end
        results = fetching_tool(job, ["I_g_q0", "Q_g_q0", "I_e_q0", "Q_e_q0", "I_g_q1", "Q_g_q1", "I_e_q1", "Q_e_q1"])
        I_g_q1, Q_g_q1, I_e_q1, Q_e_q1, I_g_q2, Q_g_q2, I_e_q2, Q_e_q2 = results.fetch_all()
        # Process the data
        fidelity_vec = [[], []]
        ground_fidelity_vec = [[], []]
        for i in range(len(scalings)):
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
        fig = plt.figure()
        plt.suptitle("Readout amplitude optimization")
        plt.subplot(121)
        plt.plot(scalings * readout_amp_q1, fidelity_vec[0], "b.-", label="averaged fidelity")
        plt.plot(scalings * readout_amp_q1, ground_fidelity_vec[0], "r.-", label="ground fidelity")
        plt.title("Qubit 1")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Fidelity [%]")
        plt.legend()
        plt.subplot(122)
        plt.title("Qubit 2")
        plt.plot(scalings * readout_amp_q2, fidelity_vec[1], "b.-", label="averaged fidelity")
        plt.plot(scalings * readout_amp_q2, ground_fidelity_vec[1], "r.-", label="ground fidelity")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Fidelity [%]")
        plt.legend()
        plt.tight_layout()

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="ro_opt_amp")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
