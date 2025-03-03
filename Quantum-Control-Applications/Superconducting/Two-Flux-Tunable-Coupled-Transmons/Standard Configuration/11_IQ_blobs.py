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
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the rotation angle (rotation_angle) in the configuration.
    - Update the g -> e threshold (ge_threshold) in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm.simulate import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.analysis import two_state_discriminator
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_runs = 10000  # Number of runs

# Data to save
save_data_dict = {
    "n_runs": n_runs,
    "config": config,
}

###################
# The QUA program #
###################
with program() as iq_blobs:
    I_g, I_g_st, Q_g, Q_g_st, n, _ = qua_declaration(nb_of_qubits=2)
    I_e, I_e_st, Q_e, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)

    with for_(n, 0, n < n_runs, n + 1):
        # ground iq blobs for both qubits
        wait(thermalization_time * u.ns)
        align()
        # play("x180", "q2_xy")
        multiplexed_readout(I_g, I_g_st, Q_g, Q_g_st, resonators=[1, 2], weights="rotated_")

        # excited iq blobs for both qubits
        align()
        # Wait for the qubit to decay to the ground state in the case of measurement induced transitions
        wait(thermalization_time * u.ns)
        # Play the qubit pi pulses
        play("x180", "q1_xy")
        play("x180", "q2_xy")
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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, iq_blobs, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
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

    #########################################
    # The two_state_discriminator gives us the rotation angle which makes it such that all of the information will be in
    # the I axis. This is being done by setting the `rotation_angle` parameter in the configuration.
    # See this for more information: https://qm-docs.qualang.io/guides/demod#rotating-the-iq-plane
    # Once we do this, we can perform active reset using:
    #########################################
    #
    # # Active reset:
    # with if_(I > threshold):
    #     play("x180", "qubit")
    #
    #########################################
    #
    # # Active reset (faster):
    # play("x180", "qubit", condition=I > threshold)
    #
    #########################################
    #
    # # Repeat until success active reset
    # with while_(I > threshold):
    #     play("x180", "qubit")
    #     align("qubit", "resonator")
    #     measure("readout", "resonator", None,
    #                 dual_demod.full("rotated_cos", "rotated_sin", I))
    #
    #########################################
    #
    # # Repeat until success active reset, up to 3 iterations
    # count = declare(int)
    # assign(count, 0)
    # cont_condition = declare(bool)
    # assign(cont_condition, ((I > threshold) & (count < 3)))
    # with while_(cont_condition):
    #     play("x180", "qubit")
    #     align("qubit", "resonator")
    #     measure("readout", "resonator", None,
    #                 dual_demod.full("rotated_cos", "rotated_sin", I))
    #     assign(count, count + 1)
    #     assign(cont_condition, ((I > threshold) & (count < 3)))
    #
    #########################################
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_g_q1_data": I_g_q1})
    save_data_dict.update({"Q_g_q1_data": Q_g_q1})
    save_data_dict.update({"I_e_q1_data": I_e_q1})
    save_data_dict.update({"Q_e_q1_data": Q_e_q1})
    save_data_dict.update({"I_g_q2_data": I_g_q2})
    save_data_dict.update({"Q_g_q2_data": Q_g_q2})
    save_data_dict.update({"I_e_q2_data": I_e_q2})
    save_data_dict.update({"Q_e_q2_data": Q_e_q2})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
