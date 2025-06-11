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

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from macros import multiplexed_readout, qua_declaration
import math
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 400  # Number of runs
# The frequency sweep around the resonators' frequency "resonator_IF_q"
freq_span = 30e6
freq_step = 0.5e6
dfs = np.arange(-freq_span, freq_span, freq_step)
iq_blobs_analysis_method = "snr"  # "snr" "fidelity" or "overlap"

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "dfs": dfs,
    "config": config,
}

###################
# The QUA program #
###################
with program() as PROGRAM:

    Ig, I_g_st, Qg, Q_g_st, n, n_st = qua_declaration(nb_of_qubits=2)
    Ie, I_e_st, Qe, Q_e_st, _, _ = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the readout frequency

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)
        with for_(*from_array(df, dfs)):

            # Update the frequency of the two resonator elements
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the ground IQ blobs
            multiplexed_readout(Ig, I_g_st, Qg, Q_g_st, resonators=[1, 2], weights="rotated_")

            align()
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the excited IQ blobs
            play("x180", "q1_xy")
            play("x180", "q2_xy")
            align()
            # Measure the state of the resonators
            multiplexed_readout(Ie, I_e_st, Qe, Q_e_st, resonators=[1, 2], weights="rotated_")

    with stream_processing():
        n_st.save("iteration")
        for i in range(2):
            # mean values
            I_g_st[i].buffer(len(dfs)).average().save(f"Ig{i+1}_avg")
            Q_g_st[i].buffer(len(dfs)).average().save(f"Qg{i+1}_avg")
            I_e_st[i].buffer(len(dfs)).average().save(f"Ie{i+1}_avg")
            Q_e_st[i].buffer(len(dfs)).average().save(f"Qe{i+1}_avg")
            # variances to get the SNR
            (
                ((I_g_st[i].buffer(len(dfs)) * I_g_st[i].buffer(len(dfs))).average())
                - (I_g_st[i].buffer(len(dfs)).average() * I_g_st[i].buffer(len(dfs)).average())
            ).save(f"Ig{i+1}_var")
            (
                ((Q_g_st[i].buffer(len(dfs)) * Q_g_st[i].buffer(len(dfs))).average())
                - (Q_g_st[i].buffer(len(dfs)).average() * Q_g_st[i].buffer(len(dfs)).average())
            ).save(f"Qg{i+1}_var")
            (
                ((I_e_st[i].buffer(len(dfs)) * I_e_st[i].buffer(len(dfs))).average())
                - (I_e_st[i].buffer(len(dfs)).average() * I_e_st[i].buffer(len(dfs)).average())
            ).save(f"Ie{i+1}_var")
            (
                ((Q_e_st[i].buffer(len(dfs)) * Q_e_st[i].buffer(len(dfs))).average())
                - (Q_e_st[i].buffer(len(dfs)).average() * Q_e_st[i].buffer(len(dfs)).average())
            ).save(f"Qe{i+1}_var")


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
    job = qmm.simulate(config, PROGRAM, simulation_config)
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
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        # Prepare the figure for live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Get results from QUA program
        data_list = [
            "Ig1_avg",
            "Qg1_avg",
            "Ie1_avg",
            "Qe1_avg",
            "Ig1_var",
            "Qg1_var",
            "Ie1_var",
            "Qe1_var",
            "Ig2_avg",
            "Qg2_avg",
            "Ie2_avg",
            "Qe2_avg",
            "Ig2_var",
            "Qg2_var",
            "Ie2_var",
            "Qe2_var",
            "iteration",
        ]
        results = fetching_tool(job, data_list=data_list, mode="live")

        while results.is_processing():

            (
                Ig1_avg,
                Qg1_avg,
                Ie1_avg,
                Qe1_avg,
                Ig1_var,
                Qg1_var,
                Ie1_var,
                Qe1_var,
                Ig2_avg,
                Qg2_avg,
                Ie2_avg,
                Qe2_avg,
                Ig2_var,
                Qg2_var,
                Ie2_var,
                Qe2_var,
                iteration,
            ) = results.fetch_all()
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Derive the SNR
            Z1 = (Ie1_avg - Ig1_avg) + 1j * (Qe1_avg - Qg1_avg)
            var1 = (Ig1_var + Qg1_var + Ie1_var + Qe1_var) / 4
            SNR1 = ((np.abs(Z1)) ** 2) / (2 * var1)
            Z2 = (Ie2_avg - Ig2_avg) + 1j * (Qe2_avg - Qg2_avg)
            var2 = (Ig2_var + Qg2_var + Ie2_var + Qe2_var) / 4
            SNR2 = ((np.abs(Z2)) ** 2) / (2 * var2)
            # Plot results
            plt.suptitle("Readout frequency optimization")
            plt.subplot(121)
            plt.cla()
            plt.plot(dfs / u.MHz, SNR1, ".-")
            plt.title(f"Qubit 1 around {resonator_IF_q1 / u.MHz} MHz")
            plt.xlabel("Readout frequency detuning [MHz]")
            plt.ylabel("SNR")
            plt.grid("on")
            plt.subplot(122)
            plt.cla()
            plt.plot(dfs / u.MHz, SNR2, ".-")
            plt.title(f"Qubit 2 around {resonator_IF_q2 / u.MHz} MHz")
            plt.xlabel("Readout frequency detuning [MHz]")
            plt.grid("on")
            plt.tight_layout()
            plt.pause(1)

        print(f"The optimal readout frequency is {dfs[np.argmax(SNR1)] + resonator_IF_q1} Hz (SNR={max(SNR1)})")
        print(f"The optimal readout frequency is {dfs[np.argmax(SNR2)] + resonator_IF_q2} Hz (SNR={max(SNR2)})")

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"Z1_data": Z1})
        save_data_dict.update({"var1_data": var1})
        save_data_dict.update({"SNR1_data": SNR1})
        save_data_dict.update({"Z2_data": Z2})
        save_data_dict.update({"var2_data": var2})
        save_data_dict.update({"SNR2_data": SNR2})
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
