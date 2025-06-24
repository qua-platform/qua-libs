"""
READOUT OPTIMISATION: INTEGRATION WEIGHTS
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results.data_handler import DataHandler

####################
# Helper functions #
####################


def normalize_complex_array(arr):
    # Calculate the simple norm of the complex array
    norm = np.sqrt(np.sum(np.abs(arr) ** 2))

    # Normalize the complex array by dividing it by the norm
    normalized_arr = arr / norm

    # Rescale the normalized array so that the maximum value is 1
    max_val = np.max(np.abs(normalized_arr))
    rescaled_arr = normalized_arr / max_val

    return rescaled_arr


def plot_three_complex_arrays(x, aq1_rr, aq2_rr, aq3_rr, axs):
    (ax1, ax2, ax3) = axs
    ax1.plot(x, aq1_rr.real, label="real")
    ax1.plot(x, aq1_rr.imag, label="imag")
    ax1.set_title("ground state")
    ax1.set_xlabel("Readout time [ns]")
    ax1.set_ylabel("demod traces [a.u.]")
    ax1.legend()
    ax2.plot(x, aq2_rr.real, label="real")
    ax2.plot(x, aq2_rr.imag, label="imag")
    ax2.set_title("excited state")
    ax2.set_xlabel("Readout time [ns]")
    ax2.set_ylabel("demod traces [a.u.]")
    ax2.legend()
    ax3.plot(x, aq3_rr.real, label="real")
    ax3.plot(x, aq3_rr.imag, label="imag")
    ax3.set_title("OPT WEIGHTS")
    ax3.set_xlabel("Readout time [ns]")
    ax3.set_ylabel("subtracted traces [a.u.]")
    ax3.legend()


##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1e3  # number of averages
# Set the sliced demod parameters
division_length = 1  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len) / (4 * division_length))
# Time axis for the plots at the end
x_plot = np.arange(division_length * 4, readout_len + 1, division_length * 4)
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

qubit = "q1_xy"
resonator = "rr1"

# Data to save
save_data_dict = {
    "qubit": qubit,
    "resonator": resonator,
    "n_avg": n_avg,
    "division_length": division_length,
    "number_of_divisions": number_of_divisions,
    "x_plot": x_plot,
    "config": config,
}

###################
# The QUA program #
###################
with program() as PROGRAM:
    II = declare(fixed, size=number_of_divisions)
    IQ = declare(fixed, size=number_of_divisions)
    QI = declare(fixed, size=number_of_divisions)
    QQ = declare(fixed, size=number_of_divisions)
    I = declare(fixed, size=number_of_divisions)
    Q = declare(fixed, size=number_of_divisions)
    ind = declare(int)

    n_st = declare_stream()
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    Ie_st = declare_stream()
    Qe_st = declare_stream()
    n = declare(int)
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        # Reset both qubits to ground
        wait(thermalization_time * u.ns)

        measure(
            "readout",
            resonator,
            None,
            demod.sliced("cos", II, division_length, "out1"),
            demod.sliced("minus_sin", IQ, division_length, "out2"),
            demod.sliced("sin", QI, division_length, "out1"),
            demod.sliced("cos", QQ, division_length, "out2"),
        )

        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ig_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qg_st)
            wait(2_000 >> 2)

        # Measure the excited IQ blobs
        align()

        # Reset both qubits to ground
        wait(thermalization_time * u.ns)

        # Measure the excited IQ blobs
        play("x180", qubit)

        align()
        # Loop over the two resonators
        measure(
            "readout",
            resonator,
            None,
            demod.sliced("cos", II, division_length, "out1"),
            demod.sliced("minus_sin", IQ, division_length, "out2"),
            demod.sliced("sin", QI, division_length, "out1"),
            demod.sliced("cos", QQ, division_length, "out2"),
        )

        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ie_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qe_st)
            wait(2_000 >> 2)

    with stream_processing():
        n_st.save("iteration")
        Ig_st.buffer(number_of_divisions).average().save("I_g")
        Qg_st.buffer(number_of_divisions).average().save("Q_g")
        Ie_st.buffer(number_of_divisions).average().save("I_e")
        Qe_st.buffer(number_of_divisions).average().save("Q_e")

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
        fetch_names = ["iteration", "I_g", "Q_g", "I_e", "Q_e"]
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, fetch_names, mode="live")
        # Prepare the figure for live plotting
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        interrupt_on_close(fig, job)
        # Live plotting
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            # Progress bar
            progress_counter(res[0], n_avg, start_time=results.start_time)

            save_data_dict["I_g"] = res[1]
            save_data_dict["Q_g"] = res[2]
            save_data_dict["I_e"] = res[3]
            save_data_dict["Q_e"] = res[4]

            ground_trace = res[1] + 1j * res[2]
            excited_trace = res[3] + 1j * res[4]
            subtracted_trace = excited_trace - ground_trace
            normalized_subtracted_trace = normalize_complex_array(
                subtracted_trace
            )  # <- these are the optimal weights :)

            # Plot the results
            plt.cla()
            plot_three_complex_arrays(x_plot, ground_trace, excited_trace, normalized_subtracted_trace, axs)
            plt.suptitle(f"Integration weight optimization for qubit associated with {resonator}")

            plt.tight_layout()
            plt.pause(2)

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

        # Reshape the optimal integration weights to match the configuration
        weights_real = normalized_subtracted_trace.real
        weights_minus_imag = -normalized_subtracted_trace.imag
        weights_imag = normalized_subtracted_trace.imag
        weights_minus_real = -normalized_subtracted_trace.real

        # Save the weights for later use in the config
        np.savez(
            f"optimal_weights_{resonator}",
            weights_real=weights_real,
            weights_minus_imag=weights_minus_imag,
            weights_imag=weights_imag,
            weights_minus_real=weights_minus_real,
            division_length=division_length,
        )

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
