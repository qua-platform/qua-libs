"""
        READOUT OPTIMISATION: DURATION
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
import math
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 10  # Number of runs
resonator = "rr1"
division_length = 1  # Size of each demodulation slice in clock cycles
number_of_divisions = int((readout_len) / (4 * division_length))
# Time axis for the plots at the end
pulse_duration = np.arange(division_length * 4, readout_len + 1, division_length * 4)
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "resonator": resonator,
    "division_length": division_length,
    "number_of_divisions": number_of_divisions,
    "pulse_duration": pulse_duration,
    "config": config,
}

###################
#   QUA Program   #
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
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        # Reset both qubits to ground
        wait(thermalization_time * u.ns)

        measure(
            "readout",
            resonator,
            None,
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
        )

        # Save the QUA vectors to their corresponding streams
        with for_(ind, 0, ind < number_of_divisions, ind + 1):
            assign(I[ind], II[ind] + IQ[ind])
            save(I[ind], Ig_st)
            assign(Q[ind], QQ[ind] + QI[ind])
            save(Q[ind], Qg_st)
            wait(2_000 >> 2)

        align()
        # Reset both qubits to ground
        wait(thermalization_time * u.ns)
        # Measure the excited IQ blobs
        play("x180", "q1_xy")
        play("x180", "q2_xy")
        align()
        measure(
            "readout",
            resonator,
            None,
            demod.accumulated("cos", II, division_length, "out1"),
            demod.accumulated("sin", IQ, division_length, "out2"),
            demod.accumulated("minus_sin", QI, division_length, "out1"),
            demod.accumulated("cos", QQ, division_length, "out2"),
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
        # mean values for |g> and |e>
        Ig_st.buffer(number_of_divisions).average().save(f"Ig_avg")
        Qg_st.buffer(number_of_divisions).average().save(f"Qg_avg")
        Ie_st.buffer(number_of_divisions).average().save(f"Ie_avg")
        Qe_st.buffer(number_of_divisions).average().save(f"Qe_avg")
        # variances for |g> and |e>
        (
            ((Ig_st.buffer(number_of_divisions) * Ig_st.buffer(number_of_divisions)).average())
            - (Ig_st.buffer(number_of_divisions).average() * Ig_st.buffer(number_of_divisions).average())
        ).save(f"Ig_var")
        (
            ((Qg_st.buffer(number_of_divisions) * Qg_st.buffer(number_of_divisions)).average())
            - (Qg_st.buffer(number_of_divisions).average() * Qg_st.buffer(number_of_divisions).average())
        ).save(f"Qg_var")
        (
            ((Ie_st.buffer(number_of_divisions) * Ie_st.buffer(number_of_divisions)).average())
            - (Ie_st.buffer(number_of_divisions).average() * Ie_st.buffer(number_of_divisions).average())
        ).save(f"Ie_var")
        (
            ((Qe_st.buffer(number_of_divisions) * Qe_st.buffer(number_of_divisions)).average())
            - (Qe_st.buffer(number_of_divisions).average() * Qe_st.buffer(number_of_divisions).average())
        ).save(f"Qe_var")

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
        # Get results from QUA program
        data_list = ["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"]
        results = fetching_tool(
            job,
            data_list=data_list,
            mode="live",
        )
        # Live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
        while results.is_processing():
            # Fetch results
            (
                Ig_avg,
                Qg_avg,
                Ie_avg,
                Qe_avg,
                Ig_var,
                Qg_var,
                Ie_var,
                Qe_var,
                iteration,
            ) = results.fetch_all()
            # Progress bar
            progress_counter(iteration, n_avg, start_time=results.get_start_time())
            # Derive the SNR
            ground_trace = Ig_avg + 1j * Qg_avg
            excited_trace = Ie_avg + 1j * Qe_avg
            var = (Ie_var + Qe_var + Ig_var + Qg_var) / 4
            SNR = (np.abs(excited_trace - ground_trace) ** 2) / (2 * var)
            # Plot results
            plt.subplot(131)
            plt.cla()
            plt.plot(ground_trace.real, label="ground")
            plt.plot(excited_trace.real, label="excited")
            plt.xlabel("Clock cycles [4ns]")
            plt.ylabel("demodulated traces [V]")
            plt.title("Real part qubit 1")
            plt.legend()
            plt.subplot(132)
            plt.cla()
            plt.plot(ground_trace.imag, label="ground")
            plt.plot(excited_trace.imag, label="excited")
            plt.xlabel("Clock cycles [4ns]")
            plt.title("Imaginary part qubit 1")
            plt.legend()
            plt.subplot(133)
            plt.cla()
            plt.plot(SNR, ".-")
            plt.xlabel("Clock cycles [4ns]")
            plt.ylabel("SNR qubit 1")
            plt.title("SNR")
            plt.tight_layout()
            plt.pause(1)
            # Get the optimal readout length in ns
            opt_readout_length = int(np.round(np.argmax(SNR) * division_length / 4) * 4 * 4)
        print(
            f"The optimal readout length for qubit associated with {resonator} is {opt_readout_length} ns (SNR={max(SNR)})"
        )

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="ro_opt_duration")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
