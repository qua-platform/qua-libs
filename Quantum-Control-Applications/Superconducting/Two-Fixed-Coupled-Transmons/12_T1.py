"""
        RAMSEY CHEVRON (IDLE TIME VS FREQUENCY)
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different qubit intermediate
frequencies and idle times.
From the results, one can estimate the qubit frequency more precisely than by doing Rabi and also gets a rough estimate
of the qubit coherence time.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubit frequency (qubit_IF_q) in the configuration.
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100
t_max = 100_000
t_min = 4
# t_step = 1
t_delays = np.geomspace(t_min, t_max, 100).astype(int)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "t_delays": t_delays,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    t = declare(int)  # QUA variable for the idle time
    df = declare(int)  # QUA variable for the qubit frequency
    phase = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_each_(t, t_delays.tolist()):

            play("x180", "q1_xy")
            play("x180", "q2_xy")

            wait(t)

            # Align the elements to measure after having waited a time "tau" after the qubit pulses.
            align()

            # Measure the state of the resonators
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")

            wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("iteration")
        for ind in range(2):
            I_st[ind].buffer(len(t_delays)).average().save(f"I{ind+1}")
            Q_st[ind].buffer(len(t_delays)).average().save(f"Q{ind+1}")


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
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, PROGRAM, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        # Prepare the figure for live plotting
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, ["iteration", "I1", "Q1", "I2", "Q2"], mode="live")
        # Live plotting
        while results.is_processing():
            # Fetch results
            n, I1, Q1, I2, Q2 = results.fetch_all()
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
            # Plot
            plt.suptitle("T1 measurement")
            plt.subplot(221)
            plt.cla()
            plt.plot(4 * t_delays, I1)
            plt.ylabel("I quadrature [V]")
            plt.title("Qubit 1")
            plt.subplot(223)
            plt.cla()
            plt.plot(4 * t_delays, Q1)
            plt.ylabel("Q quadrature [V]")
            plt.xlabel("Wait time (ns)")
            plt.subplot(222)
            plt.cla()
            plt.plot(4 * t_delays, I2)
            plt.title("Qubit 2")
            plt.subplot(224)
            plt.cla()
            plt.plot(4 * t_delays, Q2)
            plt.title("Q2")
            plt.xlabel("Wait time (ns)")
            plt.tight_layout()
            plt.pause(1)

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="T1")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
