"""
        RABI CHEVRON (AMPLITUDE VS FREQUENCY)
This sequence involves executing the qubit pulse and measuring the state
of the resonator across various qubit intermediate frequencies and pulse amplitudes.
By analyzing the results, one can determine the qubit and estimate the x180 pulse amplitude for a specified duration.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse duration (labeled as "pi_len_q").

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "qubit_IF_q", in the configuration.
    - Modify the qubit pulse amplitude setting, labeled as "pi_amp_q", in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 10  # The number of averages
# Qubit detuning sweep with respect to qubit_IF
freq_span = 40e6
freq_step = 0.1e6
dfs = np.arange(-freq_span, +freq_span, freq_step)
# Qubit pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
scaling_max = 1.00
scaling_min = 0
scaling_step = 0.25
scalings = np.arange(scaling_min, scaling_max, scaling_step)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "dfs": dfs,
    "scalings": scalings,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit detuning
    a = declare(fixed)  # QUA variable for the qubit pulse amplitude pre-factor

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two qubit elements
            update_frequency("q1_xy", df + qubit_IF_q1)
            update_frequency("q2_xy", df + qubit_IF_q2)

            with for_(*from_array(a, scalings)):
                # Play qubit pulses simultaneously
                play("x180" * amp(a), "q1_xy")
                play("x180" * amp(a), "q2_xy")
                # Measure after the qubit pulses
                align()
                # Multiplexed readout, also saves the measurement outcomes
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(scalings)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(scalings)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(scalings)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(scalings)).buffer(len(dfs)).average().save("Q2")


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
        fig = plt.figure()
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
        # Live plotting
        while results.is_processing():
            # Fetch results
            n, I1, Q1, I2, Q2 = results.fetch_all()
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            # Plots
            plt.suptitle("Rabi chevron")
            plt.subplot(221)
            plt.cla()
            plt.pcolor(scalings * pi_amp_q1, dfs, I1)
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit 1 detuning [MHz]")
            plt.title(f"q1 (f_res: {(qubit_LO_q1 + qubit_IF_q1) / u.MHz} MHz)")
            plt.subplot(223)
            plt.cla()
            plt.pcolor(scalings * pi_amp_q1, dfs, Q1)
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit 1 detuning [MHz]")
            plt.subplot(222)
            plt.cla()
            plt.pcolor(scalings * pi_amp_q2, dfs, I2)
            plt.title(f"q2 (f_res: {(qubit_LO_q2 + qubit_IF_q2) / u.MHz} MHz)")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit 2 detuning [MHz]")
            plt.subplot(224)
            plt.cla()
            plt.pcolor(scalings * pi_amp_q2, dfs, Q2)
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.ylabel("Qubit 2 detuning [MHz]")
            plt.tight_layout()
            plt.pause(1)

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="rabi_chevron")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
