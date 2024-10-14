"""
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES 2D (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively for different DRAG
detunings. Here the detuning sweep has to be performed in python, because it involves changing the DRAG waveforms in a
non-linear manner.
After such a sequence, the qubit is expected to always be in the ground state if the AC Stark shift is
properly compensated by the DRAG detuning.
One can then take a line cut for a given number of pulse and fit the 1D trace with a parabola to get the optimum
detuning and update its value in the configuration.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having calibrated the DRAG coefficient.

Next steps before going to the next node:
    - Update the DRAG detuning (AC_stark_detuning) in the configuration.
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
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
# repeated rabi
max_nb_of_pulses = 80  # Maximum number of qubit pulses
nb_of_pulses = np.arange(
    0, max_nb_of_pulses + 0.1, 1
)  # Always play an odd/even number of pulses to end up in the same state
detunings = np.arange(-5e6, 5e6, 0.1e6)  # Detuning to compensate for the AC STark-shift

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "nb_of_pulses": nb_of_pulses,
    "detunings": detunings,
    "config": config,
}

###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]
    df = declare(int)  # QUA variable for the qubit drive detuning
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    with for_(n, 0, n < n_avg, n + 1):

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(npi, nb_of_pulses)):

            with for_(*from_array(df, detunings)):
                # Update the frequency of the two qubit elements

                update_frequency("q1_xy", df + qubit_IF_q1)
                update_frequency("q2_xy", df + qubit_IF_q2)

                # Loop for error amplification (perform many qubit pulses)

                with for_(count, 0, count < npi, count + 1):
                    play("x180" * amp(1), "q1_xy")
                    play("x180" * amp(1), "q2_xy")
                    play("x180" * amp(-1), "q1_xy")
                    play("x180" * amp(-1), "q2_xy")

                # Align the elements to measure after playing the qubit pulses.
                align()
                # Multiplexed readout, also saves the measurement outcomes
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")

                wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("iteration")
        for ind in range(2):
            I_st[ind].buffer(len(detunings)).buffer(len(nb_of_pulses)).average().save(f"I{ind+1}")
            Q_st[ind].buffer(len(detunings)).buffer(len(nb_of_pulses)).average().save(f"Q{ind+1}")


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
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            # Plots
            # Power Rabi with error amplification
            plt.subplot(221)
            plt.cla()
            plt.pcolor(detunings, nb_of_pulses, I1)
            plt.title("I1")
            plt.ylabel("# of pulses")
            plt.subplot(223)
            plt.cla()
            plt.pcolor(detunings, nb_of_pulses, Q1)
            plt.title("Q1")
            plt.xlabel("qubit pulse amplitude [V]")
            plt.ylabel("# of pulses")
            plt.subplot(222)
            plt.cla()
            plt.pcolor(detunings, nb_of_pulses, I2)
            plt.title("I2")
            plt.subplot(224)
            plt.cla()
            plt.pcolor(detunings, nb_of_pulses, Q2)
            plt.title("Q2")
            plt.xlabel("Qubit pulse amplitude [V]")
            plt.tight_layout()
            plt.pause(1)

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="ac_start_shift")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
