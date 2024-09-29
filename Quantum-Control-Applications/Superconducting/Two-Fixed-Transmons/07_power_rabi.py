# %%
"""
        POWER RABI WITH ERROR AMPLIFICATION
This sequence involves repeatedly executing the qubit pulse (such as x180, square_pi, or similar) 'N' times and
measuring the state of the resonator across different qubit pulse amplitudes and number of pulses.
By doing so, the effect of amplitude inaccuracies is amplified, enabling a more precise measurement of the pi pulse
amplitude. The results are then analyzed to determine the qubit pulse amplitude suitable for the selected duration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated the IQ mixer connected to the qubit drive line (external mixer or Octave port)
    - Having found the rough qubit frequency and pi pulse duration (rabi_chevron_duration or time_rabi).
    - Having found the pi pulse amplitude (power_rabi).
    - Set the qubit frequency, desired pi pulse duration and rough pi pulse amplitude in the configuration.

Next steps before going to the next node:
    - Update the qubit pulse amplitude (pi_amp_q) in the configuration.
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout, active_reset
import math
from qualang_tools.results.data_handler import DataHandler
import matplotlib
import time

matplotlib.use('TkAgg')

##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 10  # The number of averages
standard_rabi = False
if standard_rabi:
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amp_min = 0.5
    amp_max = 1.0
    amp_step = 0.1
    amps = np.arange(amp_min, amp_max, amp_step)
    nb_of_pulses = [1]
else:
    amp_min = 0.0
    amp_max = 1.5
    amp_step = 0.005
    amps = np.arange(amp_min, amp_max, amp_step)
    # repeated rabi
    max_nb_of_pulses = 80  # Maximum number of qubit pulses
    nb_of_pulses = np.arange(0, max_nb_of_pulses, 2)  # Always play an odd/even number of pulses to end up in the same state

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
# qubits = [f"q{i}_xy" for i in [qc, qt]]
# resonators = [f"q{i}_rr" for i in [qc, qt]]
qubits = [qb for qb in QUBIT_CONSTANTS.keys()]
qubits_to_play = ['q3_xy', 'q4_xy']
resonators = [key for key in RR_CONSTANTS.keys()]

# Assertion
assert len(nb_of_pulses)*len(amps) <= 76_000, "check your frequencies and amps"
# for qb in qubits:
    # assert amp_max * QUBIT_CONSTANTS[qb]["amp"] <= 1.0, "amp_max times amplitude exceeded 0.499"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "nb_of_pulses": nb_of_pulses,
    "amps": amps,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    with for_(n, 0, n < n_avg, n + 1):

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(npi, nb_of_pulses)):
            with for_(*from_array(a, amps)):
                # Loop for error amplification (perform many qubit pulses)
                with for_(count, 0, count < npi, count + 1):
                    for qb in qubits_to_play:
                        play("x180" * amp(a), qb)

                # Align the elements to measure after playing the qubit pulses.
                align()
                # Multiplexed readout, also saves the measurement outcomes
                multiplexed_readout(I, I_st, Q, Q_st, None, None, resonators=resonators, weights=weights)
                
                # Wait for the qubit to decay to the ground state
                if reset_method == "wait":
                    wait(qb_reset_time >> 2)
                elif reset_method == "active":
                    global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)


    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(len(nb_of_pulses), len(amps)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(nb_of_pulses), len(amps)).average().save(f"Q_{rr}")


if __name__ == "__main__":
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
            fetch_names = ["iteration"]
            for rr in resonators:
                fetch_names.append(f"I_{rr}")
                fetch_names.append(f"Q_{rr}")
            # Tool to easily fetch results from the OPX (results_handle used in it)
            results = fetching_tool(job, fetch_names, mode="live")
            # Prepare the figure for live plotting
            fig = plt.figure()
            interrupt_on_close(fig, job)
            # Data analysis and plotting
            num_resonators = len(resonators)
            num_rows = math.ceil(math.sqrt(num_resonators))
            num_cols = math.ceil(num_resonators / num_rows)
            # Live plotting
            while results.is_processing(): # or not loop_flag:
                # Fetch results
                res = results.fetch_all()
                # Progress bar
                progress_counter(res[0], n_avg, start_time=results.start_time)

                plt.suptitle("Multiplexed Power Rabi - I")
                for ind, (qb, rr) in enumerate(zip(qubits, resonators)):
                    # Data analysis
                    S = res[2*ind+1] + 1j * res[2*ind+2]

                    save_data_dict[f"I_{rr}"] = res[2*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[2*ind + 2]

                    # Plot
                    if res[1].shape[0] > 1:
                        plt.subplot(num_rows, num_cols, ind + 1)
                        plt.cla()
                        plt.pcolor(amps * QUBIT_CONSTANTS[qb]["amp"], nb_of_pulses, np.imag(S), cmap='magma')
                        # plt.axvline(x=QUBIT_CONSTANTS[qb]["amp"])
                        plt.title(f"Qubit {qb}")
                        plt.ylabel("Number of Pulses")
                    else:
                        plt.subplot(num_rows, num_cols, ind + 1)
                        plt.cla()
                        plt.plot(amps * QUBIT_CONSTANTS[qb]["amp"], np.imag(S[0]), color='r')
                        plt.title(f"Qubit {qb}")
                        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")

                plt.tight_layout()
                plt.pause(2)

            elapsed_time = time.time() - results.start_time
            save_data_dict["elapsed_time"] = elapsed_time

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="power_rabi")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
