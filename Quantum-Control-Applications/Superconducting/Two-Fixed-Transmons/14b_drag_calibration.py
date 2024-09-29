# %%
"""
        DRAG PULSE CALIBRATION (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively while varying the DRAG
coefficient alpha. After such a sequence, the qubit is expected to always be in the ground state if the DRAG
coefficient has the correct value. Note that the idea is very similar to what is done in power_rabi_error_amplification.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the DRAG coefficient to a non-zero value in the config: such as drag_coef = 1

Next steps before going to the next node:
    - Update the DRAG coefficient (drag_coef) in the configuration.
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
amp_min = -1.9
amp_max = 1.9
amp_step = 0.01
amps = np.arange(amp_min, amp_max, amp_step)
# repeated rabi
max_nb_of_pulses = 40  # Maximum number of qubit pulses
nb_of_pulses = np.arange(0, max_nb_of_pulses + 0.1, 1)  # Always play an odd/even number of pulses to end up in the same state

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
qubits_to_play = qubits #['q2_xy']
resonators = [key for key in RR_CONSTANTS.keys()]

# Assertion
assert len(nb_of_pulses)*len(amps) <= 76_000, "check your frequencies and amps"
for qb in qubits:
    assert amp_max * QUBIT_CONSTANTS[qb]["amp"] <= 0.99, "amp_max times amplitude exceeded 0.499"
    assert QUBIT_CONSTANTS[qb]["drag_coefficient"] != 0, "The DRAG coefficient 'drag_coef' must be different from 0 in the config."
# Check that the DRAG coefficient is not 0

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
    state_st = [declare_stream() for _ in range(len(resonators))]
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
                        play("x180" * amp(1, 0, 0, a), qb)
                        play("x180" * amp(-1, 0, 0, -a), qb)

                # Align the elements to measure after playing the qubit pulses.
                align()
                # Multiplexed readout, also saves the measurement outcomes
                multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)
                
                # Wait for the qubit to decay to the ground state
                if reset_method == "wait":
                    wait(qb_reset_time >> 2)
                elif reset_method == "active":
                    global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(len(amps)).buffer(len(nb_of_pulses)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(amps)).buffer(len(nb_of_pulses)).average().save(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(len(amps)).buffer(len(nb_of_pulses)).average().save(f"state_{rr}")


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
                fetch_names.append(f"state_{rr}")
            # Tool to easily fetch results from the OPX (results_handle used in it)
            results = fetching_tool(job, fetch_names, mode="live")
            # Prepare the figure for live plotting
            fig, axss = plt.subplots(len(qubits), 4, figsize=(14, 4 * len(qubits)))
            interrupt_on_close(fig, job)
            # Live plotting
            while results.is_processing(): # or not loop_flag:
                # Fetch results
                res = results.fetch_all()
                # Progress bar
                progress_counter(res[0], n_avg, start_time=results.start_time)

                states = []
                plt.suptitle("Multiplexed DRAG Coefficient")
                for ind, (axs, qb, rr) in enumerate(zip(axss, qubits, resonators)):
                    # Data analysis
                    I, Q, state = res[3*ind+1], res[3*ind+2], res[3*ind+3]
                    states.append(state)

                    save_data_dict[f"I_{rr}"] = res[3*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[3*ind + 2]
                    save_data_dict[f"state_{rr}"] = res[3*ind + 3]

                    amp_min = amps[state.mean(axis=0).argmin()]

                    # Plot
                    axs[0].pcolor(amps * QUBIT_CONSTANTS[qb]["drag_coefficient"], nb_of_pulses, I, cmap='magma')
                    # axs[0].axvline(x=amp_min * QUBIT_CONSTANTS[qb]["drag_coefficient"], c='r')
                    axs[0].set_title(f"{qb}: I")
                    axs[0].set_xlabel("drag coefficient")
                    axs[0].set_ylabel("Number of Pulses")

                    axs[1].pcolor(amps * QUBIT_CONSTANTS[qb]["drag_coefficient"], nb_of_pulses, Q, cmap='magma')
                    # axs[1].axvline(x=amp_min * QUBIT_CONSTANTS[qb]["drag_coefficient"], c='r')
                    axs[1].set_title(f"{qb}: Q")
                    axs[1].set_xlabel("drag coefficient")
                    axs[1].set_ylabel("Number of Pulses")

                    axs[2].pcolor(amps * QUBIT_CONSTANTS[qb]["drag_coefficient"], nb_of_pulses, state, cmap='magma')
                    # axs[2].axvline(x=amp_min * QUBIT_CONSTANTS[qb]["drag_coefficient"], c='r')
                    axs[2].set_title(f"{qb}: state")
                    axs[2].set_xlabel("drag coefficient")
                    axs[2].set_ylabel("Number of Pulses")
                    
                    axs[3].cla()
                    axs[3].plot(amps * QUBIT_CONSTANTS[qb]["drag_coefficient"], state.mean(axis=0))
                    # axs[3].axvline(x=amp_min * QUBIT_CONSTANTS[qb]["drag_coefficient"], c='r')
                    axs[3].set_title(f"{qb}: averaged state")
                    axs[3].set_ylabel("averaged state")
                    axs[3].set_xlabel("detunings")

                plt.tight_layout()
                plt.pause(2)

            for qb, state in zip(qubits, states):
                print(f"{qb}: drag coefficient = ", amps[state.mean(axis=0).argmin()] * QUBIT_CONSTANTS[qb]["drag_coefficient"])

            elapsed_time = time.time() - results.start_time
            save_data_dict["elapsed_time"] = elapsed_time

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="drag_coefficient")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
