# %%
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
from macros import qua_declaration, multiplexed_readout, active_reset
import math
from qualang_tools.results.data_handler import DataHandler


##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 10  # The number of averages
# Qubit detuning sweep with respect to qubit_IF
freq_span = 40e6
freq_step = 0.1e6
dfs = np.arange(-freq_span, +freq_span, freq_step)
# Qubit pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amp_max = 1.00
amp_min = 0
amp_step = 0.25
amps = np.arange(amp_min, amp_max, amp_step)

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"q{i}_rr" for i in [qc, qt]]

# Assertion
assert len(dfs)*len(amps) <= 76_000, "check your frequencies and amps"
for qb in qubits:
    assert amp_max * QUBIT_CONSTANTS[qb]["amp"], f"{qb} amp_max times amplitude exceeded 0.499"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "dfs": dfs,
    "amps": amps,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    df = declare(int)  # QUA variable for the qubit detuning
    a = declare(fixed)  # QUA variable for the qubit pulse amplitude pre-factor

    with for_(n, 0, n < n_avg, n + 1):

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the frequency of the two qubit elements
            for qb in qubits:
                update_frequency(qb, df + QUBIT_CONSTANTS[qb]["IF"])

            with for_(*from_array(a, amps)):
                # Play qubit pulses simultaneously
                for qb in qubits:
                    play("x180" * amp(a), qb)

                # Measure after the qubit pulses
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
            I_st[ind].buffer(len(dfs), len(amps)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(dfs), len(amps)).average().save(f"Q_{rr}")


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
        plt.show(block=False)
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
            while results.is_processing():
                # Fetch results
                res = results.fetch_all()
                # Progress bar
                progress_counter(res[0], n_avg, start_time=results.start_time)

                plt.suptitle("Multiplexed rabi chevron - I")

                for ind, rr in enumerate(resonators):
                    # Data analysis
                    S = res[2*ind+1] + 1j * res[2*ind+2]

                    save_data_dict[f"I_{rr}"] = res[2*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[2*ind + 2]

                    # Plot
                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.cla()
                    plt.pcolor(amps * QUBIT_CONSTANTS[qb]["amp"], (dfs + QUBIT_CONSTANTS[qb]["IF"]) / u.MHz, np.real(S), cmap='magma')
                    lo_val = QUBIT_CONSTANTS[qb]["LO"] / u.GHz
                    plt.title(f"Qb - {qb}, LO {lo_val}")
                    plt.ylabel("Freqs [MHz]")

                plt.tight_layout()
                plt.pause(2)

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

# %%
