# %%
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
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
import math
from qualang_tools.results.data_handler import DataHandler
import matplotlib

matplotlib.use('TkAgg')

###################
# The QUA program #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 8*60  # The number of averages
t_max = 8_000
t_min = 4
t_step = 120
t_delays = np.arange(t_min, t_max, t_step)  # Idle time sweep in clock cycles (Needs to be a list of integers)
freq_detuning = 0.5 * u.MHz
delta_phase = 4e-09 * freq_detuning * t_step

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
qubits = [f"q{i}_xy" for i in [qc, qt]]
qubits_to_play = ["q4_xy"]
resonators = ['q4_rr'] #[f"q{i}_rr" for i in [qc, qt]]

# Assertion
assert len(t_delays) <= 76_000, "check your delays"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "t_delays": t_delays,
    "detuning": freq_detuning,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    t = declare(int)  # QUA variable for the idle time
    df = declare(int)  # QUA variable for the qubit frequency
    phase = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)
        assign(phase, 0)

        with for_(*from_array(t, t_delays)):

            assign(phase, phase + delta_phase)

            for qb in qubits_to_play:
                play('x90', qb)
                wait(t, qb)
                frame_rotation_2pi(phase, qb)
                play('x90', qb)

            # Align the elements to measure after having waited a time "tau" after the qubit pulses.
            align()

            wait(4)

            # Measure the state of the resonators
            multiplexed_readout(I, I_st, Q, Q_st, None, None, resonators=resonators, weights=weights)

            # Wait for the qubit to decay to the ground state
            if reset_method == "wait":
                wait(qb_reset_time >> 2)
            elif reset_method == "active":
                global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

            for qb in qubits:
                reset_frame(qb)


    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(len(t_delays)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(t_delays)).average().save(f"Q_{rr}")


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

                plt.suptitle("Multiplexed ramsey chevron - I")

                for ind, (qb, rr) in enumerate(zip(qubits, resonators)):
                    # Data analysis
                    S = res[2*ind+1] + 1j * res[2*ind+2]

                    save_data_dict[f"I_{rr}"] = res[2*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[2*ind + 2]

                    # Plot
                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.cla()
                    plt.plot(t_delays * 4, np.real(S), color='r')
                    lo_val = QUBIT_CONSTANTS[qb]["LO"] / u.GHz
                    plt.title(f"Qb - {qb}, LO {lo_val}")
                    plt.ylabel("Freqs [MHz]")

                plt.tight_layout()
                plt.pause(2)

            from qualang_tools.plot.fitting import Fit
            fit = Fit()
            fig_analysis = plt.figure(figsize=(6,6))
            ramsey_fit = fit.ramsey(4 * t_delays, np.real(S), plot=True)
            # qubit_T2 = np.abs(ramsey_fit["T2"][0])
            # qubit_detuning = ramsey_fit["f"][0] * u.GHz - delta_phase
            # plt.xlabel("Idle time [ns]")
            # plt.ylabel("I quadrature [V]")
            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="ramsey")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
