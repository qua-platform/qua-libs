# %%
"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures.
This is done across various readout intermediate frequencies and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy_multiplexed").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF_q", in the configuration.
    - Adjust the readout amplitude, labeled as "readout_amp_q", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration_mw_fem import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
import math
from qualang_tools.results.data_handler import DataHandler
import matplotlib
from scipy import signal
import time

matplotlib.use('TkAgg')


##################
#   Parameters   #
##################

# Qubits and resonators 
rl = "rl1"
resonators = [key for key in RR_CONSTANTS.keys()]
resonators_LO = RL_CONSTANTS[rl]["LO"]

# Parameters Definition
n_avg = 20  # The number of averages
# The frequency sweep around the resonators' frequency
span = 38.0 * u.MHz  # the span around the resonant frequencies
step = 200 * u.kHz
dfs = np.arange(-span, span, step)
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
a_min = 0.01
a_max = 1.00
da = 0.01
# amplitudes = np .arange(a_min, a_max + da / 2, da)  # The amplitude vector +da/2 to add a_max to the scan
amplitudes = np.geomspace(a_min, a_max, 100)  # The amplitude vector +da/2 to add a_max to the scan

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Assertion
# assert len(qubits) == len(resonators), "qubits and resonators under study don't have the same length"
# assert all([qb.replace("_xy", "") == rr.replace("_rr", "") for qb, rr in zip(qubits, resonators)]), "qubits and resonators don't correspond"
# assert len(dfs) <= 32, "check your frequencies"
# assert len(amplitudes) <= 40, "check you amps vals"
for rr in resonators:
    assert a_max * RR_CONSTANTS[rr]["amp"] <= 0.499, f"{rr} max amp scan exceeded 0.499"

# Data to save
save_data_dict = {
    "resonators": resonators,
    "resonators_LO": resonators_LO,
    "n_avg": n_avg,
    "dfs": dfs,
    "amplitudes": amplitudes,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    # QUA macro to declare the measurement variables and their corresponding streams for a given number of resonators
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    df = declare(int)  # QUA variable for sweeping the readout frequency detuning around the resonance
    a = declare(fixed)  # QUA variable for sweeping the readout amplitude pre-factor

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the two resonator elements
            for rr in resonators:
                update_frequency(rr, df + RR_CONSTANTS[rr]["IF"])

            # with for_(*from_array(a, amplitudes)):  # QUA for_ loop for sweeping the readout amplitude
            with for_each_(a, amplitudes.tolist()):  # QUA for_ loop for sweeping the readout amplitude

                multiplexed_readout(I, I_st, Q, Q_st, None, None, resonators, amplitude=a)

                # wait for the resonators to empty
                wait(rr_reset_time >> 2)

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(len(dfs), len(amplitudes)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(dfs), len(amplitudes)).average().save(f"Q_{rr}")


if __name__ == "__main__":
    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    #######################
    # Simulate or execute #
    #######################
    simulate = False

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, PROGRAM, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show(block=False)

    else:
        try:
            # Open a quantum machine to execute the QUA program
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

                plt.suptitle("Multiplexed res spec vs amp")

                for ind, rr in enumerate(resonators):

                    S = res[2*ind+1] + 1j * res[2*ind+2]
                    R = np.abs(S) 
                    # phase = np.unwrap(np.angle(S))
                    phase = signal.detrend(np.unwrap(np.angle(S)))
                    row_sums = R.sum(axis=0)
                    R /= row_sums[np.newaxis, :]

                    save_data_dict[f"I_{rr}"] = res[2*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[2*ind + 2]

                    # Plot
                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.cla()
                    plt.pcolor(amplitudes * RR_CONSTANTS[rr]["amp"],(RR_CONSTANTS[rr]["IF"] + dfs) / u.MHz, R, cmap='magma')
                    # plt.pcolor(amplitudes * RR_CONSTANTS[rr]["amp"],(RR_CONSTANTS[rr]["IF"] + dfs) / u.MHz, phase, cmap='magma')
                    plt.xscale('log')
                    plt.xlim(amplitudes[0]*RR_CONSTANTS[rr]["amp"], amplitudes[-1]*RR_CONSTANTS[rr]["amp"])
                    # plt.axvline(x=RR_CONSTANTS[rr]["amp"])
                    # plt.axhline(y=RR_CONSTANTS[rr]["IF"] / u.MHz, linestyle='--', color="k")
                    lo_val = resonators_LO / u.GHz
                    plt.title(f"{rr} - LO: {lo_val} GHz")
                    plt.ylabel("Freqs [MHz]")

                plt.tight_layout()
                plt.pause(5)

            elapsed_time = time.time() - results.start_time
            save_data_dict["elapsed_time"] = elapsed_time

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="resonator_spectroscopy_amplitude")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
