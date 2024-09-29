# %%
"""
        RESONATOR SPECTROSCOPY MULTIPLEXED
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies for the two resonators simultaneously.
The data is then post-processed to determine the resonators' resonance frequency.
This frequency can be used to update the readout intermediate frequency in the configuration.

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Having found each resonator resonant frequency and updated the configuration (resonator_spectroscopy).
    - Specify the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF_q1" and "resonator_IF_q2", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration_mw_fem import *
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import multiplexed_readout
import math
from qualang_tools.results.data_handler import DataHandler
import matplotlib
from scipy import signal

matplotlib.use('TkAgg')

##################
#   Parameters   #
##################

# Qubits and resonators 
rl = "rl1"
resonators = [key for key in RR_CONSTANTS.keys()]
# resonators = ["q1_rr", "q3_rr"]
resonators_LO = RL_CONSTANTS[rl]["LO"]

# Parameters Definition
n_avg = 1_000  # The number of averages
# The frequency sweep parameters (for both resonators)
span = 30.0 * u.MHz  # the span around the resonant frequencies
step = 100 * u.kHz
dfs = np.arange(-span, span, step)

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Assertion
# assert len(dfs) <= 32, "check your frequencies"

# Data to save
save_data_dict = {
    "resonators": resonators,
    "resonators_LO": resonators_LO,
    "n_avg": n_avg,
    "dfs": dfs,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency detuning around the resonance
    # Here we define one 'I', 'Q', 'I_st' & 'Q_st' for each resonator via a python list
    I = [declare(fixed) for _ in range(len(resonators))]
    Q = [declare(fixed) for _ in range(len(resonators))]
    I_st = [declare_stream() for _ in range(len(resonators))]
    Q_st = [declare_stream() for _ in range(len(resonators))]

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            
            # wait for the resonators to empty
            wait(rr_reset_time >> 2)

            for rr in resonators:
                update_frequency(rr, df + RR_CONSTANTS[rr]["IF"])

            multiplexed_readout(I, I_st, Q, Q_st, None, None, resonators)
        
    with stream_processing():

        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(len(dfs)).average().save(f"I_{rr}")
            Q_st[ind].buffer(len(dfs)).average().save(f"Q_{rr}")


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
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, PROGRAM, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show(block=False)

    else:
        try:
            # Open a quantum machine to execute the QUA program
            qm = qmm.open_qm(config)
            # Execute the QUA program
            job = qm.execute(PROGRAM)
            # Tool to easily fetch results from the OPX (results_handle used in it)
            fetch_names = []
            for rr in resonators:
                fetch_names.append(f"I_{rr}")
                fetch_names.append(f"Q_{rr}")
            results = fetching_tool(job, fetch_names)
            fig = plt.figure()
            res = results.fetch_all()

            # Data analysis and plotting
            num_resonators = len(resonators)
            num_rows = math.ceil(math.sqrt(num_resonators))
            num_cols = math.ceil(num_resonators / num_rows)

            plt.suptitle("Multiplexed resonator spectroscopy")

            for ind, rr in enumerate(resonators):
                S = res[2*ind + 0] + 1j * res[2*ind + 1]
                R = np.abs(S) # np.unwarp(np.angle(S))
                phase = signal.detrend(np.unwrap(np.angle(S)))

                # Plot
                plt.subplot(num_rows, num_cols, ind + 1)
                plt.plot((RR_CONSTANTS[rr]["IF"] + dfs) / u.MHz, R)
                # plt.plot((RR_CONSTANTS[rr]["IF"] + dfs) / u.MHz, phase)
                plt.title(f"{rr} - LO: {resonators_LO / u.GHz} GHz")
                plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
                # plt.ylabel(r"Phase [rad]")

                save_data_dict[f"I_{rr}"] = res[2*ind + 0]
                save_data_dict[f"Q_{rr}"] = res[2*ind + 1]

            plt.tight_layout()

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="resonator_spectroscopy_multiplexed")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)


# %%
