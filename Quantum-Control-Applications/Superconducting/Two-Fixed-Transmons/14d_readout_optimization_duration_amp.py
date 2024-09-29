# %%
"""
        READOUT OPTIMISATION: DURATION & AMP
"""

from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from macros import iq_blobs_analysis
import math
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results.data_handler import DataHandler


##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 10  # Number of runs
# The frequency sweep around the resonators' frequency "resonator_IF_q"
a_max = 1.5
a_min = 0.5
a_step = 0.1
amps = np.arange(a_min, a_max, a_step)
iq_blobs_analysis_method = "snr" # "fidelity" or "overlap"
division_length = 10  # Size of each demodulation slice in clock cycles
number_of_divisions = int((READOUT_LEN) / (4 * division_length))
# Time axis for the plots at the end
pulse_duration = np.arange(division_length * 4, READOUT_LEN + 1, division_length * 4)
print("Integration weights chunk-size length in clock cycles:", division_length)
print("The readout has been sliced in the following number of divisions", number_of_divisions)

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
assert number_of_divisions <= 4_000, "check the number of divisions"
assert number_of_divisions*len(amps) <= 38_000, "check your frequencies and amps"
for rr in resonators:
    assert a_max * RR_CONSTANTS[rr]["amp"] < 0.5, f"{rr} a_max times amplitude exceeded 0.499"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "amps": amps,
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
    a = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(a, amps)):

            # Reset both qubits to ground
            wait(qb_reset_time >> 2)

            measure(
                "readout"*amp(a),
                resonators,
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
            wait(qb_reset_time >> 2)
            # Measure the excited IQ blobs
            for qb in qubits:
                play("x180", qb)
            align()
            measure(
                "readout"*amp(a),
                resonators,
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
        # for ind, rr in enumerate(resonators):
        for rr in resonators:
            Ig_st.buffer(len(amps), number_of_divisions).save_all(f"I_g_{rr}")
            Qg_st.buffer(len(amps), number_of_divisions).save_all(f"Q_g_{rr}")
            Ie_st.buffer(len(amps), number_of_divisions).save_all(f"I_e_{rr}")
            Qe_st.buffer(len(amps), number_of_divisions).save_all(f"Q_e_{rr}")


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
            fetch_names = ["iteration"]
            for rr in resonators:
                fetch_names.append(f"I_g_{rr}")
                fetch_names.append(f"Q_g_{rr}")
                fetch_names.append(f"I_e_{rr}")
                fetch_names.append(f"Q_e_{rr}")
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

                plt.suptitle("Readout Opt duration-amp")

                for ind, (qb, rr) in enumerate(zip(qubits, resonators)):

                    max_len = len(res[4*ind + 1])
                    _, _, iq_blobs_result = iq_blobs_analysis(
                        res[4*ind + 1][:max_len],
                        res[4*ind + 2][:max_len],
                        res[4*ind + 3][:max_len],
                        res[4*ind + 4][:max_len],
                        method=iq_blobs_analysis_method,
                    )
                    save_data_dict[rr+"_I_g"] = res[4*ind + 1]
                    save_data_dict[rr+"_Q_g"] = res[4*ind + 2]
                    save_data_dict[rr+"_I_e"] = res[4*ind + 3]
                    save_data_dict[rr+"_Q_e"] = res[4*ind + 4]


                    plt.subplot(num_rows, num_cols, ind + 1)
                    plt.clf()
                    plt.pcolor(pulse_duration, amps * RR_CONSTANTS[rr]["amp"], iq_blobs_result, cmap='magma')
                    plt.axhline(y=RR_CONSTANTS[rr]["amp"])
                    plt.ylabel("df [MHz]")
                    plt.title(f"Qb - {qb}")

                plt.tight_layout()
                plt.colorbar()
                plt.pause(1)

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="ro_opt_duration_amp")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

    # %%