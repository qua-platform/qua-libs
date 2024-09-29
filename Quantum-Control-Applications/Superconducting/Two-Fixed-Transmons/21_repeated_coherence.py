# %%
"""
        T1, T2, T2e MEASUREMENT
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
import time, datetime

matplotlib.use('TkAgg')

##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
SAVE_DIR = Path(r"C:\\Users\\swusr\\OneDrive - QM Machines LTD\\fujitsu_results\\repeated_coherence")
n_avg = 240  # The number of averages
params_T1 = {   
    "xscale": "log",
    "tau_min": 4, # in clock cycles
    "tau_max": 45_000, # in clock cycles
    "step_tau": 20, # in clock cycles
    "num_points": 20,
    "add_to_xaxis": 0,
}
params_T2 = {
    "xscale": "linear",
    "tau_min": 4, # in clock cycles
    "tau_max": 4_000, # in clock cycles
    "step_tau": 120, # in clock cycles
    "add_to_xaxis": PI_LEN,
    "freq_detuning":  0.15 * u.MHz,
}
params_T2e = {
    "xscale": "log",
    "tau_min": 4, # in clock cycles
    "tau_max": 30000, # in clock cycles
    "num_points": 20,
    "add_to_xaxis": 2 * PI_LEN,
}
PARAMSS = [params_T1, params_T2, params_T2e]

for params in PARAMSS:

    if params["xscale"] == "log":

        params["t_delays"] = np.geomspace(params["tau_min"], params["tau_max"], params["num_points"]).astype(int)

    else:
        params["t_delays"] = np.arange(params["tau_min"], params["tau_max"], params["step_tau"])
    
    params["xaxis"] = 4 * params["t_delays"] + params["add_to_xaxis"]

params_T2["delta_phase"] = 4e-09 * params_T2["freq_detuning"] * params_T2["step_tau"]


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
resonators = [key for key in RR_CONSTANTS.keys()]

# Assertion
# for qb in qubits:
#     assert QUBIT_CONSTANTS[qb]["drag_coefficient"] != 0, "The DRAG coefficient 'drag_coef' must be different from 0 in the config."
# Check that the DRAG coefficient is not 0

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "n_avg": n_avg,
    "qubits": qubits,
    "params_T1": params_T1,
    "params_T2": params_T2,
    "params_T2e": params_T2e,
    "config": config,
}


###################
#   QUA program   #
###################

# T1
def make_QUA_PROGRAM_T1(qb="q1_xy", rr="q1_rr"):
    with program() as T1:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        state = declare(bool)
        n_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        state_st = declare_stream()
        t = declare(int)  # QUA variable for the wait time

        with for_(n, 0, n < n_avg, n + 1):
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

            with for_each_(t, params_T1["t_delays"]):

                play("x180", qb)
                wait(t, qb)

                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
                # Measure the state of the resonators
                measure(
                    "readout",
                    rr,
                    None,
                    dual_demod.full(weights + "cos", weights + "sin", I),
                    dual_demod.full(weights + "minus_sin", weights + "cos", Q),
                )
                assign(state, I > RR_CONSTANTS[rr]["ge_threshold"])
                
                # Save
                save(I, I_st)
                save(Q, Q_st)
                save(state, state_st)
                # Wait for the qubit to decay to the ground state
                wait(qb_reset_time >> 2)

        with stream_processing():
            n_st.save("iteration")
            I_st.buffer(len(params_T1["t_delays"])).average().save(f"I_{rr}")
            Q_st.buffer(len(params_T1["t_delays"])).average().save(f"Q_{rr}")
            state_st.boolean_to_int().buffer(len(params_T1["t_delays"])).average().save(f"state_{rr}")

    return T1


# T2
def make_QUA_PROGRAM_T2(qb="q1_xy", rr="q1_rr"):
    with program() as T2:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        state = declare(bool)
        n_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        state_st = declare_stream()
        t = declare(int)  # QUA variable for the wait time
        d_phase = declare(fixed, value=params_T2["delta_phase"])
        phase = declare(fixed, value=0)

        with for_(n, 0, n < n_avg, n + 1):
            # Save the averaging iteration to get the progress bar
            save(n, n_st)
            assign(phase, 0)

            with for_(*from_array(t, params_T2["t_delays"])):

                assign(phase, phase + d_phase)

                play("x90", qb)
                wait(t, qb)
                frame_rotation_2pi(phase, qb)
                play("x90", qb)  # 2nd x90 gate

                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
                # Measure the state of the resonators
                measure(
                    "readout",
                    rr,
                    None,
                    dual_demod.full(weights + "cos", weights + "sin", I),
                    dual_demod.full(weights + "minus_sin", weights + "cos", Q),
                )
                assign(state, I > RR_CONSTANTS[rr]["ge_threshold"])
                
                # Save
                save(I, I_st)
                save(Q, Q_st)
                save(state, state_st)
                # Reset frame
                reset_frame(qb)
                # Wait for the qubit to decay to the ground state
                wait(qb_reset_time >> 2)

        with stream_processing():
            n_st.save("iteration")
            I_st.buffer(len(params_T2["t_delays"])).average().save(f"I_{rr}")
            Q_st.buffer(len(params_T2["t_delays"])).average().save(f"Q_{rr}")
            state_st.boolean_to_int().buffer(len(params_T2["t_delays"])).average().save(f"state_{rr}")

    return T2


# T2e
def make_QUA_PROGRAM_T2e(qb="q1_xy", rr="q1_rr"):
    with program() as T2e:
        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        state = declare(bool)
        n_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        state_st = declare_stream()
        t = declare(int)  # QUA variable for the wait time

        with for_(n, 0, n < n_avg, n + 1):
            # Save the averaging iteration to get the progress bar
            save(n, n_st)

            with for_each_(t, params_T2e["t_delays"]):

                play("x90", qb)
                wait(t >> 1, qb)
                play("x180", qb)
                wait(t >> 1, qb)
                play("x90", qb)  # 2nd x90 gate

                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()
                # Measure the state of the resonators
                measure(
                    "readout",
                    rr,
                    None,
                    dual_demod.full(weights + "cos", weights + "sin", I),
                    dual_demod.full(weights + "minus_sin", weights + "cos", Q),
                )
                assign(state, I > RR_CONSTANTS[rr]["ge_threshold"])
                
                # Save
                save(I, I_st)
                save(Q, Q_st)
                save(state, state_st)
                # Wait for the qubit to decay to the ground state
                wait(qb_reset_time >> 2)

        with stream_processing():
            n_st.save("iteration")
            I_st.buffer(len(params_T2e["t_delays"])).average().save(f"I_{rr}")
            Q_st.buffer(len(params_T2e["t_delays"])).average().save(f"Q_{rr}")
            state_st.boolean_to_int().buffer(len(params_T1["t_delays"])).average().save(f"state_{rr}")

    return T2e


if __name__ == "__main__":
    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

    ###########################
    # Run or Simulate Program #
    ###########################

    simulate = False

    if simulate:
        program_to_execute = T1
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, program_to_execute, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show(block=False)
    else:
        try:
            PROGRAM_FUNCS = [make_QUA_PROGRAM_T1, make_QUA_PROGRAM_T2, make_QUA_PROGRAM_T2e]
            PROGRAM_NAMES = ["T1", "T2", "T2e"]
            SAVE_DIR.mkdir(exist_ok=True)
            default_additional_files = {"configuration_mw_fem.py": "configuration_mw_fem.py"}
            # Open the quantum machine
            qm = qmm.open_qm(config)

            n_rep = 0
            while True:
                # Increment
                n_rep += 1

                fig = plt.figure()
                for PROGRAM_FUNC, PROGRAM_NAME, PARAMS in zip(PROGRAM_FUNCS, PROGRAM_NAMES, PARAMSS):
                    
                    plt.suptitle(f"Multiplexed {PROGRAM_NAME} - state")

                    try:
                        for ind, (qb, rr) in enumerate(zip(qubits, resonators)):
                            # Save results
                            prog = PROGRAM_FUNC(qb, rr)
                            job = qm.execute(prog)
                            data_handler = DataHandler(root_data_folder=SAVE_DIR)
                            fetch_names = [
                                "iteration",
                                f"I_{rr}",
                                f"Q_{rr}",
                                f"state_{rr}",
                            ]
                            # Tool to easily fetch results from the OPX (results_handle used in it)
                            results = fetching_tool(job, fetch_names)
                            # Prepare the figure for live plotting
                            interrupt_on_close(fig, job)
                            # Data analysis and plotting
                            num_resonators = len(resonators)
                            num_rows = math.ceil(math.sqrt(num_resonators))
                            num_cols = math.ceil(num_resonators / num_rows)
                            # Fetch results
                            res = results.fetch_all()

                            I, Q, state = res[1], res[2], res[3]
                            save_data_dict[f"I_{rr}"] = res[1]
                            save_data_dict[f"Q_{rr}"] = res[2]
                            save_data_dict[f"state_{rr}"] = res[3]

                            # Plot
                            plt.subplot(num_rows, num_cols, ind + 1)
                            plt.cla()
                            plt.plot(PARAMS["xaxis"], I, color="r" if qb in qubits else "b")
                            plt.title(qb if qb in qubits else f"{qb}: NOT PLAYED")
                            plt.xlabel("Idle time [ns]")
                            plt.ylabel("I [a.u.]")

                        plt.tight_layout()

                        save_data_dict["fig_live"] = fig
                        # Repetition count
                        save_data_dict["repetition"] = n_rep
                        # Current time & 
                        curr_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        save_data_dict["current_time"] = curr_time

                        # Fit the data
                        try:
                            from qualang_tools.plot.fitting import Fit
                            fig_analyses = []
                            for qb, rr in zip(qubits, resonators):
                                I = save_data_dict[f"I_{rr}"]
                                fit = Fit()
                                fig_analysis = plt.figure(figsize=(6,6))
                                if PROGRAM_NAME in ["T1", "T2e"]:
                                    decay_fit = fit.T1(PARAMS["xaxis"], I, plot=True)
                                    qubit_T = np.round(np.abs(decay_fit["T1"][0]) / 4) * 4
                                elif PROGRAM_NAME == "T2":
                                    ramsey_fit = fit.ramsey(PARAMS["xaxis"], I, plot=True)
                                    qubit_T = np.abs(ramsey_fit["T2"][0])
                                plt.xlabel("Idle times [ns]")
                                plt.ylabel("I [a.u.]")
                                plt.legend((f"{PROGRAM_NAME} = {qubit_T:.0f} ns",))
                                print(f"{curr_time}: {n_rep}th {qb} {PROGRAM_NAME} = {qubit_T:.0f} nsec")
                                plt.title(f"{PROGRAM_NAME} measurement of {qb}")
                                save_data_dict[f"{PROGRAM_NAME}_{qb}_{n_rep:05d}"] = qubit_T
                                save_data_dict.update({f"fig_analysis_{qb}": fig_analysis})
                                fig_analyses.append(fig_analysis)
                        
                        except:
                            print(f"fitting failed at {n_rep}th {PROGRAM_NAME}")
                            save_data_dict[f"{PROGRAM_NAME}_{qb}"] = None
                            save_data_dict[f"fig_analysis_{qb}"] = None
                        
                        finally:
                            # plt.show(block=False)
                            time.sleep(5)
                            # Save data
                            script_name = Path(__file__).name
                            data_handler.additional_files = {script_name: script_name}
                            data_handler.save_data(data=save_data_dict, name=f"repeated_{PROGRAM_NAME}_{n_rep:05d}")
                            # Close figures
                            for fig_ in fig_analyses:
                                plt.close(fig_)
                            time.sleep(5)

                    except Exception as e:
                        print(f"An exception occurred at {n_rep}th {PROGRAM_NAME}: {e}")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")

# %%