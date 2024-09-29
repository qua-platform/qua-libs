# %%
"""
                                 CNOT

Prerequisites:
    - 

Next steps before going to the next node:
    - 

Reference: Sarah Sheldon, Easwar Magesan, Jerry M. Chow, and Jay M. Gambetta Phys. Rev. A 93, 060302(R) (2016)
"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration_mw_fem import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
from qualang_tools.results.data_handler import DataHandler
import time
import warnings
import matplotlib
from macros import qua_declaration, multiplexed_readout

matplotlib.use('TkAgg')

##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_shots = 10_000
cr_type = "direct+cancel+echo" # "direct+cancel", "direct+cancel+echo"

# Readout Parameters
weights = "rotated_" # ["", "rotated_", "opt_"]
reset_method = "wait" # ["wait", "active"]
readout_operation = "readout" # ["readout", "midcircuit_readout"]

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
cr_drive = f"cr_drive_c{qc}t{qt}"
cr_cancel = f"cr_cancel_c{qc}t{qt}"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"q{i}_rr" for i in [qc, qt]]

cr_drive_amp = CR_DRIVE_CONSTANTS[cr_drive]["square_positive_amp"]
cr_drive_phase = CR_DRIVE_CONSTANTS[cr_drive]["square_positive_phase"]
cr_cancel_amp = CR_CANCEL_CONSTANTS[cr_cancel]["square_positive_amp"] # ratio
cr_cancel_phase = CR_CANCEL_CONSTANTS[cr_cancel]["square_positive_phase"] # in units of 2pi
cr_drive_len = CR_DRIVE_CONSTANTS[cr_drive]["square_positive_len"]
cr_cancel_len = CR_CANCEL_CONSTANTS[cr_cancel]["square_positive_len"]

# Assertion
# assert n_shots <= 10_000, "revise your number of shots"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "cr_cancel_amp": cr_cancel_amp,
    "cr_drive_amp": cr_drive_amp,
    "cr_cancel_phase": cr_cancel_phase,
    "cr_drive_phase": cr_drive_phase,
    "n_shots": n_shots,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]
    c = declare(int)
    with_cnot = declare(int)
    
    with for_(n, 0, n < n_shots, n + 1):
        save(n, n_st)
        # to allow time to save the data
        wait(1000 * u.ns)
        with for_(c, 0, c < 4, c + 1):
            with for_(with_cnot, 0, with_cnot < 2, with_cnot + 1):
                # QST on Target
                with switch_(c):
                    with case_(0):  # projection along X
                        wait(PI_LEN * u.ns)
                    with case_(1):  # projection along Y
                        play("x180", qt_xy)
                        align()
                    with case_(2):  # projection along Z
                        play("x180", qc_xy)
                        align()
                    with case_(3):  # projection along Z
                        play("x180", qc_xy)
                        play("x180", qt_xy)
                        align()

                with if_(with_cnot == 1):
                    # Play ZI(-pi/2) and IX(-pi/2)
                    frame_rotation_2pi(+0.25, qc_xy) # +0.25 for Z(-pi/2)
                    play("-x90", qt_xy)

                    # Shift frames to the calibrated phases
                    frame_rotation_2pi(cr_drive_phase, cr_drive)
                    frame_rotation_2pi(cr_cancel_phase, cr_cancel)

                    # Play CR
                    align(qc_xy, qt_xy, cr_drive, cr_cancel)
                    play("square_positive", cr_drive)
                    play("square_positive", cr_cancel)
                    # echo
                    align(qc_xy, cr_drive, cr_cancel)
                    play("x180", qc_xy)
                    align(qc_xy, cr_drive, cr_cancel)
                    play("square_negative", cr_drive)
                    play("square_negative", cr_cancel)
                    align(qc_xy, cr_drive, cr_cancel)
                    play("x180", qc_xy)
        
                    align(qc_xy, qt_xy, *resonators)

                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)

                # reset phase shift for cancel drive
                reset_frame(cr_drive)
                reset_frame(cr_cancel)

                align()

                # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                if reset_method == "wait":
                    wait(qb_reset_time >> 2)
                elif reset_method == "active":
                    global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(2).buffer(4).save_all(f"I_{rr}")
            Q_st[ind].buffer(2).buffer(4).save_all(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(2).buffer(4).save_all(f"state_{rr}")


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
        simulation_config = SimulationConfig(duration=3_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, PROGRAM, simulation_config)
        job.get_simulated_samples().con1.plot(analog_ports=['1', '2', '3', '4', '5', '6'])
        plt.show()

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
            fetch_names = ["iteration"]
            for rr in resonators:
                fetch_names.append(f"I_{rr}")
                fetch_names.append(f"Q_{rr}")
                fetch_names.append(f"state_{rr}")
            results = fetching_tool(job, fetch_names, mode="live")
            # Live plotting
            while results.is_processing():
                start_time = results.get_start_time()
                # Fetch results
                res = results.fetch_all()
                for ind, rr in enumerate(resonators):
                    save_data_dict[f"I_{rr}"] = res[3*ind + 1]
                    save_data_dict[f"Q_{rr}"] = res[3*ind + 2]
                    save_data_dict[f"state_{rr}"] = res[3*ind + 3]
                iterations, I1, Q1, state_c, I2, Q2, state_t = res
                # Progress bar
                progress_counter(iterations, n_shots, start_time=results.start_time)
                # # Plot
                # plt.suptitle("CNOT calib")
                # plt.subplot(1, 2, 1)
                # plt.cla()
                # plt.imshow(state_c, interpolation='nearest', cmap='Blues')
                # plt.xlabel("prepared target state")
                # plt.ylabel("prepared control state")
                # plt.title(f"measured control state")
                # plt.colorbar()
                # plt.subplot(1, 2, 2)
                # plt.cla()
                # plt.imshow(state_t, interpolation='nearest', cmap='Blues')
                # plt.xlabel("prepared target state")
                # plt.ylabel("prepared control state")
                # plt.title(f"measured target state")
                # plt.colorbar()
                plt.tight_layout()
                plt.pause(2)

            import pandas as pd

            data = []
            for i in range(2):
                cnot = {0: "N", 1: "Y"}[i]
                for j in range(4):
                    case = {0: "00", 1: "01", 2: "10", 3: "11"}[j]
                    for k in range(n_shots):
                        c, t = state_c[k, j, i], state_t[k, j, i]
                        data.append([cnot, case, f"{c}{t}"])
            df = pd.DataFrame(data, columns=["cnot", "case", "res"])

            # Plot bar plot for side-by-side comparison
            fig_analysis, axss = plt.subplots(2, 2, figsize=(10, 8))
            cases = ["00", "01", "10", "11"]
            for ax, case in zip(axss.ravel(), cases):
                case_data = df[df["case"] == case]  # Filter data for the current "case"
                
                # Group by "cnot" and "res" and count occurrences
                grouped = case_data.groupby(["cnot", "res"]).size().unstack(fill_value=0)

                
                # Generate side-by-side bar plot
                width = 0.35  # Width of bars
                index = np.arange(len(grouped.columns))  # The x locations for the groups
                
                # Bar plot for cnot=True
                ax.bar(index - width/2, grouped.loc["N"], width, label='cnot=No')
                
                # Bar plot for cnot=False
                ax.bar(index + width/2, grouped.loc["Y"], width, label='cnot=Yes')

                # Add some text for labels, title, and axes ticks
                ax.set_xlabel('res')
                ax.set_ylabel('Count')
                ax.set_title(f'Counts of states in case of preparing {case}')
                ax.set_xticks(index)
                ax.set_xticklabels(grouped.columns)
                ax.legend()
            plt.tight_layout()
            save_data_dict.update({"fig_analysis": fig_analysis})

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="cnot_calib")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            # qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
