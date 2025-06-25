"""
        CNOT

This script measures two-qubit states by preparing the states |00⟩, |01⟩, |10⟩, and |11⟩, then applying a CNOT gate.
The CNOT gate is constructed using a calibrated CR (cross-resonance) gate, which serves as the ZX interaction, along with single-qubit gates Z_{-90}I and IX_{-90}.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having calibrated cr drive and cr cancel by running Hamiltonian Tomography

"""

from qm.qua import *
from qm import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler
from macros import qua_declaration, multiplexed_readout
import pandas as pd


##################
#   Parameters   #
##################
# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Parameters Definition
n_shots = 10_000

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
cr_drive = f"cr_drive_c{qc}t{qt}"
cr_cancel = f"cr_cancel_c{qc}t{qt}"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"rr{i}" for i in [qc, qt]]

cr_drive_amp = cr_drive_square_amp_c1t2
cr_drive_phase = cr_drive_square_phase_c1t2
cr_cancel_amp = cr_cancel_square_amp_c1t2
cr_cancel_phase = cr_cancel_square_phase_c1t2
cr_drive_phase_ZI_correct = cr_drive_square_phase_ZI_correct_c1t2

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
    "cr_drive_phase_ZI_correct": cr_drive_phase_ZI_correct,
    "n_shots": n_shots,
    "config": config,
}

###################
# The QUA program #
###################
with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]
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
                        wait(pi_len * u.ns)
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
                    frame_rotation_2pi(+0.25, qc_xy)  # +0.25 for Z(-pi/2)
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

                    # Shift frames to the calibrated phases
                    frame_rotation_2pi(cr_drive_phase + cr_drive_phase_ZI_correct, cr_drive)
                    frame_rotation_2pi(cr_cancel_phase, cr_cancel)

                    align(qc_xy, qt_xy, *resonators)

                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                assign(state[0], I[0] > ge_threshold_q1)
                save(state[0], state_st[0])
                assign(state[1], I[1] > ge_threshold_q2)
                save(state[1], state_st[1])

                # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("iteration")
        # control qubit
        I_st[0].buffer(2).buffer(4).save_all("I1")
        Q_st[0].buffer(2).buffer(4).save_all("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(4).save_all("state1")
        # target qubit
        I_st[1].buffer(2).buffer(4).save_all("I2")
        Q_st[1].buffer(2).buffer(4).save_all("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(4).save_all("state2")


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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PROGRAM, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

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
        fetch_names = ["iteration", "I1", "Q1", "state1", "I2", "Q2", "state2"]
        results = fetching_tool(job, fetch_names, mode="live")
        # Live plotting
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            iterations, I1, Q1, state_c, I2, Q2, state_t = res
            # Progress bar
            progress_counter(iterations, n_shots, start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)

        # Save data
        save_data_dict.update({"fig_live": fig})
        for fname, r in zip(fetch_names[1:], res[1:]):
            save_data_dict[fname] = r

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
            ax.bar(index - width / 2, grouped.loc["N"], width, label="cnot=No")

            # Bar plot for cnot=False
            ax.bar(index + width / 2, grouped.loc["Y"], width, label="cnot=Yes")

            # Add some text for labels, title, and axes ticks
            ax.set_xlabel("res")
            ax.set_ylabel("Count")
            ax.set_title(f"Counts of states in case of preparing {case}")
            ax.set_xticks(index)
            ax.set_xticklabels(grouped.columns)
            ax.legend()

        plt.tight_layout()
        save_data_dict.update({"fig_analysis": fig_analysis})

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
