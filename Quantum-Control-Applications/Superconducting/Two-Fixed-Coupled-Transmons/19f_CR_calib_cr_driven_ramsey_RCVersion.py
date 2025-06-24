"""
        CR_CALIB_CR_DRIVEN_RAMSEY

The CR_calib scripts are designed for calibrating cross-resonance (CR) gates involving a system
with a control qubit and a target qubit. These scripts help estimate the parameters of a Hamiltonian,
which is represented as:
    H = I ⊗ (a_X X + a_Y Y + a_Z Z) + Z ⊗ (b_I I + b_X X + b_Y Y + b_Z Z)

The sequence extracts the phase shift due to b_I  Z ⊗ I to enable its correction.
When incorporating the CR gate into a sequence of gates (and not at the end),
it is necessary to correct for this phase shift.

For the calibration sequences, we use one of the following CR drive configurations:
"direct," "direct + echo," "direct + cancel," or "direct + cancel + echo."

                                   _____           ______
                Control(fC): _____| y90 |_________| -y90 |___________
                                           ______
                     CR(fT): _____________|      |___________________
                                  ______  |  CR  |
                 Target(fT): ____| x180 |_|______|___________________
                                                   ______
                Readout(fR): _____________________|  RR  |___

This script performs a Ramsey experiment as a function of phase, where the idle time corresponds to the duration of the CR (cross-resonance) gate.
The CR gate induces a Stark shift in the control qubit due to the off-resonant drive, which we aim to correct.
The pulse sequence is repeated with the control qubit in both the |0⟩ and |1⟩ states to measure this effect.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having found the amplitudes and phases for cr drive and cr cancel.

Next steps before going to the next node:
    - Measure the phase shift between the |0⟩ and |1⟩ states, which will provide the correction phase.
      The shift is equal to -2 times the phase required to correct the Stark shift.
      Update:
        - cr_drive_square_phase_ZI_correct = -0.5 * phase shift

Reference: Sarah Sheldon, Easwar Magesan, Jerry M. Chow, and Jay M. Gambetta Phys. Rev. A 93, 060302(R) (2016)
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

##################
#   Parameters   #
##################
# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Parameters Definition
n_avg = 100
cr_type = "direct+cancel+echo"  # "direct", "direct+echo", "direct+cancel", "direct+cancel+echo"
cr_drive_amp = 1.0  # ratio
cr_drive_phase = 0.0  # in units of 2pi
cr_cancel_amp = 0.5  # ratio
cr_cancel_phase = 0.0  # in units of 2pi
ts_cycles = np.arange(4, 100, 1)  # in clock cylcle = 4ns
phases = np.arange(0.0, 1.01, 0.05)  # ratio relative to 2 * pi

# Derived parameters
qc_xy = f"q{qc}_xy"
qt_xy = f"q{qt}_xy"
cr_drive = f"cr_drive_c{qc}t{qt}"
cr_cancel = f"cr_cancel_c{qc}t{qt}"
qubits = [f"q{i}_xy" for i in [qc, qt]]
resonators = [f"rr{i}" for i in [qc, qt]]
ts_ns = 4 * ts_cycles  # in clock cylcle = 4ns

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "cr_drive_amp": cr_drive_amp,
    "cr_drive_phase": cr_drive_phase,
    "cr_cancel_amp": cr_cancel_amp,
    "cr_cancel_phase": cr_cancel_phase,
    "ts_ns": ts_ns,
    "phases": phases,
    "n_avg": n_avg,
    "config": config,
}


###################
# The QUA program #
###################
with program() as prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]
    s = declare(int)  # 0:s, 1:e for control state
    ph = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(ph, phases)):
            with for_(s, 0, s < 2, s + 1):  # states
                # Prepare Qt to |1>
                with if_(s == 1):
                    play("x180", qc_xy)
                with else_():
                    wait(pi_len // 4, qc_xy)

                # Bring Qc to x axis
                play("y90", qc_xy)

                align()

                if cr_type == "direct":
                    align(qc_xy, cr_drive)
                    play("square_positive", cr_drive, duration=ph)
                    align(qt_xy, cr_drive)

                elif cr_type == "direct+echo":
                    # phase shift for cancel drive
                    frame_rotation_2pi(cr_drive_phase, cr_drive)
                    # direct + cancel
                    align(qc_xy, cr_drive)
                    play("square_positive", cr_drive)
                    # pi pulse on control
                    align(qc_xy, cr_drive)
                    play("x180", qc_xy)
                    # echoed direct + cancel
                    align(qc_xy, cr_drive)
                    play("square_negative", cr_drive)
                    # pi pulse on control
                    align(qc_xy, cr_drive)
                    play("x180", qc_xy)
                    reset_frame(cr_drive)

                elif cr_type == "direct+cancel":
                    # phase shift for cancel drive
                    frame_rotation_2pi(cr_drive_phase, cr_drive)
                    frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                    # direct + cancel
                    align(qc_xy, cr_drive, cr_cancel)
                    play("square_positive", cr_drive)
                    play("square_positive", cr_cancel)
                    # align for the next step and clear the phase shift
                    align(qc_xy, cr_drive, cr_cancel)
                    reset_frame(cr_drive)
                    reset_frame(cr_cancel)

                elif cr_type == "direct+cancel+echo":
                    # phase shift for cancel drive
                    frame_rotation_2pi(cr_drive_phase, cr_drive)
                    frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                    # direct + cancel
                    align(qc_xy, cr_drive, cr_cancel)
                    play("square_positive", cr_drive)
                    play("square_positive", cr_cancel)
                    # pi pulse on control
                    align(qc_xy, cr_drive, cr_cancel)
                    play("x180", qc_xy)
                    # echoed direct + cancel
                    align(qc_xy, cr_drive, cr_cancel)
                    play("square_negative", cr_drive)
                    play("square_negative", cr_cancel)
                    # pi pulse on control
                    align(qc_xy, cr_drive, cr_cancel)
                    play("x180", qc_xy)
                    reset_frame(cr_drive)
                    reset_frame(cr_cancel)

                align()

                # phase shift
                frame_rotation(ph, qc_xy)

                # Bring Qc back to z axis
                play("-y90", qc_xy)

                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()

                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                assign(state[0], I[0] > ge_threshold_q1)
                save(state[0], state_st[0])
                assign(state[1], I[1] > ge_threshold_q2)
                save(state[1], state_st[1])

                # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # control qubit
        I_st[0].buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("I1")
        Q_st[0].buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("state1")
        # target qubit
        I_st[1].buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("I2")
        Q_st[1].buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(len(ts_cycles)).buffer(len(phases)).average().save("state2")


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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, prog, simulation_config)
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
        job = qm.execute(prog)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        fetch_names = ["n", "I1", "Q1", "state1", "I2", "Q2", "state2"]
        results = fetching_tool(job, fetch_names, mode="live")
        # Prepare the figure for live plotting
        fig, axss = plt.subplots(2, 3, figsize=(8, 6))
        interrupt_on_close(fig, job)
        # Live plotting
        while results.is_processing():
            # Fetch results
            res = results.fetch_all()
            iterations, I1, Q1, state_c, I2, Q2, state_t = res
            # Progress bar
            progress_counter(iterations, n_avg, start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            # Progress bar
            progress_counter(iterations, n_avg, start_time=results.start_time)
            # Live plot data
            fig.suptitle(f"Phase calibration for {qc_xy}")
            for ax, fname, V in zip(axss.ravel(), fetch_names[1:], res[1:]):
                ax.plot(phases, V[:, 0])
                ax.plot(phases, V[:, 1])
                ax.set_xlabel("Phase [2pi rad.]")
                ax.set_ylabel(fname.replace("1", "c").replace("2", "t"))
                ax.set_title(fname)
                ax.legend(["qc=|0>", "qc=|1>"])
            fig.tight_layout()
            plt.pause(1)

        # Save data
        save_data_dict.update({"fig_live": fig})
        for fname, r in zip(fetch_names[1:], res[1:]):
            save_data_dict[fname] = r

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
