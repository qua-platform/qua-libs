"""
                                 CR_calib_cancel_drive_phase

The CR_calib scripts are designed for calibrating cross-resonance (CR) gates involving a system
with a control qubit and a target qubit. These scripts help estimate the parameters of a Hamiltonian,
which is represented as:
    H = I ⊗ (a_X X + a_Y Y + a_Z Z) + Z ⊗ (b_I I + b_X X + b_Y Y + b_Z Z)


For the calibration sequences, we employ echoed CR drive.
                                   ____      ____ 
            Control(fC): _________| pi |____| pi |________________
                             ____                     
                 CR(fT): ___|    |_____      _____________________
                                       |____|     _____
             Target(fT): ________________________| QST |__________
                                                         ______
            Readout(fR): _____________________ _________|  RR  |__

This script is to calibrate the phase of CR cancellation drive.
CR cancellation pulse is applied to the target qubit at the target qubit frequency.
Each sequence, which varies in the duration of the CR drive and the phase of CR cancel drive,
ends with state tomography of the target state (across X, Y, and Z bases).
This process is repeated with the control state in both |0> and |1> states.
We fit the two sets of CR duration versus tomography data to a theoretical model,
yielding two sets of three parameters: delta, omega_x, and omega_y.
Using these parameters, we estimate the interaction coefficients of the Hamiltonian.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Find the phase to shift for the CR cancel drive via phi = arctan(IY/IX).
      Alternatively, find the phase where a_Y (coeff of I_Y) is zero. We call it phi1.
      phi = phi0 - phi1.
      Note that the phase is in units of 2 * pi as it is used with `frame_rotation_2pi`.

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
from cr_hamiltonian_tomography import (
    CRHamiltonianTomographyAnalysis,
    plot_cr_duration_vs_scan_param,
    plot_interaction_coeffs,
    PAULI_2Q,
)


##################
#   Parameters   #
##################

# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Parameters Definition
n_avg = 10
cr_type = "direct+cancel+echo"  # "direct" "direct+cancel", "direct+cancel+echo"
cr_drive_amp = 1.0  # ratio
cr_drive_phase = 0.0  # in units of 2pi
cr_cancel_amp = 0.5  # ratio
cr_cancel_phase = 0.0  # in units of 2pi
ts_cycles = np.arange(4, 100, 1)  # in clock cylcle = 4ns
phases = np.arange(0.0, 1.01, 0.05)  # ratio relative to 2 * pi

# Readout Parameters
weights = "rotated_"  # ["", "rotated_", "opt_"]

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
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]
    t = declare(int)
    ph = declare(fixed)
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(ph, phases)):
            with for_(*from_array(t, ts_cycles)):
                with for_(c, 0, c < 3, c + 1):  # bases
                    with for_(s, 0, s < 2, s + 1):  # states
                        with if_(s == 1):
                            play("x180", qc_xy)
                            align(qc_xy, cr_drive)

                        if cr_type == "direct+cancel":
                            # phase shift for cancel drive
                            frame_rotation_2pi(cr_drive_phase, cr_drive)
                            frame_rotation_2pi(ph, cr_cancel)
                            # direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_positive", cr_drive, duration=t)
                            play("square_positive", cr_cancel, duration=t)
                            # align for the next step and clear the phase shift
                            align(qt_xy, cr_drive, cr_cancel)
                            reset_frame(cr_drive)
                            reset_frame(cr_cancel)

                        elif cr_type == "direct+cancel+echo":
                            # phase shift for cancel drive
                            frame_rotation_2pi(cr_drive_phase, cr_drive)
                            frame_rotation_2pi(ph, cr_cancel)
                            # direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_positive", cr_drive, duration=t)
                            play("square_positive", cr_cancel, duration=t)
                            # pi pulse on control
                            align(qc_xy, cr_drive, cr_cancel)
                            play("x180", qc_xy)
                            # echoed direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_negative", cr_drive, duration=t)
                            play("square_negative", cr_cancel, duration=t)
                            # pi pulse on control
                            align(qc_xy, cr_drive, cr_cancel)
                            play("x180", qc_xy)
                            # align for the next step and clear the phase shift
                            align(qc_xy, qt_xy)
                            reset_frame(cr_drive)
                            reset_frame(cr_cancel)

                        # QST on Target
                        with switch_(c):
                            with case_(0):  # projection along X
                                play("-y90", qt_xy)
                            with case_(1):  # projection along Y
                                play("x90", qt_xy)
                            with case_(2):  # projection along Z
                                wait(pi_len * u.ns, qt_xy)

                        align(qt_xy, *resonators)

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
        I_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save(
            "state1"
        )
        # target qubit
        I_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(phases)).average().save(
            "state2"
        )


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
    job.get_simulated_samples().con1.plot()
    plt.show()

else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        # Prepare the figure for live plotting
        fig, axss = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
        interrupt_on_close(fig, job)
        # Tool to easily fetch results from the OPX (results_handle used in it)
        fetch_names = ["n", "I1", "Q1", "state1", "I2", "Q2", "state2"]
        results = fetching_tool(job, fetch_names, mode="live")
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
            bloch_c, bloch_t = -2 * state_c + 1, -2 * state_t + 1  # convert |0> -> 1, |1> -> -1
            # Progress bar
            progress_counter(iterations, n_avg, start_time=results.start_time)
            # plotting data
            fig = plot_cr_duration_vs_scan_param(bloch_c, bloch_t, ts_ns, phases, "cr cancel phase [2pi]", axss)
            plt.tight_layout()
            plt.pause(1)

        # Save data
        save_data_dict.update({"fig_live": fig})
        for fname, r in zip(fetch_names[1:], res[1:]):
            save_data_dict[fname] = r

        # Perform CR Hamiltonian tomography
        coeffs = []
        fig_analyses = []
        for idx, ph in enumerate(phases):
            print("-" * 40)
            print(f"fitting for phase = {ph}")
            try:
                crht = CRHamiltonianTomographyAnalysis(
                    ts=ts_ns,
                    data=bloch_t[idx, ...],  # target data: len(phases) x len(t_vec_cycle) x 3 x 2
                )
                crht.fit_params()
                coeffs.append(crht.interaction_coeffs_MHz)
                fig_analysis = crht.plot_fit_result(do_show=False)
                save_data_dict[f"fig_analysis_phase={ph:5.4f}".replace(".", "-")] = fig_analysis
            except:
                print(f"fitting failed at phase = {ph}")
                crht.interaction_coeffs_MHz = {k: None for k, v in crht.interaction_coeffs_MHz.items()}
                coeffs.append({p: None for p in PAULI_2Q})
            finally:
                # plt.show(block=False)
                # save_data_dict["fig_analysis"] = fig_analysis
                save_data_dict[f"ham_tomo_params_fitted_@phase={ph:5.4f}"] = crht.params_fitted_dict
                save_data_dict[f"ham_tomo_interaction_coeffs_MHz_@phase={ph:5.4f}"] = crht.interaction_coeffs_MHz

        # Plot the estimated interaction coefficients
        fig_summary = plot_interaction_coeffs(coeffs, phases, xlabel="cr cancel phase")
        save_data_dict["fig_summary"] = fig_summary

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        data_handler.save_data(data=save_data_dict, name="cr_calib_ham_tomo_cancel_vs_phase")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
