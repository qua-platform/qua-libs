"""
<<<<<<< HEAD:Quantum-Control-Applications/Superconducting/Two-Fixed-Coupled-Transmons/18e_CR_calib_cr_cancel_amplitude.py
        CR_calib_cancel_drive_amplitude
This script is to calibrate the phase of CR cancellation drive, corresponding to Fig. 3(b) of the referenced paper.
CR drive (cancellation) pulse is applied to the control(target) qubit at the target qubit frequency.
Each sequence, which varies in the duration of the CR drive and the phase of CR cancellation drive,
=======
                                 CR_calib_cancel_drive_amplitude

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
>>>>>>> main:Quantum-Control-Applications/Superconducting/Two-Fixed-Coupled-Transmons/19e_CR_calib_cr_cancel_amplitude.py
ends with state tomography of the target state (across X, Y, and Z bases).
This process is repeated with the control state in both |0> and |1> states.
We fit the two sets of CR duration versus tomography data to a theoretical model,
yielding two sets of three parameters: delta, omega_x, and omega_y.
Using these parameters, we estimate the interaction coefficients of the Hamiltonian.
(a_X, a_Y, a_Z, b_X, b_Y, b_Z described in the 18a_CR_calib_unit_hamiltonian_tomography.py)

For the calibration sequences, one needs to choose one of the following CR drive configurations:
cr_type = "direct," "direct + echo," "direct + cancel," or "direct + cancel + echo."

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Find the amplitude where a_X (coeff of I_X) and a_Y (coeff of I_Y) is zero simultaneously.
      Set cr_cancel_square_amp_c1t2 in the configuration file.

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
from macros import qua_declaration, multiplexed_readout, active_reset
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
amp_scalings = np.arange(0.5, 1.01, 0.05)  # scaling factor for amplitude

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
    "amp_scalings": amp_scalings,
    "n_avg": n_avg,
    "config": config,
}


###################
# The QUA program #
###################
with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    state = [declare(bool) for _ in range(2)]
    state_st = [declare_stream() for _ in range(2)]
    t = declare(int)
    a = declare(fixed)
    counter = declare(int, value=0)
    counter_st = declare_stream()
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(*from_array(a, amp_scalings)):
        save(counter, counter_st)
        wait(60 * u.ms)
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(t, ts_cycles)):
                with for_(c, 0, c < 3, c + 1):  # bases
                    with for_(s, 0, s < 2, s + 1):  # states
                        with if_(s == 1):
                            play("x180", qc_xy)
                            align(qc_xy, cr_drive)

                        if cr_type == "direct+cancel":
                            # phase shift for cancel drive
                            frame_rotation_2pi(cr_drive_phase, cr_drive)
                            frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                            # direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_positive" * amp(cr_drive_amp), cr_drive, duration=t)
                            play("square_positive" * amp(a), cr_cancel, duration=t)
                            # align for the next step and clear the phase shift
                            align(qt_xy, cr_drive, cr_cancel)
                            reset_frame(cr_drive)
                            reset_frame(cr_cancel)

                        elif cr_type == "direct+cancel+echo":
                            # phase shift for cancel drive
                            frame_rotation_2pi(cr_drive_phase, cr_drive)
                            frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                            # direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_positive" * amp(cr_drive_amp), cr_drive, duration=t)
                            play("square_positive" * amp(a), cr_cancel, duration=t)
                            # pi pulse on control
                            align(qc_xy, cr_drive, cr_cancel)
                            play("x180", qc_xy)
                            # echoed direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_negative" * amp(cr_drive_amp), cr_drive, duration=t)
                            play("square_negative" * amp(a), cr_cancel, duration=t)
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
        assign(counter, counter + 1)
    with stream_processing():
        counter_st.save("n")
        # control qubit
        I_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(amp_scalings)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(amp_scalings)).average().save("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(
            len(amp_scalings)
        ).average().save("state1")
        # target qubit
        I_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(amp_scalings)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(len(amp_scalings)).average().save("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(
            len(amp_scalings)
        ).average().save("state2")


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
            progress_counter(iterations, len(amp_scalings), start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            bloch_c, bloch_t = -2 * state_c + 1, -2 * state_t + 1  # convert |0> -> 1, |1> -> -1
            # Progress bar
            progress_counter(iterations, n_avg, start_time=results.start_time)

        # plotting data
        fig = plot_cr_duration_vs_scan_param(bloch_c, bloch_t, ts_ns, amp_scalings, "cr cancel amplitude", axss)
        plt.tight_layout()
        # plt.pause(1)

        # Save data
        save_data_dict.update({"fig_live": fig})
        for fname, r in zip(fetch_names[1:], res[1:]):
            save_data_dict[fname] = r

        # Perform CR Hamiltonian tomography
        coeffs = []
        fig_analyses = []
        for idx, a in enumerate(amp_scalings):
            print("-" * 40)
            print(f"fitting for amp = {a}")
            try:
                crht = CRHamiltonianTomographyAnalysis(
                    ts=ts_ns,
                    data=bloch_t[idx, ...],  # target data: len(amp_scalings) x len(t_vec_cycle) x 3 x 2
                )
                crht.fit_params()
                coeffs.append(crht.interaction_coeffs_MHz)
                fig_analysis = crht.plot_fit_result(do_show=False)
                save_data_dict[f"fig_analysis_amp_scaling={a:5.4f}".replace(".", "-")] = fig_analysis
            except:
                print(f"-> failed")
                crht.interaction_coeffs_MHz = {k: None for k, v in crht.interaction_coeffs_MHz.items()}
                coeffs.append({p: None for p in PAULI_2Q})
            finally:
                save_data_dict[f"ham_tomo_params_fitted_@amp={a:5.4f}"] = crht.params_fitted_dict
                save_data_dict[f"ham_tomo_interaction_coeffs_MHz_@amp={a:5.4f}"] = crht.interaction_coeffs_MHz

        # Plot the estimated interaction coefficients
        fig_summary = plot_interaction_coeffs(coeffs, amp_scalings, xlabel="cr cancel amplitude")
        save_data_dict["fig_summary"] = fig_summary

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
