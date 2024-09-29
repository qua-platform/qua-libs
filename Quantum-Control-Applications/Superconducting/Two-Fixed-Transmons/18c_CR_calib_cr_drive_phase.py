# %%
"""
                                 CR_calib_cr_drive_phase

The CR_calib scripts are designed for calibrating cross-resonance (CR) gates involving a system
with a control qubit and a target qubit. These scripts help estimate the parameters of a Hamiltonian,
which is represented as:
    H = I ⊗ (a_X X + a_Y Y + a_Z Z) + Z ⊗ (b_X X + b_Y Y + b_Z Z)

For the calibration sequences, we employ echoed CR drive.
                                   ____      ____ 
            Control(fC): _________| pi |____| pi |________________
                             ____                     
                 CR(fT): ___| CR |_____      _____________________
                                       |____|     _____
             Target(fT): ________________________| QST |__________
                                                         ______
            Readout(fR): _____________________ _________|  RR  |__

This script is to calibrate the phase of CR drive.
Each sequence, which varies in the duration and phase of the CR drive, ends with
state tomography of the target state (across X, Y, and Z bases).
This process is repeated with the control state in both |0> and |1> states.
We fit the two sets of CR duration versus tomography data to a theoretical model,
yielding two sets of three parameters: delta, omega_x, and omega_y.
Using these parameters, we estimate the interaction coefficients of the Hamiltonian.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Find the phase where b_Y (coeff of Z_Y) is zero. We call it phi0.
      Set cr_drive_phase = phi0 in the configuration file.
      Note that the phase is in units of 2 * pi as it is used with `frame_rotation_2pi`.

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
from cr_hamiltonian_tomography import (
    CRHamiltonianTomographyAnalysis, plot_cr_duration_vs_scan_param, 
    plot_interaction_coeffs, plot_crqst_result_3D, PAULI_2Q
)

import matplotlib

matplotlib.use('TkAgg')



##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 10
cr_type = "direct+cancel+echo" # "direct" "direct+cancel", "direct+cancel+echo"
cr_drive_amp = 1.0 # ratio
cr_drive_phase = 0.0 # in units of 2pi
cr_cancel_amp =  0.1 # ratio
cr_cancel_phase = 0.0 # in units of 2pi
ts_cycles = np.arange(4, 100, 1) # in clock cylcle = 4ns
phases = np.arange(0.0, 1.01, 0.05) # ratio relative to 2 * pi

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
ts_ns = 4 * ts_cycles # in clock cylcle = 4ns
# cr_drive_phase = CR_DRIVE_CONSTANTS[cr_drive]["square_phase"] # in units of 2pi
# cr_cancel_phase = CR_CANCEL_CONSTANTS[cr_drive]["square_phase"] # in units of 2pi

# Assertion
# assert n_avg <= 10_000, "revise your number of shots"
# assert np.all(ts_cycles % 2 == 0) and (ts_cycles.min() >= 4), "ts_cycles should only have even numbers if play echoes"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "ts_ns": ts_ns,
    "phases": phases,
    "n_avg": n_avg,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]
    t = declare(int)
    t_half = declare(int)
    ph = declare(fixed)
    ph_st = declare_stream()
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(*from_array(ph, phases)):
        save(ph, ph_st)
        wait(60 * u.ms)
        with for_(n, 0, n < n_avg, n + 1):
            with for_(*from_array(t, ts_cycles)):
                with for_(c, 0, c < 3, c + 1): # bases 
                    with for_(s, 0, s < 2, s + 1): # states
                        with if_(s == 1):
                            play("x180", qc_xy)
                            align(qc_xy, cr_drive)

                        if cr_type == "direct":
                            align(qc_xy, cr_drive)
                            play("square_positive", cr_drive, duration=t)
                            align(qt_xy, cr_drive)
                        
                        elif cr_type == "direct+cancel":
                            # phase shift for cancel drive
                            frame_rotation_2pi(ph, cr_drive)
                            frame_rotation_2pi(cr_cancel_phase, cr_cancel)
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
                            frame_rotation_2pi(ph, cr_drive)
                            frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                            # direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_positive" * amp(cr_drive_amp), cr_drive, duration=t)
                            play("square_positive" * amp(cr_cancel_amp), cr_cancel, duration=t)
                            # pi pulse on control
                            align(qc_xy, cr_drive, cr_cancel)
                            play("x180", qc_xy)
                            # echoed direct + cancel
                            align(qc_xy, cr_drive, cr_cancel)
                            play("square_negative" * amp(cr_drive_amp), cr_drive, duration=t)
                            play("square_negative" * amp(cr_cancel_amp), cr_cancel, duration=t)
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
                                wait(PI_LEN * u.ns, qt_xy)

                        align(qt_xy, *resonators)

                        # Measure the state of the resonators
                        multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)

                        # reset phase shift for cancel drive
                        reset_frame(cr_drive)

                        # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                        if reset_method == "wait":
                            wait(qb_reset_time >> 2)
                        elif reset_method == "active":
                            global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

    with stream_processing():
        ph_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(phases)).save(f"I_{rr}")
            Q_st[ind].buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(phases)).save(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).buffer(n_avg).map(FUNCTIONS.average()).buffer(len(phases)).save(f"state_{rr}") 


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
            fig, axss = plt.subplots(3, 4, figsize=(12, 9), sharex=True, sharey=True)
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
                    save_data_dict[f"I_{rr}"] = u.demod2volts(res[3*ind + 1], READOUT_LEN)
                    save_data_dict[f"Q_{rr}"] = u.demod2volts(res[3*ind + 2], READOUT_LEN)
                    save_data_dict[rr+"_state"] = res[3*ind + 3]
                iterations, _, _, state_c, _, _, state_t = res
                ph_idx = np.where(np.isclose(phases, iterations, atol=1e-5))[0][0]
                # Progress bar
                progress_counter(ph_idx, n_avg, start_time=results.start_time)
            # calculate the elapsed time
            elapsed_time = time.time() - start_time
            # plotting data
            # control qubit
            plot_cr_duration_vs_scan_param(-2 * state_c + 1, -2 * state_t + 1, ts_ns, phases, "phase [2pi]", axss)
            plt.tight_layout()
            plt.pause(2)

            save_data_dict["fig_live"] = fig

            # Perform CR Hamiltonian tomography
            coeffs = []
            fig_analyses = []
            for ph in range(len(phases)):
                print("-" * 40)
                print(f"fitting for phase = {phases[ph]}")
                try:
                    crht = CRHamiltonianTomographyAnalysis(
                        ts=ts_ns,
                        data=-2 * state_t[ph, ...] + 1, # target data: len(phases) x len(t_vec_cycle) x 3 x 2
                    )
                    crht.fit_params()
                    coeffs.append(crht.interaction_coeffs_MHz)
                    fig_analysis = crht.plot_fit_result(do_show=False)
                    save_data_dict[f"fig_analysis_phase={phases[ph]:5.4f}".replace(".","-")] = fig_analysis
                except:
                    print(f"fitting failed at phase = {phases[ph]}")
                    crht.interaction_coeffs_MHz = {k: None for k, v in crht.interaction_coeffs_MHz.items()}
                    phases[ph] = None
                    coeffs.append({p: None for p in PAULI_2Q})
                finally:
                    # plt.show(block=False)
                    # save_data_dict["fig_analysis"] = fig_analysis
                    save_data_dict[f"ham_tomo_params_fitted_@phase={phases[ph]:5.4f}"] = crht.params_fitted_dict
                    save_data_dict[f"ham_tomo_interaction_coeffs_MHz_@phase={phases[ph]:5.4f}"] = crht.interaction_coeffs_MHz

            # Plot the estimated interaction coefficients
            fig_summary = plot_interaction_coeffs(coeffs, phases, xlabel="cr phase")
            save_data_dict["fig_summary"] = fig_summary

            # # plot 3D
            # fig_3d = plot_crqst_result_3D(ts_ns, -2 * state_t + 1)
    
            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            data_handler.save_data(data=save_data_dict, name="cr_calib_ham_tomo_direct_vs_phase")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%