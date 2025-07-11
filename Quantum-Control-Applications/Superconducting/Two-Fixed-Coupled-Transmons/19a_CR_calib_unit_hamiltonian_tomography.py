"""
        CR_calib_unit_hamiltonian_tomography

This script is to try Hamiltonian tomography for a set of specified parameters for CR drive and cancellation pulse.
This (unit) protocol will be repeated as a function of amplitude and phase of CR drive and cancellation pulse in the subsequent scripts.

The CR_calib scripts are designed for calibrating cross-resonance (CR) gates involving a system
with a control qubit and a target qubit. These scripts help estimate the parameters of a Hamiltonian,
which is represented as:
        H = I ⊗ (a_X X + a_Y Y + a_Z Z) + Z ⊗ (b_I I + b_X X + b_Y Y + b_Z Z)

The sequence extracts the six coefficients from a set of CR time Rabi traces:
two traces from the prepared control states (|g⟩ and |e⟩) and three traces from
applying quantum state tomography (QST) on the target qubit.

<<<<<<< HEAD:Quantum-Control-Applications/Superconducting/Two-Fixed-Coupled-Transmons/18a_CR_calib_unit_hamiltonian_tomography.py
For the calibration sequences, one need to choose one of the following CR drive configurations:
"direct," "direct + echo," "direct + cancel," or "direct + cancel + echo."
                                   ____      ____ 
=======
For the calibration sequences, we employ echoed CR drive.
                                   ____      ____
>>>>>>> main:Quantum-Control-Applications/Superconducting/Two-Fixed-Coupled-Transmons/19a_CR_calib_unit_hamiltonian_tomography.py
            Control(fC): _________| pi |____| pi |________________
                             ____
                 CR(fT): ___|    |_____      _____________________
                                       |____|     _____
             Target(fT): ________________________| QST |__________
                                                         ______
            Readout(fR): _______________________________|  RR  |__

Each sequence, which varies in the duration of the CR drive, ends with state tomography of the target state
(across X, Y, and Z bases). This process is repeated with the control state in both |0> and |1> states.
We fit the two sets of CR duration versus tomography data to a theoretical model,
yielding two sets of three parameters: delta, omega_x, and omega_y.
Using these parameters, we estimate the interaction coefficients of the Hamiltonian.
We consider this method a form of unit Hamiltonian tomography.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - This is only to test that you can obtain the data and fit to it successfully.

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
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler
from macros import qua_declaration, multiplexed_readout
from cr_hamiltonian_tomography import (
    CRHamiltonianTomographyAnalysis,
    plot_crqst_result_2D,
    plot_crqst_result_3D,
)

##################
#   Parameters   #
##################
# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Parameters Definition
n_avg = 200
cr_type = "direct+cancel+echo"  # "direct", "direct+echo", "direct+cancel", "direct+cancel+echo"
cr_drive_amp = 1.0  # ratio
cr_drive_phase = 0.0  # in units of 2pi
cr_cancel_amp = 0.5  # ratio
cr_cancel_phase = 0.0  # in units of 2pi
ts_cycles = np.arange(4, 400, 4)  # in clock cylcle = 4ns

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
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(t, ts_cycles)):
            with for_(c, 0, c < 3, c + 1):  # bases
                with for_(s, 0, s < 2, s + 1):  # states
                    with if_(s == 1):
                        play("x180", qc_xy)
                        align(qc_xy, cr_drive)

                    if cr_type == "direct":
                        align(qc_xy, cr_drive)
                        play("square_positive", cr_drive, duration=t)
                        align(qt_xy, cr_drive)

                    elif cr_type == "direct+cancel":
                        # phase shift for cancel drive
                        frame_rotation_2pi(cr_drive_phase, cr_drive)
                        frame_rotation_2pi(cr_cancel_phase, cr_cancel)
                        # direct + cancel
                        align(qc_xy, cr_drive, cr_cancel)
                        play("square_positive" * amp(cr_drive_amp), cr_drive, duration=t)
                        play("square_positive" * amp(cr_cancel_amp), cr_cancel, duration=t)
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
        I_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("Q1")
        state_st[0].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("state1")
        # target qubit
        I_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("Q2")
        state_st[1].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).average().save("state2")


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
        fig, axss = plt.subplots(4, 2, figsize=(10, 10))
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
            fig = plot_crqst_result_2D(ts_ns, bloch_c, bloch_t, fig, axss)
            plt.tight_layout()
            plt.pause(1)

        # cross resonance Hamiltonian tomography analysis
        crht = CRHamiltonianTomographyAnalysis(
            ts=ts_ns,
            data=bloch_t,  # target data
        )
        crht.fit_params()
        fig_analysis = crht.plot_fit_result(do_show=False)
        save_data_dict["I1"] = I1
        save_data_dict["Q1"] = Q1
        save_data_dict["I2"] = I2
        save_data_dict["Q2"] = Q2
        save_data_dict["fig_live"] = fig
        save_data_dict["fig_analysis"] = fig_analysis
        save_data_dict["ham_tomo_params_fitted"] = crht.params_fitted_dict
        save_data_dict["ham_tomo_interaction_coeffs_MHz"] = crht.interaction_coeffs_MHz

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
