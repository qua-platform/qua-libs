# %%
"""
        CZ CALIB CZ PULSE VS LOCAL PHASE
The CZ calibration scripts are designed to calibrate the CZ gate by compenstating the phases for ZI and IZ.
CZ = exp(- i/2 * pi/2 * (- ZI - IZ + ZZ)) and CZ pulse = exp(- i/2 * (a * ZI + b * IZ + pi/2 * ZZ)) without phase compensations.
By adding phases phi_ZI and phi_IZ to ZI and IZ, CZ <- exp(- i/2 * ((a + phi_ZI) * ZI + (b + phi_IZ) * IZ + pi/2 * ZZ)) 
Namely, we want to compensate the phases such that it forms CZ.

    a + phi_ZI = -pi/2
    b + phi_IZ = -pi/2

The pulse sequences are as follow:
                                   _____                    ______
                Control(fC): _____| y90 |__________________| -y90 |___________
                                          ______  ________                    
   ZZ_control (fT-detuning): ____________|  ZZ  || phi_ZI |___________________
                                  ______  ______  ________ 
    ZZ_target (fT-detuning): ____| x180 ||  ZZ  || phi_IZ |___________________
                                                                     ______
                Readout(fR): _______________________________________|  RR  |___

This script measures entanglement as a function of phi_ZI (by flipping the role of qc and qz: phi_IZ), replicating Fig. S2(b) of the referenced paper.
The pulse sequence is repeated with the control qubit in both the |0⟩ and |1⟩ states.
The optimal phi_ZI and phi_IZ are selected where a + phi_ZI = -pi/2 and b + phi_IZ = -pi/2.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having found the frequency, amplitudes and relative phase shift of zz_control and zz_target.

Next steps before going to the next node:
    - Pick phi_ZI and phi_IZ such that (a + phi_ZI) = -pi/2 and (b + phi_IZ) = -pi/2 and update the config for
        - ZZ_CONTROL_CONSTANTS["zz_control_c{qc}t{qt}"]["square_phi_ZI"]
        - ZZ_TARGET_CONSTANTS["zz_target_c{qc}t{qt}"]["square_phi_IZ"]

Reference: Bradley K. Mitchell, et al, Phys. Rev. Lett. 127, 200502 (2021)
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


##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 4 # index of control qubit
qt = 3 # index of target qubit

# Parameters Definition
n_avg = 100
cr_type = "direct+cancel+echo" # "direct+cancel", "direct+cancel+echo"
phases = np.arange(0, 2, 0.25) # ratio relative to 2 * pi
cr_drive_phase = 0.0
cr_cancel_phase = 0.0

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

# Assertion
assert n_avg <= 10_000, "revise your number of shots"

# Data to save
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "phases": phases,
    "n_avg": n_avg,
    "config": config,
}


###################
#   QUA Program   #
###################

with program() as prog:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]
    s = declare(int) # 0:s, 1:e for control state
    ph = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

        with for_(*from_array(ph, phases)):

            with for_(s, 0, s < 2, s + 1): # states

                # Prepare Qt to |1>
                with if_(s == 1):
                    play("x180", qc_xy)
                with else_():
                    wait(PI_LEN >> 2, qc_xy)

                # Bring Qc to x axis
                play("y90", qc_xy)

                align()

                if cr_type == "direct":
                    align(qc_xy, cr_drive)
                    play("square_positive", cr_drive, duration=ph)
                    align(qt_xy, cr_drive)
                
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

                # phase shift
                frame_rotation(ph, qc_xy)

                # Bring Qc back to z axis                
                play("-y90", qc_xy)

                # Align the elements to measure after having waited a time "tau" after the qubit pulses.
                align()

                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, state, state_st, resonators=resonators, weights=weights)

                # Wait for the qubit to decay to the ground state
                if reset_method == "wait":
                    wait(qb_reset_time >> 2)
                elif reset_method == "active":
                    global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)


    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(2).buffer(len(phases)).average().save(f"I_{rr}")
            Q_st[ind].buffer(2).buffer(len(phases)).average().save(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(2).buffer(len(phases)).average().save(f"state_{rr}")


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
        simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
        job = qmm.simulate(config, prog, simulation_config)
        job.get_simulated_samples().con1.plot()
        plt.show(block=False)
    else:
        try:
            # Open the quantum machine
            qm = qmm.open_qm(config)
            # Send the QUA program to the OPX, which compiles and executes it
            job = qm.execute(prog)
            fetch_names = ["iteration"]
            for rr in resonators:
                fetch_names.append(f"I_{rr}")
                fetch_names.append(f"Q_{rr}")
                fetch_names.append(f"state_{rr}")
            # Tool to easily fetch results from the OPX (results_handle used in it)
            results = fetching_tool(job, fetch_names, mode="live")
            # Prepare the figure for live plotting
            fig, axss = plt.subplots(2, 1, figsize=(8, 6))
            interrupt_on_close(fig, job)
            # Live plotting
            while results.is_processing():
                # Fetch results
                res = results.fetch_all()
                iteration, Ic, Qc, Sc, It, Qt, St = res
                Ic, Qc = u.demod2volts(Ic, READOUT_LEN), u.demod2volts(Qc, READOUT_LEN)
                Vnames = ["Ic", "Qc"]
                Vs = [Ic, Qc]

                # Progress bar
                progress_counter(iteration, n_avg, start_time=results.start_time)

                # Live plot data
                fig.suptitle(f"Phase calibration for {qc_xy}")
                for ax, Vname, V, fname in zip(axss.ravel(), Vnames, Vs, fetch_names[1:3]):
                    ax.plot(phases, V[:, 0])
                    ax.plot(phases, V[:, 1])
                    ax.set_xlabel("Phase [2pi rad.]")
                    ax.set_ylabel(f"{Vname} [V]")
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
            data_handler.save_data(data=save_data_dict, name="cr_calib_cr_driven_ramsey_vs_phase")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
