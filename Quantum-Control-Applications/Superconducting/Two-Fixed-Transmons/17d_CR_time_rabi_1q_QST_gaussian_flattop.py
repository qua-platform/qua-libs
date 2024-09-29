# %%
"""
                                    Cross-Resonance Time Rabi
The sequence consists two consecutive pulse sequences with the qubit's thermal decay in between.
In the first sequence, we set the control qubit in |g> and play a rectangular cross-resonance pulse to
the target qubit; the cross-resonance pulse has a variable duration. In the second sequence, we initialize the control
qubit in |e> and play the variable duration cross-resonance pulse to the target qubit. Note that in
the second sequence after the cross-resonance pulse we send a x180_c pulse. With it, the target qubit starts
in |g> in both sequences when CR lenght -> zero.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Reference: A. D. Corcoles et al., Phys. Rev. A 87, 030301 (2013)

"""

from qm.qua import *
from qm import QuantumMachinesManager
# from configuration_mw_fem import *
from configuration_opxplus_without_octave import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout, active_reset
from qualang_tools.results.data_handler import DataHandler
import time
from qualang_tools.bakery import baking


##################
#   Parameters   #
##################

# Qubits and resonators 
qc = 1 # index of control qubit
qt = 2 # index of target qubit

# Parameters Definition
n_avg = 100

cr_flattop_amp = 0.2
cr_drive_amp = cr_flattop_amp
cr_cancel_amp = 0.6 * cr_flattop_amp
cr_flattop_len = 100
cr_gaussian_rise_fall_amp = cr_flattop_amp
cr_gaussian_rise_fall_len = 40
cr_gaussian_rise_fall_pad_len = 0 if cr_gaussian_rise_fall_len % 4 == 0 else (4 - cr_gaussian_rise_fall_len % 4)
cr_square_phase = 0.0
cr_flattop_phase = 0.0
cr_gaussian_flattop_phase = 0.0
# readout_gaussian_flattop_total_len
cr_gaussian_flattop_total_amp = cr_flattop_amp
cr_gaussian_flattop_total_len = 2 * cr_gaussian_rise_fall_len + cr_flattop_len

t_min_flattop = 4 // 4
t_max_flattop = 204 // 4
dt = 1 # 4 // 4
ts_cycles_flattop = np.arange(t_min_flattop, t_max_flattop, dt) # in clock cylcle = 4ns
ts_cycles = (2 * cr_gaussian_rise_fall_len // 4) + np.arange(t_min_flattop, t_max_flattop, dt) # in clock cylcle = 4ns

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
ts_ns_flattop = 4 * ts_cycles_flattop
ts_ns = 4 * ts_cycles # in clock cylcle = 4ns

cr_drive_constants = CR_DRIVE_CONSTANTS[cr_drive]
cr_cancel_constants = CR_CANCEL_CONSTANTS[cr_cancel]

# Assert
assert ts_ns.min() > 16, ""

# Data to save        
save_data_dict = {
    "qubits": qubits,
    "resonators": resonators,
    "qc_xy": qc_xy,
    "qt_xy": qt_xy,
    "cr_drive": cr_drive,
    "cr_cancel": cr_cancel,
    "ts_ns": ts_ns,
    "n_avg": n_avg,
    "config": config,
}


###################
#   QUA Bakery    #
###################

def get_flattop_gaussian(cr_amp, cr_flattop_len, cr_rise_fall_len, cr_rise_fall_pad_len):
    cr_lpad = [0] * cr_rise_fall_pad_len
    cr_rpad = cr_lpad
    cr_gf = flattop_gaussian_waveform(
        cr_amp,
        cr_flattop_len, # cr_constants["flattop_len"],
        cr_rise_fall_len,
        return_part="all",
    )
    cr_wf_I = cr_lpad + cr_gf + cr_rpad
    cr_wf_Q = [0.0] * len(cr_wf_I)
    return cr_wf_I, cr_wf_Q


baked_flattop_gaussians = []
for t in ts_ns_flattop:
    with baking(config, padding_method="right") as b:
        # generate wf
        cr_drive_wf_I, cr_drive_wf_Q = get_flattop_gaussian(
            cr_drive_amp,
            t, # cr_flattop_len,
            cr_gaussian_rise_fall_len,
            cr_gaussian_rise_fall_pad_len,
        )
        cr_cancel_wf_I, cr_cancel_wf_Q = get_flattop_gaussian(
            cr_cancel_amp,
            t, #cr_flattop_len,
            cr_gaussian_rise_fall_len,
            cr_gaussian_rise_fall_pad_len,
        )
        # add operations
        b.add_op("fg_drive", cr_drive, [cr_drive_wf_I, cr_drive_wf_Q])
        b.add_op("fg_cancel", cr_cancel, [cr_cancel_wf_I, cr_cancel_wf_Q])
        # play baked pulses
        b.play("fg_drive", cr_drive)
        b.play("fg_cancel", cr_cancel)
    baked_flattop_gaussians.append(b)


###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(resonators)
    state = [declare(bool) for _ in range(len(resonators))]
    state_st = [declare_stream() for _ in range(len(resonators))]
    ti = declare(int)
    s = declare(int)  # QUA variable for the control state
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(ti, 0, ti < len(ts_cycles), ti + 1):
            with for_(c, 0, c < 3, c + 1): # bases 
                with for_(s, 0, s < 2, s + 1): # states

                    with if_(s == 1):
                        play("x180", qc_xy)
                        align(qc_xy, cr_drive)

                    # switch case to select the baked waveform corresponding to the burst duration
                    with switch_(ti, unsafe=False):
                        for ii in range(len(ts_cycles)):
                            with case_(ii):
                                baked_flattop_gaussians[ii].run()

                    align(cr_drive, qt_xy)
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

                    # Wait for the qubit to decay to the ground state - Can be replaced by active reset
                    if reset_method == "wait":
                        # wait(qb_reset_time >> 2)
                        wait(200 >> 2)
                    elif reset_method == "active":
                        global_state = active_reset(I, None, Q, None, state, None, resonators, qubits, state_to="ground", weights=weights)

    with stream_processing():
        n_st.save("iteration")
        for ind, rr in enumerate(resonators):
            I_st[ind].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save(f"I_{rr}")
            Q_st[ind].buffer(2).buffer(3).buffer(len(ts_cycles)).average().save(f"Q_{rr}")
            state_st[ind].boolean_to_int().buffer(2).buffer(3).buffer(len(ts_cycles)).average().save(f"state_{rr}")


if __name__ == "__main__":
    #####################################
    #  Open Communication with the QOP  #
    #####################################
    qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

    ###########################
    # Run or Simulate Program #
    ###########################
    simulate = True

    if simulate:
        # Simulates the QUA program for the specified duration
        simulation_config = SimulationConfig(duration=200)  # In clock cycles = 4ns
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
            fig, axss = plt.subplots(3, 2, figsize=(8, 8), sharex=True)
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
                    save_data_dict[rr+"_state"] = res[3*ind + 3]
                iterations, I1, Q1, state1, I2, Q2, state2 = res

                # Progress bar
                progress_counter(iterations, n_avg, start_time=results.start_time)
                # calculate the elapsed time
                elapsed_time = time.time() - start_time
                # Convert the results into Volts
                I1, Q1 = u.demod2volts(I1, READOUT_LEN), u.demod2volts(Q1, READOUT_LEN)
                I2, Q2 = u.demod2volts(I2, READOUT_LEN), u.demod2volts(Q2, READOUT_LEN)
                # Plots
                plt.suptitle("echo CR Time Rabi")
                for i, (axs, bss) in enumerate(zip(axss, ["X", "y", "z"])):
                    for ax, q in zip(axs, ["c", "t"]):
                        I = I1 if q == "c" else I2
                        ax.cla()
                        for j, st in enumerate(["0", "1"]):
                            ax.plot(ts_ns, I[:, i, j], label=[f"|{st}>"])
                        ax.legend(["0", "1"])
                        ax.set_title(f"Q_{q}") if i == 0 else None
                        ax.set_xlabel("cr durations [ns]") if i == 2 else None
                        ax.set_ylabel(f"I quadrature of <{bss}> [V]") if q == "c" else None
                plt.tight_layout()
                plt.pause(2)

            # Save results
            script_name = Path(__file__).name
            data_handler = DataHandler(root_data_folder=save_dir)
            save_data_dict.update({"fig_live": fig})
            data_handler.additional_files = {script_name: script_name, **default_additional_files}
            data_handler.save_data(data=save_data_dict, name="cr_time_rabi")

        except Exception as e:
            print(f"An exception occurred: {e}")

        finally:
            qm.close()
            print("Experiment QM is now closed")
            plt.show(block=True)

# %%
