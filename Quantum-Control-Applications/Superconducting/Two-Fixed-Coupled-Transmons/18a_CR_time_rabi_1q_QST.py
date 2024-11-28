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
from configuration import *
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import progress_counter
from macros import qua_declaration, multiplexed_readout
from qualang_tools.results.data_handler import DataHandler


##################
#   Parameters   #
##################

# Qubits and resonators
qc = 1  # index of control qubit
qt = 2  # index of target qubit

# Parameters Definition
n_avg = 100
cr_drive_amp = 0.8  # ratio
cr_drive_phase = 0.5  # in units of 2pi
ts_cycles = np.arange(4, 400, 4)  # in clock cylcle = 4ns

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
    "ts_ns": ts_ns,
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

                    # Control
                    play("square_positive", cr_drive, duration=t)

                    align(cr_drive, qt_xy)
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
    job = qmm.simulate(config, PROGRAM, simulation_config)
    job.get_simulated_samples().con1.plot()

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
        fetch_names = ["n", "I1", "Q1", "state1", "I2", "Q2", "state2"]
        results = fetching_tool(job, fetch_names, mode="live")
        # Live plotting
        while results.is_processing():
            # Fetch results
            iterations, I1, Q1, state1, I2, Q2, state2 = results.fetch_all()
            # Progress bar
            progress_counter(iterations, n_avg, start_time=results.start_time)
            # Convert the results into Volts
            I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
            I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
            # Plots
            plt.suptitle("non-echo CR Time Rabi")
            for i, (axs, bss) in enumerate(zip(axss, ["X", "Y", "Z"])):
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
            plt.pause(1)

        # Save live plot
        save_data_dict.update({"fig_live": fig})

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="cr_time_rabi")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
