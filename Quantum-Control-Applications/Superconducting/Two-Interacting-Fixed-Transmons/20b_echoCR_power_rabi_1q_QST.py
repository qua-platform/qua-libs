"""
        echo-Cross-Resonance Power Rabi with single-qubit Quantum State Tomography
    The sequence consists two consecutive pulse sequences with the qubit's thermal decay in between.
In the first sequence, we set the control qubit in |g> and play a rectangular echo-cross-resonance pulse to
the target qubit; the echo-cross-resonance pulse has a variable amplitude. In the second sequence, we initialize the control
qubit in |e> and play the variable amplitude echo-cross-resonance pulse to the target qubit. At the end of both
sequences we perform single-qubit Quantum State Tomography on the target qubit.

To recreate the echo-cross-resonance pulse we play (CR--x180_c--CR)--x180_c if the control was initialized in |g>, or
(CR--x180_c--CR) if the control was initialized in |e>. The second x180_c in the first sequence guarantees that the
target qubit is at |g> in the limit of CR amplitude -> zero.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Reference: A. D. Corcoles et al., Phys. Rev. A 87, 030301 (2013)

"""
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from macros import qua_declaration, multiplexed_readout, one_qb_QST, plot_1qb_tomography_results
import warnings

warnings.filterwarnings("ignore")


###################
# The QUA program #
###################
amplitudes = np.arange(0.0, 1.9, 0.1)
n_avg = 1000

with program() as CR_power_rabi_one_qst:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    c = declare(int)  # QUA variable for the projection index in QST

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(a, amplitudes)):
            with for_(c, 0, c < 3, c + 1):
                # |0> control - CR
                play("square_positive" * amp(a), "cr_c1t2")
                align()
                play("x180", "q1_xy")
                align()
                play("square_negative" * amp(a), "cr_c1t2")
                align()
                play("x180", "q1_xy")
                align()
                one_qb_QST("q2_xy", pi_len, c)
                align()
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)

                align()  # global align

                # |1> control - CR
                play("x180", "q1_xy")
                align()
                play("square_positive" * amp(a), "cr_c1t2")
                align()
                play("x180", "q1_xy")
                align()
                play("square_negative" * amp(a), "cr_c1t2")
                align()
                one_qb_QST("q2_xy", pi_len, c)
                align()
                # Measure the state of the resonators
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2], weights="rotated_")
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("I1")
        Q_st[0].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("I2")
        Q_st[1].buffer(2).buffer(3).buffer(len(amplitudes)).average().save("Q2")

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
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CR_power_rabi_one_qst, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(CR_power_rabi_one_qst)
    # Prepare the figure for live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)
    # Tool to easily fetch results from the OPX (results_handle used in it)
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Plot tomography
        plot_1qb_tomography_results(I2, amplitudes)
    # Close the quantum machines at the end
    qm.close()
