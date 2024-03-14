"""
        READOUT OPTIMISATION: FREQUENCY
This sequence involves measuring the state of the resonator in two scenarios: first, after thermalization
(with the qubit in the |g> state) and then after applying a pi pulse to the qubit (transitioning the qubit to the
|e> state). This is done while varying the readout frequency.
The average I & Q quadratures for the qubit states |g> and |e>, along with their variances, are extracted to
determine the Signal-to-Noise Ratio (SNR). The readout frequency that yields the highest SNR is selected as the
optimal choice.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF_q) in the configuration.
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool, progress_counter
from macros import multiplexed_readout, qua_declaration


###################
# The QUA program #
###################
n_avg = 4000
# The frequency sweep around the resonators' frequency "resonator_IF_q"
dfs = np.arange(-10e6, 10e6, 0.1e6)

with program() as ro_freq_opt:
    Ig, Ig_st, Qg, Qg_st, n, n_st = qua_declaration(nb_of_qubits=2)
    Ie, Ie_st, Qe, Qe_st, _, _ = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the readout frequency

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two resonator elements
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the ground IQ blobs
            multiplexed_readout(Ig, Ig_st, Qg, Qg_st, resonators=[1, 2], weights="rotated_")

            align()
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the excited IQ blobs
            play("x180", "q1_xy")
            play("x180", "q2_xy")
            align()
            multiplexed_readout(Ie, Ie_st, Qe, Qe_st, resonators=[1, 2], weights="rotated_")
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        for i in range(2):
            # mean values
            Ig_st[i].buffer(len(dfs)).average().save(f"Ig{i}_avg")
            Qg_st[i].buffer(len(dfs)).average().save(f"Qg{i}_avg")
            Ie_st[i].buffer(len(dfs)).average().save(f"Ie{i}_avg")
            Qe_st[i].buffer(len(dfs)).average().save(f"Qe{i}_avg")
            # variances to get the SNR
            (
                ((Ig_st[i].buffer(len(dfs)) * Ig_st[i].buffer(len(dfs))).average())
                - (Ig_st[i].buffer(len(dfs)).average() * Ig_st[i].buffer(len(dfs)).average())
            ).save(f"Ig{i}_var")
            (
                ((Qg_st[i].buffer(len(dfs)) * Qg_st[i].buffer(len(dfs))).average())
                - (Qg_st[i].buffer(len(dfs)).average() * Qg_st[i].buffer(len(dfs)).average())
            ).save(f"Qg{i}_var")
            (
                ((Ie_st[i].buffer(len(dfs)) * Ie_st[i].buffer(len(dfs))).average())
                - (Ie_st[i].buffer(len(dfs)).average() * Ie_st[i].buffer(len(dfs)).average())
            ).save(f"Ie{i}_var")
            (
                ((Qe_st[i].buffer(len(dfs)) * Qe_st[i].buffer(len(dfs))).average())
                - (Qe_st[i].buffer(len(dfs)).average() * Qe_st[i].buffer(len(dfs)).average())
            ).save(f"Qe{i}_var")

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
    job = qmm.simulate(config, ro_freq_opt, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(ro_freq_opt)
    # Get results from QUA program
    data_list = [
        "Ig1_avg",
        "Qg1_avg",
        "Ie1_avg",
        "Qe1_avg",
        "Ig1_var",
        "Qg1_var",
        "Ie1_var",
        "Qe1_var",
        "Ig2_avg",
        "Qg2_avg",
        "Ie2_avg",
        "Qe2_avg",
        "Ig2_var",
        "Qg2_var",
        "Ie2_var",
        "Qe2_var",
        "iteration",
    ]
    results = fetching_tool(job, data_list=data_list, mode="live")
    (
        Ig1_avg,
        Qg1_avg,
        Ie1_avg,
        Qe1_avg,
        Ig1_var,
        Qg1_var,
        Ie1_var,
        Qe1_var,
        Ig2_avg,
        Qg2_avg,
        Ie2_avg,
        Qe2_avg,
        Ig2_var,
        Qg2_var,
        Ie2_var,
        Qe2_var,
        iteration,
    ) = results.fetch_all()
    # Progress bar
    progress_counter(iteration, n_avg, start_time=results.get_start_time())
    # Derive the SNR
    Z1 = (Ie1_avg - Ig1_avg) + 1j * (Qe1_avg - Qg1_avg)
    var1 = (Ig1_var + Qg1_var + Ie1_var + Qe1_var) / 4
    SNR1 = ((np.abs(Z1)) ** 2) / (2 * var1)
    Z2 = (Ie2_avg - Ig2_avg) + 1j * (Qe2_avg - Qg2_avg)
    var2 = (Ig2_var + Qg2_var + Ie2_var + Qe2_var) / 4
    SNR2 = ((np.abs(Z2)) ** 2) / (2 * var2)
    # Plot results
    plt.suptitle("Readout frequency optimization")
    plt.subplot(121)
    plt.cla()
    plt.plot(dfs / u.MHz, SNR1, ".-")
    plt.title(f"Qubit 1 around {resonator_IF_q1 / u.MHz} MHz")
    plt.xlabel("Readout frequency detuning [MHz]")
    plt.ylabel("SNR")
    plt.grid("on")
    plt.subplot(121)
    plt.cla()
    plt.plot(dfs / u.MHz, SNR2, ".-")
    plt.title(f"Qubit 2 around {resonator_IF_q2 / u.MHz} MHz")
    plt.xlabel("Readout frequency detuning [MHz]")
    plt.grid("on")
    plt.pause(0.1)
    print(f"The optimal readout frequency is {dfs[np.argmax(SNR1)] + resonator_IF_q1} Hz (SNR={max(SNR1)})")
    print(f"The optimal readout frequency is {dfs[np.argmax(SNR2)] + resonator_IF_q2} Hz (SNR={max(SNR2)})")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
