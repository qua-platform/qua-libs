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
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - Set the desired flux bias

Next steps before going to the next node:
    - Update the readout frequency (f_opt) in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
from macros import multiplexed_readout
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Build the config
config = build_config(machine)
# The resonator frequencies
res_if_1 = rr1.f_opt - machine.local_oscillators.readout[rr1.LO_index].freq
res_if_2 = rr2.f_opt - machine.local_oscillators.readout[rr2.LO_index].freq

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# The frequency sweep parameters with respect to the resonators resonance frequencies
dfs = np.arange(-2e6, 2e6, 0.02e6)

with program() as ro_freq_opt:
    n = declare(int)
    I_g = [declare(fixed) for _ in range(2)]
    Q_g = [declare(fixed) for _ in range(2)]
    I_e = [declare(fixed) for _ in range(2)]
    Q_e = [declare(fixed) for _ in range(2)]
    DI = declare(fixed)
    DQ = declare(fixed)
    D = [declare(fixed) for _ in range(2)]
    df = declare(int)
    D_st = [declare_stream() for _ in range(2)]

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the resonator frequencies
            update_frequency(rr1.name, df + res_if_1)
            update_frequency(rr2.name, df + res_if_2)

            # Wait for the qubit to decay to the ground state
            wait(cooldown_time * u.ns)
            align()
            # Measure the state of the resonators
            multiplexed_readout(I_g, None, Q_g, None, resonators=active_qubits, weights="rotated_")

            align()
            # Wait for thermalization again in case of measurement induced transitions
            wait(cooldown_time * u.ns)
            # Play the x180 gate to put the qubits in the excited state
            play("x180", qb1.name + "_xy")
            play("x180", qb2.name + "_xy")
            # Align the elements to measure after playing the qubit pulses.
            align()
            # Measure the state of the resonator
            multiplexed_readout(I_e, None, Q_e, None, resonators=active_qubits, weights="rotated_")

            # Derive the distance between the blobs for |g> and |e>
            for i in range(len(active_qubits)):
                assign(DI, (I_e[i] - I_g[i]) * 100)
                assign(DQ, (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI * DI + DQ * DQ)
                save(D[i], D_st[i])

    with stream_processing():
        for i in range(len(active_qubits)):
            D_st[i].buffer(len(dfs)).average().save(f"D{i+1}")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

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
    results = fetching_tool(job, ["D1", "D2"])
    # fetch data
    D1, D2 = results.fetch_all()
    # Plot the results
    plt.subplot(211)
    plt.plot(dfs, D1)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{qb1.name} - f_opt = {int(rr1.f_opt / u.MHz)} MHz")
    plt.subplot(212)
    plt.plot(dfs, D2)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title(f"{qb2.name} - f_opt = {int(rr2.f_opt / u.MHz)} MHz")
    plt.tight_layout()
    print(f"{rr1.name}: Shift readout frequency by {dfs[np.argmax(D1)]} Hz")
    print(f"{rr2.name}: Shift readout frequency by {dfs[np.argmax(D2)]} Hz")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Update the state
    rr1.f_opt = dfs[np.argmax(D1)] + res_if_1 + machine.local_oscillators.readout[rr1.LO_index].freq
    rr2.f_opt = dfs[np.argmax(D2)] + res_if_2 + machine.local_oscillators.readout[rr2.LO_index].freq
    # machine._save("quam_bootstrap_state.json")
