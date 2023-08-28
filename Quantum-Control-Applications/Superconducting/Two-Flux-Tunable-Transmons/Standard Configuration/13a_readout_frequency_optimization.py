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

from qm.QuantumMachinesManager import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
from qualang_tools.results import fetching_tool
from macros import multiplexed_readout
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################
n_avg = 4000
dfs = np.arange(-0.5e6, 0.5e6, 0.02e6)

with program() as ro_freq_opt:
    n = declare(int)  # QUA variable for the averaging loop
    I_g = [declare(fixed) for _ in range(2)]  # QUA variable for the 'I' quadrature when the qubit is in |g>
    Q_g = [declare(fixed) for _ in range(2)]  # QUA variable for the 'Q' quadrature when the qubit is in |g>
    I_e = [declare(fixed) for _ in range(2)]  # QUA variable for the 'I' quadrature when the qubit is in |e>
    Q_e = [declare(fixed) for _ in range(2)]  # QUA variable for the 'Q' quadrature when the qubit is in |e>
    DI = declare(fixed)  # QUA variable for the distance between I in |g> abd |e>
    DQ = declare(fixed)  # QUA variable for the distance between Q in |g> abd |e>
    D = [declare(fixed) for _ in range(2)]  # QUA variable for the distance between the IQ blobs in |g> abd |e>
    df = declare(int)  # QUA variable for the readout frequency
    D_st = [declare_stream() for _ in range(2)]

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two resonator elements
            update_frequency("rr1", df + resonator_IF_q1)
            update_frequency("rr2", df + resonator_IF_q2)

            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the ground IQ blobs
            multiplexed_readout(I_g, None, Q_g, None, resonators=[1, 2], weights="rotated_")

            align()
            # Reset both qubits to ground
            wait(thermalization_time * u.ns)
            # Measure the excited IQ blobs
            play("x180", "q1_xy")
            play("x180", "q2_xy")
            align()
            multiplexed_readout(I_e, None, Q_e, None, resonators=[1, 2], weights="rotated_")

            # Derive the averaged distance between the blobs for the qubits in |g> and |e>
            for i in range(2):
                assign(DI, (I_e[i] - I_g[i]) * 100)
                assign(DQ, (Q_e[i] - Q_g[i]) * 100)
                assign(D[i], DI * DI + DQ * DQ)
                save(D[i], D_st[i])

    with stream_processing():
        D_st[0].buffer(len(dfs)).average().save("D1")
        D_st[1].buffer(len(dfs)).average().save("D2")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

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
    D1, D2 = results.fetch_all()
    # Plot
    plt.suptitle("Readout frequency optimization")
    plt.subplot(121)
    plt.plot(dfs / u.MHz, D1)
    plt.xlabel("Readout detuning [MHz]")
    plt.ylabel("Distance between IQ blobs [a.u.]")
    plt.title("Resonator 2")
    plt.subplot(122)
    plt.plot(dfs / u.MHz, D2)
    plt.xlabel("Readout detuning [MHz]")
    plt.title("Resonator 1")
    print(f"Shift readout frequency 1 by {dfs[np.argmax(D1)]} Hz")
    print(f"Shift readout frequency 2 by {dfs[np.argmax(D2)]} Hz")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
