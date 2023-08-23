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
    - Update the readout frequency (resonator_IF) in the configuration.
"""

from qm.qua import *
from qm import SimulationConfig
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
from qualang_tools.loops import from_array
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

n_avg = 1000  # The number of averages
# The frequency sweep parameters
f_min = 70e6
f_max = 80e6
df = 0.1e6
frequencies = np.arange(f_min, f_max + df / 2, df)  # + df/2 to add f_max to frequencies

with program() as ro_freq_opt:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency
    I_g = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |g>
    Q_g = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |g>
    Ig_st = declare_stream()
    Qg_st = declare_stream()
    I_e = declare(fixed)  # QUA variable for the 'I' quadrature when the qubit is in |e>
    Q_e = declare(fixed)  # QUA variable for the 'Q' quadrature when the qubit is in |e>
    Ie_st = declare_stream()
    Qe_st = declare_stream()
    n_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, frequencies)):
            # Update the frequency of the digital oscillator linked to the qubit element
            update_frequency("resonator", f)
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_g),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_g),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I_e' & 'Q_e' quadratures to their respective streams
            save(I_g, Ig_st)
            save(Q_g, Qg_st)

            align()  # global align
            # Play the x180 gate to put the qubit in the excited state
            play("x180", "qubit")
            # Align the two elements to measure after playing the qubit pulse.
            align("qubit", "resonator")
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I_e),
                dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q_e),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I_e' & 'Q_e' quadratures to their respective streams
            save(I_e, Ie_st)
            save(Q_e, Qe_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        # mean values
        Ig_st.buffer(len(frequencies)).average().save("Ig_avg")
        Qg_st.buffer(len(frequencies)).average().save("Qg_avg")
        Ie_st.buffer(len(frequencies)).average().save("Ie_avg")
        Qe_st.buffer(len(frequencies)).average().save("Qe_avg")
        # variances to get the SNR
        (
            ((Ig_st.buffer(len(frequencies)) * Ig_st.buffer(len(frequencies))).average())
            - (Ig_st.buffer(len(frequencies)).average() * Ig_st.buffer(len(frequencies)).average())
        ).save("Ig_var")
        (
            ((Qg_st.buffer(len(frequencies)) * Qg_st.buffer(len(frequencies))).average())
            - (Qg_st.buffer(len(frequencies)).average() * Qg_st.buffer(len(frequencies)).average())
        ).save("Qg_var")
        (
            ((Ie_st.buffer(len(frequencies)) * Ie_st.buffer(len(frequencies))).average())
            - (Ie_st.buffer(len(frequencies)).average() * Ie_st.buffer(len(frequencies)).average())
        ).save("Ie_var")
        (
            ((Qe_st.buffer(len(frequencies)) * Qe_st.buffer(len(frequencies))).average())
            - (Qe_st.buffer(len(frequencies)).average() * Qe_st.buffer(len(frequencies)).average())
        ).save("Qe_var")

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
    job = qm.execute(ro_freq_opt)  # execute QUA program
    # Get results from QUA program
    results = fetching_tool(
        job,
        data_list=["Ig_avg", "Qg_avg", "Ie_avg", "Qe_avg", "Ig_var", "Qg_var", "Ie_var", "Qe_var", "iteration"],
        mode="live",
    )
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        Ig_avg, Qg_avg, Ie_avg, Qe_avg, Ig_var, Qg_var, Ie_var, Qe_var, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Derive the SNR
        Z = (Ie_avg - Ig_avg) + 1j * (Qe_avg - Qg_avg)
        var = (Ig_var + Qg_var + Ie_var + Qe_var) / 4
        SNR = ((np.abs(Z)) ** 2) / (2 * var)
        # Plot results
        plt.cla()
        plt.plot(frequencies / u.MHz, SNR, ".-")
        plt.title("Readout optimization")
        plt.xlabel("Readout frequency [MHz]")
        plt.ylabel("SNR")
        plt.pause(0.1)
    print(f"The optimal readout frequency is {frequencies[np.argmax(SNR)]} Hz (SNR={max(SNR)})")
