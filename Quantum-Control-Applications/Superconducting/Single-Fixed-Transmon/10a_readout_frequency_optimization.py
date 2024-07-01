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

Next steps before going to the next node:
    - Update the readout frequency (resonator_IF) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt


###################
# The QUA program #
###################
n_avg = 100  # The number of averages
# The frequency sweep parameters
span = 10 * u.MHz
df = 200 * u.kHz
dfs = np.arange(-span, +span + 0.1, df)

with program() as ro_freq_opt:
    n = declare(int)  # QUA variable for the averaging loop
    df = declare(int)  # QUA variable for the readout frequency
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
        with for_(*from_array(df, dfs)):
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency("resonator", df + resonator_IF)
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I_g),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q_g),
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
                dual_demod.full("rotated_cos", "rotated_sin", I_e),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q_e),
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
        Ig_st.buffer(len(dfs)).average().save("Ig_avg")
        Qg_st.buffer(len(dfs)).average().save("Qg_avg")
        Ie_st.buffer(len(dfs)).average().save("Ie_avg")
        Qe_st.buffer(len(dfs)).average().save("Qe_avg")
        # variances to get the SNR
        (
            ((Ig_st.buffer(len(dfs)) * Ig_st.buffer(len(dfs))).average())
            - (Ig_st.buffer(len(dfs)).average() * Ig_st.buffer(len(dfs)).average())
        ).save("Ig_var")
        (
            ((Qg_st.buffer(len(dfs)) * Qg_st.buffer(len(dfs))).average())
            - (Qg_st.buffer(len(dfs)).average() * Qg_st.buffer(len(dfs)).average())
        ).save("Qg_var")
        (
            ((Ie_st.buffer(len(dfs)) * Ie_st.buffer(len(dfs))).average())
            - (Ie_st.buffer(len(dfs)).average() * Ie_st.buffer(len(dfs)).average())
        ).save("Ie_var")
        (
            ((Qe_st.buffer(len(dfs)) * Qe_st.buffer(len(dfs))).average())
            - (Qe_st.buffer(len(dfs)).average() * Qe_st.buffer(len(dfs)).average())
        ).save("Qe_var")

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
        plt.plot(dfs / u.MHz, SNR, ".-")
        plt.title(f"Readout frequency optimization around {resonator_IF / u.MHz} MHz")
        plt.xlabel("Readout frequency detuning [MHz]")
        plt.ylabel("SNR")
        plt.grid("on")
        plt.pause(0.1)
    print(f"The optimal readout frequency is {dfs[np.argmax(SNR)] + resonator_IF} Hz (SNR={max(SNR)})")
