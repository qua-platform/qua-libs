"""
        PARITY MEASUREMENT
This sequence involves two consecutive measurements:
The first one measures the background, by applying a displacement pulse, then apply x90 - wait time - -x90 and measure the resonator, while sweeping over the idle time.
The second one starts by applying a displacement pulse, then apply x90 - wait time - -x90 and then measure by applying a selective pi-pulse (x180_long)
with the frequency that corresponds to Fock state n=1 (this can be changed by changing the IF, while sweeping over the idle time).
Then we subtract the two measurements in order to get the behaviour of Fock state n=1 (or a different n) in order to find t parity
(which is the time we need to wait between the two pi/2 pulses in order to distinguish between even and odd parity).


Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the resonator drive line (whether it's an external mixer or an Octave port).
    - Identification of the qubit's resonance frequency (referred to as "qubit_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Having calibrated qubit pi pulse (x180_len) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated qubit's frequency that corresponds to Fock state n=1 by running number_splitting_spectroscopy.
    - Specification of the expected storage_thermalization_time of the storage in the configuration.

Before proceeding to the next node:
    - Update the time we need to wait in parity measurements, labeled as t_parity.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
import macros as macros
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1000

tau_min = 16 // 4
tau_max = 1000 // 4
d_tau = 4 // 4
taus = np.arange(
    tau_min, tau_max + 0.1, d_tau
)  # + 0.1 to add tau_max to taus. Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "taus": taus,
    "config": config,
}

###################
# The QUA program #
###################
with program() as parity_meas:
    n = declare(int)  # QUA variable for the averaging loop
    tau = declare(int)  # QUA variable for the idle time
    state1 = declare(bool)
    state2 = declare(bool)
    I1 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q1 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I1_st = declare_stream()  # Stream for the 'I' quadrature
    Q1_st = declare_stream()  # Stream for the 'Q' quadrature
    I2 = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q2 = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I2_st = declare_stream()  # Stream for the 'I' quadrature
    Q2_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    state1_st = declare_stream()
    state2_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(tau, taus)):
            # Background measurement
            # Play the x90 qubit pulses with the frequency that corresponds to Fock state n=0
            update_frequency("qubit", qubit_IF)
            # Play a displacement pulse to the cavity
            play("cw", "storage")
            align()
            play("x90", "qubit")
            # Wait a varying idle time
            wait(tau, "qubit")
            # 2nd x90 gate
            play("-x90", "qubit")
            # Align the two elements to measure after playing the qubit pulses.
            align("qubit", "resonator")
            # Measure the state of the resonator
            state1, I1, Q1 = macros.readout_macro(threshold=ge_threshold, state=state1, I=I1, Q=Q1)
            # Wait for the storage to decay to the ground state
            align("storage", "resonator")
            wait(storage_thermalization_time * u.ns, "storage")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I1, I1_st)
            save(Q1, Q1_st)
            save(state1, state1_st)

            # Fock state parity measurement
            align()
            # Play a displacement pulse to the cavity
            play("cw", "storage")
            align()
            play("x90", "qubit")
            # Wait a varying idle time
            wait(tau, "qubit")
            # 2nd x90 gate
            play("-x90", "qubit")

            # Measure the storage state by applying a selective pi-pulse (n=1) to the qubit and measure the qubit state
            update_frequency("qubit", qubit_IF_n1)
            play("x180_long", "qubit")
            align("qubit", "resonator")
            state2, I2, Q2 = macros.readout_macro(threshold=ge_threshold, state=state2, I=I2, Q=Q2)

            # Wait for the storage to decay to the ground state
            align("storage", "resonator")
            wait(storage_thermalization_time * u.ns, "storage")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I2, I2_st)
            save(Q2, Q2_st)
            save(state2, state2_st)

        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        (I1_st.buffer(len(taus)) - I2_st.buffer(len(taus))).average().save("I")
        (Q1_st.buffer(len(taus)) - Q2_st.buffer(len(taus))).average().save("Q")
        (state1_st.boolean_to_int().buffer(len(taus)) - state2_st.boolean_to_int().buffer(len(taus))).average().save(
            "state"
        )
        n_st.save("iteration")

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
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, parity_meas, simulation_config)
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
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(parity_meas)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig1, ax1 = plt.subplots(2, 1)
    fig2, ax2 = plt.subplots(1, 1)
    interrupt_on_close(fig1, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        fig1.suptitle(f"Parity measurement")
        ax1[0].clear()
        ax1[1].clear()
        ax1[0].plot(4 * taus, I, ".")
        ax1[0].set_ylabel("I quadrature [V]")
        ax1[1].plot(4 * taus, Q, ".")
        ax1[1].set_xlabel("Idle time [ns]")
        ax1[1].set_ylabel("Q quadrature [V]")
        plt.pause(1)
        plt.tight_layout()

        ax2.clear()
        ax2.plot(4 * taus, state, ".")
        ax2.set_ylabel(r"$P_e$")
        ax2.set_xlabel("Idle time [ns]")
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"state_data": state})
    save_data_dict.update({"fig1_live": fig1})
    save_data_dict.update({"fig2_live": fig2})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
