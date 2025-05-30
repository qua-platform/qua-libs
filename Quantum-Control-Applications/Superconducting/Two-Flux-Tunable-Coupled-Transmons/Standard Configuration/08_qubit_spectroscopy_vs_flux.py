"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Having calibrated the resonator frequency versus flux fit parameters (amplitude_fit, frequency_fit, phase_fit, offset_fit) in the configuration
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "qubit_IF_q", in the configuration.
    - Update the relevant flux points in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler


# Get the resonator frequency vs flux trend from the node 05_resonator_spec_vs_flux.py in order to always measure on
# resonance while sweeping the flux
def cosine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset


##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1000  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)
# Qubit detuning sweep with respect to qubit_IF
dfs = np.arange(-20e6, +20e6, 0.5e6)
# Flux sweep
dcs = np.arange(-0.5, 0.49, 0.02)

# The fit parameters are take from the config
fitted_curve1 = (cosine_func(dcs, amplitude_fit1, frequency_fit1, phase_fit1, offset_fit1)).astype(int)
fitted_curve2 = (cosine_func(dcs, amplitude_fit2, frequency_fit2, phase_fit2, offset_fit2)).astype(int)


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "saturation_len": saturation_len,
    "saturation_amp": saturation_amp,
    "dfs": dfs,
    "dcs": dcs,
    "config": config,
}

###################
# The QUA program #
###################
with program() as multi_qubit_spec_vs_flux:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit detuning
    dc = declare(fixed)  # QUA variable for the flux bias
    resonator_freq1 = declare(int, value=fitted_curve1.tolist())  # res freq vs flux table
    resonator_freq2 = declare(int, value=fitted_curve2.tolist())  # res freq vs flux table
    index = declare(int, value=0)  # index to get the right resonator freq for a given flux

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two qubit elements
            update_frequency("q1_xy", df + qubit_IF_q1)
            update_frequency("q2_xy", df + qubit_IF_q2)
            assign(index, 0)
            with for_(*from_array(dc, dcs)):
                # Flux sweeping
                set_dc_offset("q1_z", "single", dc)
                # set_dc_offset("q2_z", "single", dc)
                wait(flux_settle_time * u.ns)  # Wait for the flux to settle
                # Update the resonator frequency to always measure on resonance
                # update_frequency("rr1", resonator_freq1[index] + resonator_IF_q1)
                # update_frequency("rr2", resonator_freq2[index] + resonator_IF_q2)

                # Saturate qubit
                play("saturation" * amp(saturation_amp), "q1_xy", duration=saturation_len * u.ns)
                # play("saturation" * amp(saturation_amp), "q2_xy", duration=saturation_len * u.ns)

                # Multiplexed readout, also saves the measurement outcomes
                multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2])
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns)
                # Update the resonator frequency vs flux index
                assign(index, index + 1)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dcs)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dcs)).buffer(len(dfs)).average().save("Q2")

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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
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
    plt.show()
else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_qubit_spec_vs_flux)
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
        # Data analysis
        S1 = u.demod2volts(I1 + 1j * Q1, readout_len)
        S2 = u.demod2volts(I2 + 1j * Q2, readout_len)
        R1 = np.abs(S1)
        phase1 = np.angle(S1)
        R2 = np.abs(S2)
        phase2 = np.angle(S2)
        # Plots
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(dcs, (qubit_IF_q1 + dfs) / u.MHz, R1)
        plt.xlabel("Flux bias [V]")
        plt.ylabel("q1 IF [MHz]")
        plt.title(f"q1 (f_res: {(qubit_LO_q1 + qubit_IF_q1) / u.MHz} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(dcs, (qubit_IF_q1 + dfs) / u.MHz, phase1)
        plt.xlabel("Flux bias [V]")
        plt.ylabel("q1 IF [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(dcs, (qubit_IF_q2 + dfs) / u.MHz, R2)
        plt.title(f"q2 (f_res: {(qubit_LO_q2 + qubit_IF_q2) / u.MHz} MHz)")
        plt.xlabel("Flux bias [V]")
        plt.ylabel("q2 IF [MHz]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(dcs, (qubit_IF_q2 + dfs) / u.MHz, phase2)
        plt.xlabel("Flux bias [V]")
        plt.ylabel("q2 IF [MHz]")
        plt.tight_layout()
        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I1_data": I1})
    save_data_dict.update({"Q1_data": Q1})
    save_data_dict.update({"I2_data": I2})
    save_data_dict.update({"Q2_data": Q2})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
