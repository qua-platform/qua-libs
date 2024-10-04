"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly without having to modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be used to adjust
the qubit intermediate frequency in the configuration under "qubit_IF".

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

This step can be repeated using the "x180" operation to adjust the pulse parameters (amplitude, duration, frequency)
before performing the next calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy_multiplexed").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Configuration of the cw pulse amplitude (const_amp) and duration (CONST_LEN) to transition the qubit into a mixed state.
    - Specification of the expected qubits T1 in the configuration.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "qubit_IF_q", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration_mw_fem import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1_000  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
# Qubit detuning sweep with respect to qubit_IF
span = 20.0 * u.MHz
freq_step = 100 * u.kHz
dfs = np.arange(-span, +span, freq_step)
saturation_len = 10 * u.us  # In ns
saturation_scaling = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "dfs": dfs,
    "config": config,
    "saturation_scaling": saturation_scaling,
    "saturation_len": saturation_len,
}

###################
#   QUA Program   #
###################

with program() as PROGRAM:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the readout frequency

    # Adjust the flux line biases if needed
    # set_dc_offset("q1_z", "single", 0.0)
    # set_dc_offset("q2_z", "single", 0.0)
    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(df, dfs)):
            # Update the frequency of the two qubit elements
            update_frequency("q1_xy", df + qubit_IF_q1)
            update_frequency("q2_xy", df + qubit_IF_q2)
            # Play the saturation pulse to put the qubit in a mixed state - Can adjust the amplitude on the fly [-2; 2)
            # qubit 1
            play("cw" * amp(saturation_scaling), "q1_xy", duration=saturation_len * u.ns)
            align("q1_xy", "rr1")
            # qubit 2
            play("cw" * amp(saturation_scaling), "q2_xy", duration=saturation_len * u.ns)
            align("q2_xy", "rr2")

            # Multiplexed readout, also saves the measurement outcomes
            multiplexed_readout(I, I_st, Q, Q_st, resonators=[1, 2])
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(dfs)).average().save("Q2")


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
    job = qmm.simulate(config, PROGRAM, simulation_config)
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try:
        # Open a quantum machine to execute the QUA program
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
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
            plt.plot((dfs + qubit_IF_q1) / u.MHz, R1)
            plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
            plt.title(f"Qubit 1 - LO = {qubit_LO_q1 / u.GHz} GHz)")
            plt.subplot(223)
            plt.cla()
            plt.plot((dfs + qubit_IF_q1) / u.MHz, np.unwrap(phase1))
            plt.ylabel("Phase [rad]")
            plt.xlabel("Qubit intermediate frequency [MHz]")
            plt.subplot(222)
            plt.cla()
            plt.plot((dfs + qubit_IF_q2) / u.MHz, np.abs(R2))
            plt.title(f"Qubit 2 - LO = {qubit_LO_q2 / u.GHz} GHz)")
            plt.subplot(224)
            plt.cla()
            plt.plot((dfs + qubit_IF_q2) / u.MHz, np.unwrap(phase2))
            plt.xlabel("Qubit intermediate frequency [MHz]")
            plt.tight_layout()
            plt.pause(1)

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="qubit_spectroscopy")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
