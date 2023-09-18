"""
        QUBIT SPECTROSCOPY
This sequence involves sending a saturation pulse to the qubit, placing it in a mixed state,
and then measuring the state of the resonator across various qubit drive intermediate dfs.
In order to facilitate the qubit search, the qubit pulse duration and amplitude can be changed manually in the QUA
program directly without having to modify the configuration.

The data is post-processed to determine the qubit resonance frequency, which can then be used to adjust
the qubit intermediate frequency in the configuration under "center".

Note that it can happen that the qubit is excited by the image sideband or LO leakage instead of the desired sideband.
This is why calibrating the qubit mixer is highly recommended.

This step can be repeated using the "x180" operation instead of "saturation" to adjust the pulse parameters (amplitude,
duration, frequency) before performing the next calibration steps.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Set the flux bias to the maximum frequency point, labeled as "max_frequency_point", in the state.
    - Configuration of the saturation pulse amplitude and duration to transition the qubit into a mixed state.
    - Specification of the expected qubit T1 in the state.

Before proceeding to the next node:
    - Update the qubit frequency, labeled as f_01, in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import qua_declaration, multiplexed_readout
import warnings

warnings.filterwarnings("ignore")


#######################################################
# Get the config from the machine in configuration.py #
#######################################################
# Build the config
config = build_config(machine)

# Get the qubit frequencies (IFs and LOs)
lo1 = machine.local_oscillators.qubits[qb1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[qb2.xy.LO_index].freq
qb_if_1 = qb1.xy.f_01 - lo1
qb_if_2 = qb2.xy.f_01 - lo2

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# Qubit detuning sweep with respect to their resonance frequencies
dfs = np.arange(-60e6, +80e6, 1e6)

with program() as multi_qubit_spec:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit frequency

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)
        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)

            # qubit 1
            play("cw" * amp(saturation_amp), qb1.name + "_xy", duration=saturation_len * u.ns)
            align(qb1.name + "_xy", rr1.name)
            # qubit 2
            play("cw" * amp(saturation_amp), qb2.name + "_xy", duration=saturation_len * u.ns)
            align(qb2.name + "_xy", rr2.name)

            # QUA macro the readout the state of the active resonators (defined in macros.py)
            multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits)
            # Wait for the qubit to decay to the ground state
            wait(cooldown_time * u.ns)

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
qmm = QuantumMachinesManager(machine.network.qop_ip, cluster_name=machine.network.cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(multi_qubit_spec)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Convert results into Volts
        s1 = u.demod2volts(I1 + 1j * Q1, rr1.readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, rr2.readout_pulse_length)
        # Plot results
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(221)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s1))
        plt.grid("on")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{qb1.name} (f_01: {qb1.xy.f_01 / u.MHz} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s1))
        plt.grid("on")
        plt.ylabel("Phase [rad]")
        plt.xlabel(f"{qb1.name} detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.plot(dfs / u.MHz, np.abs(s2))
        plt.grid("on")
        plt.title(f"{qb2.name} (f_01: {qb2.xy.f_01 / u.MHz} MHz)")
        plt.subplot(224)
        plt.cla()
        plt.plot(dfs / u.MHz, np.angle(s2))
        plt.grid("on")
        plt.xlabel(f"{qb2.name} detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Fit the results to extract the resonance frequency
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        plt.suptitle("Qubit spectroscopy")
        plt.subplot(121)
        res_1 = fit.reflection_resonator_spectroscopy((qb_if_1 + dfs) / u.MHz, -np.angle(s1), plot=True)
        plt.legend((f"f = {res_1['f'][0]:.3f} MHz",))
        plt.xlabel(f"{qb1.name} IF [MHz]")
        plt.ylabel(r"R=$\sqrt{I^2 + Q^2}$ [V]")
        plt.title(f"{qb1.name}")
        plt.subplot(122)
        res_2 = fit.reflection_resonator_spectroscopy((qb_if_2 + dfs) / u.MHz, np.abs(s2), plot=True)
        plt.legend((f"f = {res_2['f'][0]:.3f} MHz",))
        plt.xlabel(f"{qb2.name} IF [MHz]")
        plt.title(f"{qb2.name}")
        plt.tight_layout()

        qb1.xy.f_01 = res_1["f"][0] * u.MHz + lo1
        qb2.xy.f_01 = res_2["f"][0] * u.MHz + lo2
    except (Exception,):
        pass

# machine._save("current_state.json")
