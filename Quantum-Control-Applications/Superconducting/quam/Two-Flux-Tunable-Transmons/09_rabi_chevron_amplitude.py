"""
        RABI CHEVRON (AMPLITUDE VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse amplitudes.
By analyzing the results, one can determine the qubit and estimate the x180 pulse amplitude for a specified duration.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse duration (labeled as "pi_len").
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "f_01", in the state.
    - Modify the qubit pulse amplitude setting, labeled as "pi_amp", in the state.
    - Save the current state by calling machine.save("quam")
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


#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Build the config
config = build_config(machine)

# Get the qubit frequencies (IFs and LOs)
lo1 = machine.local_oscillators.qubits[q1.xy.LO_index].freq
lo2 = machine.local_oscillators.qubits[q2.xy.LO_index].freq
q1.xy.intermediate_frequency = q1.xy.f_01 - lo1
q2.xy.intermediate_frequency = q2.xy.f_01 - lo2

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
cooldown_time = 5 * max(q1.T1, q2.T1)
# The frequency sweep with respect to the qubits resonance frequencies
dfs = np.arange(-100e6, +100e6, 1e6)
# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(0.0, 1.9, 0.02)


with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit detuning
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor

    # Bring the active qubits to the minimum frequency point
    set_dc_offset(q1_z, "single", q1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", q2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(q1.xy.name, df + q1.xy.intermediate_frequency)
            update_frequency(q2.xy.name, df + q2.xy.intermediate_frequency)

            with for_(*from_array(a, amps)):
                # Play the qubit drives
                play("x180" * amp(a), q1.xy.name)
                play("x180" * amp(a), q2.xy.name)
                # Align all elements to measure after playing the qubit pulse.
                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(machine, I, I_st, Q, Q_st)
                # Wait for the qubit to decay to the ground state
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(amps)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(amps)).buffer(len(dfs)).average().save("Q2")


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
    job = qmm.simulate(config, rabi_chevron, simulation_config)
    job.get_simulated_samples().con1.plot()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(rabi_chevron)
    # Get results from QUA program
    results = fetching_tool(job, ["n", "I1", "Q1", "I2", "Q2"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        n, I1, Q1, I2, Q2 = results.fetch_all()
        # Progress bar
        progress_counter(n, n_avg, start_time=results.start_time)
        # Convert results into Volts
        I1, Q1 = u.demod2volts(I1, q1.resonator.operations["readout"].length), u.demod2volts(Q1, q1.resonator.operations["readout"].length)
        I2, Q2 = u.demod2volts(I2, q2.resonator.operations["readout"].length), u.demod2volts(Q2, q2.resonator.operations["readout"].length)
        # Plot results
        plt.suptitle("Rabi chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(amps * q1.xy.pi_amp, dfs / u.MHz, I1)
        plt.plot(q1.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.title(f"{q1.name} (f_res1: {int((q1.xy.intermediate_frequency + lo1) / u.MHz)} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(amps * q1.xy.pi_amp, dfs / u.MHz, Q1)
        plt.plot(q1.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(amps * q2.xy.pi_amp, dfs / u.MHz, I2)
        plt.plot(q2.xy.pi_amp, 0, "r*")
        plt.title(f"{q2.name} (f_res2: {int((q2.xy.intermediate_frequency + lo2) / u.MHz)} MHz)")
        plt.ylabel("Qubit detuning [MHz]")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(amps * q2.xy.pi_amp, dfs / u.MHz, Q2)
        plt.plot(q2.xy.pi_amp, 0, "r*")
        plt.xlabel("Qubit pulse amplitude [V]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
