"""
        RABI CHEVRON (DURATION VS FREQUENCY)
This sequence involves executing the qubit pulse (such as x180, square_pi, or other types) and measuring the state
of the resonator across various qubit intermediate frequencies and pulse durations.
By analyzing the results, one can determine the qubit and estimate the x180 pulse duration for a specified amplitude.

Prerequisites:
    - Determination of the resonator's resonance frequency when coupled to the qubit of interest (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (be it an external mixer or an Octave port).
    - Identification of the approximate qubit frequency (referred to as "qubit_spectroscopy").
    - Configuration of the qubit frequency and the desired pi pulse amplitude (labeled as "pi_amp").
    - Set the desired flux bias

Before proceeding to the next node:
    - Adjust the qubit frequency setting, labeled as "f_01", in the state.
    - Modify the qubit pulse amplitude setting, labeled as "pi_len", in the state.
    - Save the current state by calling machine._save("current_state.json")
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import qua_declaration, multiplexed_readout
import matplotlib.pyplot as plt
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
cooldown_time = 5 * max(qb1.T1, qb2.T1)
# The frequency sweep with respect to the qubits resonance frequencies
dfs = np.arange(-100e6, +100e6, 1e6)
# Pulse duration sweep (in clock cycles = 4ns) - must be larger than 4 clock cycles
durations = np.arange(4, 100, 2)


with program() as rabi_chevron:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    df = declare(int)  # QUA variable for the qubit detuning
    t = declare(int)  # QUA variable for the qubit pulse duration

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        with for_(*from_array(df, dfs)):
            # Update the qubit frequencies
            update_frequency(qb1.name + "_xy", df + qb_if_1)
            update_frequency(qb2.name + "_xy", df + qb_if_2)

            with for_(*from_array(t, durations)):
                # Play the qubit drives
                play("x180", qb1.name + "_xy", duration=t)
                play("x180", qb2.name + "_xy", duration=t)
                # Align all elements to measure after playing the qubit pulse.
                align()
                # QUA macro the readout the state of the active resonators (defined in macros.py)
                multiplexed_readout(I, I_st, Q, Q_st, resonators=active_qubits)
                # Wait for the qubit to decay to the ground state
                wait(cooldown_time * u.ns)

    with stream_processing():
        n_st.save("n")
        # resonator 1
        I_st[0].buffer(len(durations)).buffer(len(dfs)).average().save("I1")
        Q_st[0].buffer(len(durations)).buffer(len(dfs)).average().save("Q1")
        # resonator 2
        I_st[1].buffer(len(durations)).buffer(len(dfs)).average().save("I2")
        Q_st[1].buffer(len(durations)).buffer(len(dfs)).average().save("Q2")


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
        I1, Q1 = u.demod2volts(I1, rr1.readout_pulse_length), u.demod2volts(Q1, rr1.readout_pulse_length)
        I2, Q2 = u.demod2volts(I2, rr2.readout_pulse_length), u.demod2volts(Q2, rr2.readout_pulse_length)
        # Plot results
        plt.suptitle("Rabi chevron")
        plt.subplot(221)
        plt.cla()
        plt.pcolor(durations * 4, dfs / u.MHz, I1)
        plt.plot(qb1.xy.pi_length, 0, "r*")
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.title(f"{qb1.name} (f_res1: {int(qb1.xy.f_01 / u.MHz)} MHz)")
        plt.subplot(223)
        plt.cla()
        plt.pcolor(durations * 4, dfs / u.MHz, Q1)
        plt.plot(qb1.xy.pi_length, 0, "r*")
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.subplot(222)
        plt.cla()
        plt.pcolor(durations * 4, dfs / u.MHz, I2)
        plt.plot(qb2.xy.pi_length, 0, "r*")
        plt.title(f"{qb2.name} (f_res2: {int(qb2.xy.f_01 / u.MHz)} MHz)")
        plt.ylabel("Qubit detuning [MHz]")
        plt.xlabel("Qubit pulse duration [ns]")
        plt.subplot(224)
        plt.cla()
        plt.pcolor(durations * 4, dfs / u.MHz, Q2)
        plt.plot(qb2.xy.pi_length, 0, "r*")
        plt.xlabel("Qubit pulse duration [ns]")
        plt.ylabel("Qubit detuning [MHz]")
        plt.tight_layout()
        plt.pause(0.1)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
