"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for both resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "f_res" and "f_opt", in the state.
    - Adjust the readout amplitude, labeled as "readout_pulse_amp", in the state.
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
from macros import qua_declaration
import warnings

warnings.filterwarnings("ignore")

#######################################################
# Get the config from the machine in configuration.py #
#######################################################

# Update the readout amplitude for the sweep since we can only multiply it by a factor <2 in QUA.
prev_amp1 = machine.resonators[active_qubits[0]].readout_pulse_amp
prev_amp2 = machine.resonators[active_qubits[1]].readout_pulse_amp
machine.resonators[active_qubits[0]].readout_pulse_amp = 0.01
machine.resonators[active_qubits[1]].readout_pulse_amp = 0.01

# Build the config
config = build_config(machine)

# Get the resonators frequencies
res_if_1 = rr1.f_res - machine.local_oscillators.readout[rr1.LO_index].freq
res_if_2 = rr2.f_res - machine.local_oscillators.readout[rr2.LO_index].freq

###################
# The QUA program #
###################
n_avg = 100  # The number of averages
depletion_time = 5 * max(rr1.depletion_time, rr2.depletion_time)
# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amps = np.arange(0.05, 1.99, 0.01)
# The frequency sweep around the resonator resonance frequencies f_opt
dfs = np.arange(-10e6, +10e6, 0.1e6)


with program() as multi_res_spec_vs_amp:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(nb_of_qubits=2)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency

    # Bring the active qubits to the maximum frequency point
    set_dc_offset(q1_z, "single", qb1.z.max_frequency_point)
    set_dc_offset(q2_z, "single", qb2.z.max_frequency_point)

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        save(n, n_st)

        with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
            # Update the resonator frequencies
            update_frequency(rr1.name, df + res_if_1)
            update_frequency(rr2.name, df + res_if_2)

            with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the readout amplitude
                # resonator 1
                wait(depletion_time * u.ns, rr1.name)  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    rr1.name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[0]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[0]),
                )
                save(I[0], I_st[0])
                save(Q[0], Q_st[0])

                # align(rr1.name, rr2.name) # sequential to avoid overflow

                # resonator 2
                wait(depletion_time * u.ns, rr2.name)  # wait for the resonator to relax
                measure(
                    "readout" * amp(a),
                    rr2.name,
                    None,
                    dual_demod.full("cos", "out1", "sin", "out2", I[1]),
                    dual_demod.full("minus_sin", "out1", "cos", "out2", Q[1]),
                )
                save(I[1], I_st[1])
                save(Q[1], Q_st[1])

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

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open a quantum machine to execute the QUA program
    qm = qmm.open_qm(config)
    # Execute the QUA program
    job = qm.execute(multi_res_spec_vs_amp)
    # Prepare the figures for live plotting
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
        s1 = u.demod2volts(I1 + 1j * Q1, machine.resonators[0].readout_pulse_length)
        s2 = u.demod2volts(I2 + 1j * Q2, machine.resonators[0].readout_pulse_length)

        A1 = np.abs(s1)
        A2 = np.abs(s2)
        # Normalize data
        row_sums = A1.sum(axis=0)
        A1 = A1 / row_sums[np.newaxis, :]
        row_sums = A2.sum(axis=0)
        A2 = A2 / row_sums[np.newaxis, :]
        # Plot
        plt.suptitle("Resonator spectroscopy vs amplitude")
        plt.subplot(121)
        plt.cla()
        plt.title(f"{rr1.name} - f_cent: {int(rr1.f_res / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr1.readout_pulse_amp, dfs / u.MHz, A1)
        plt.axhline(0, color="k", linestyle="--")
        plt.axvline(prev_amp1, color="k", linestyle="--")
        plt.subplot(122)
        plt.cla()
        plt.title(f"{rr2.name} - f_cent: {int(rr2.f_res / u.MHz)} MHz")
        plt.xlabel("Readout amplitude [V]")
        plt.ylabel("Readout detuning [MHz]")
        plt.pcolor(amps * rr2.readout_pulse_amp, dfs / u.MHz, A2)
        plt.axhline(0, color="k", linestyle="--")
        plt.axvline(prev_amp2, color="k", linestyle="--")
        plt.tight_layout()

        plt.pause(0.1)
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # rr1.readout_pulse_amp =
    # rr2.readout_pulse_amp =
    # machine._save("current_state.json")
