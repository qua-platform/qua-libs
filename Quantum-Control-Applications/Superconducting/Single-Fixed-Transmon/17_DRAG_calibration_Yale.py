"""
        DRAG PULSE CALIBRATION (YALE METHOD)
The sequence consists in applying successively x180-y90 and y180-x90 to the qubit while varying the DRAG
coefficient alpha. The qubit is reset to the ground state between each sequence and its state is measured and stored.
Each sequence will bring the qubit to the same state only when the DRAG coefficient is set to its correct value.

This protocol is described in Reed's thesis (Fig. 5.8) https://rsl.yale.edu/sites/default/files/files/RSL_Theses/reed.pdf
This protocol was also cited in: https://doi.org/10.1103/PRXQuantum.2.040202

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the DRAG coefficient to a non-zero value in the config: such as drag_coef = 1

Next steps before going to the next node:
    - Update the DRAG coefficient (drag_coef) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from macros import readout_macro
import matplotlib.pyplot as plt


###################
# The QUA program #
###################

n_avg = 100

# Scan the DRAG coefficient pre-factor
a_min = -1.0
a_max = 1.0
da = 0.1
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

# Check that the DRAG coefficient is not 0
assert drag_coef != 0, "The DRAG coefficient 'drag_coef' must be different from 0 in the config."

with program() as drag:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the DRAG coefficient pre-factor
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)  # QUA variable for the qubit state
    I1_st = declare_stream()  # Stream for the 'I' quadrature for the 1st sequence x180-y90
    Q1_st = declare_stream()  # Stream for the 'Q' quadrature for the 1st sequence x180-y90
    I2_st = declare_stream()  # Stream for the 'Q' quadrature for the 2nd sequence y180-x90
    Q2_st = declare_stream()  # Stream for the 'Q' quadrature for the 2nd sequence y180-x90
    state1_st = declare_stream()  # Stream for the qubit state for the 1st sequence x180-y90
    state2_st = declare_stream()  # Stream for the qubit state for the 2nd sequence y180-x90
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(a, amps)):
            # Play the 1st sequence with varying DRAG coefficient
            play("x180" * amp(1, 0, 0, a), "qubit")
            play("y90" * amp(a, 0, 0, 1), "qubit")
            # Align the two elements to measure after playing the qubit pulses.
            align("qubit", "resonator")
            # Measure the resonator and extract the qubit state
            state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I1_st)
            save(Q, Q1_st)
            save(state, state1_st)

            align()  # Global align between the two sequences

            # Play the 2nd sequence with varying DRAG coefficient
            play("y180" * amp(a, 0, 0, 1), "qubit")
            play("x90" * amp(1, 0, 0, a), "qubit")
            # Align the two elements to measure after playing the qubit pulses.
            align("qubit", "resonator")
            # Measure the resonator and extract the qubit state
            state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I2_st)
            save(Q, Q2_st)
            save(state, state2_st)
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I1_st.buffer(len(amps)).average().save("I1")
        Q1_st.buffer(len(amps)).average().save("Q1")
        I2_st.buffer(len(amps)).average().save("I2")
        Q2_st.buffer(len(amps)).average().save("Q2")
        state1_st.boolean_to_int().buffer(len(amps)).average().save("state1")
        state2_st.boolean_to_int().buffer(len(amps)).average().save("state2")
        n_st.save("iteration")

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
    job = qmm.simulate(config, drag, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(drag)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I1", "I2", "Q1", "Q2", "state1", "state2", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        I1, I2, Q1, Q2, state1, state2, iteration = results.fetch_all()
        # Convert the results into Volts
        I1, Q1 = u.demod2volts(I1, readout_len), u.demod2volts(Q1, readout_len)
        I2, Q2 = u.demod2volts(I2, readout_len), u.demod2volts(Q2, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("DRAG coefficient calibration (Yale)")
        plt.subplot(311)
        plt.cla()
        plt.plot(amps * drag_coef, I1, label="x180y90")
        plt.plot(amps * drag_coef, I2, label="y180x90")
        plt.ylabel("I [V]")
        plt.legend()
        plt.subplot(312)
        plt.cla()
        plt.plot(amps * drag_coef, Q1, label="x180y90")
        plt.plot(amps * drag_coef, Q2, label="y180x90")
        plt.ylabel("Q [V]")
        plt.legend()
        plt.subplot(313)
        plt.cla()
        plt.plot(amps * drag_coef, state1, label="x180y90")
        plt.plot(amps * drag_coef, state2, label="y180x90")
        plt.xlabel("Drag coefficient")
        plt.ylabel("g-e transition probability")
        plt.legend()
        plt.tight_layout()
        plt.pause(0.1)
