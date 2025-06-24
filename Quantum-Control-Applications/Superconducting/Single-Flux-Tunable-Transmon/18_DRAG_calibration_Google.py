"""
        DRAG PULSE CALIBRATION (GOOGLE METHOD)
The sequence consists in applying an increasing number of x180 and -x180 pulses successively while varying the DRAG
coefficient alpha. After such a sequence, the qubit is expected to always be in the ground state if the DRAG
coefficient has the correct value. Note that the idea is very similar to what is done in power_rabi_error_amplification.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the DRAG coefficient to a non-zero value in the config: such as drag_coef = 1
    - Set the desired flux bias.

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
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100

# Scan the DRAG coefficient pre-factor
a_min = 0.0
a_max = 1.0
da = 0.1
amps = np.arange(a_min, a_max + da / 2, da)  # + da/2 to add a_max to amplitudes

# Scan the number of pulses
iter_min = 0
iter_max = 25
d = 1
iters = np.arange(iter_min, iter_max + 0.1, d)

# Check that the DRAG coefficient is not 0
assert drag_coef != 0, "The DRAG coefficient 'drag_coef' must be different from 0 in the config."

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "amps": amps,
    "iters": iters,
    "config": config,
}

###################
# The QUA program #
###################
with program() as drag:
    n = declare(int)  # QUA variable for the averaging loop
    a = declare(fixed)  # QUA variable for the DRAG coefficient pre-factor
    it = declare(int)  # QUA variable for the number of qubit pulses
    pulses = declare(int)  # QUA variable for counting the qubit pulses
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)  # QUA variable for the qubit state
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    state_st = declare_stream()  # Stream for the qubit state
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(a, amps)):  # QUA for_ loop for sweeping the pulse amplitude
            with for_(*from_array(it, iters)):  # QUA for_ loop for sweeping the number of pulses
                # Loop for error amplification (perform many qubit pulses with varying DRAG coefficients)
                with for_(pulses, iter_min, pulses <= it, pulses + d):
                    play("x180" * amp(1, 0, 0, a), "qubit")
                    play("x180" * amp(-1, 0, 0, -a), "qubit")
                # Align the two elements to measure after playing the qubit pulses.
                align("qubit", "resonator")
                # Measure the resonator and extract the qubit state
                state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
                save(state, state_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(iters)).buffer(len(amps)).average().save("I")
        Q_st.buffer(len(iters)).buffer(len(amps)).average().save("Q")
        state_st.boolean_to_int().buffer(len(iters)).buffer(len(amps)).average().save("state")
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
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, drag, simulation_config)
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
    job = qm.execute(drag)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "state", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    while results.is_processing():
        # Fetch results
        I, Q, state, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("DRAG calibration (Google)")
        plt.subplot(231)
        plt.cla()
        plt.pcolor(iters, amps * drag_coef, I, cmap="magma")
        plt.xlabel("Number of iterations")
        plt.ylabel(r"Drag coefficient $\alpha$")
        plt.title("I [V]")
        plt.subplot(232)
        plt.cla()
        plt.pcolor(iters, amps * drag_coef, Q, cmap="magma")
        plt.xlabel("Number of iterations")
        plt.title("Q [V]")
        plt.subplot(233)
        plt.cla()
        plt.pcolor(iters, amps * drag_coef, state, cmap="magma")
        plt.xlabel("Number of iterations")
        plt.title("State")
        plt.subplot(212)
        plt.cla()
        plt.plot(amps * drag_coef, np.sum(I, axis=1))
        plt.xlabel(r"Drag coefficient $\alpha$")
        plt.ylabel("Sum along the iterations")
        plt.tight_layout()
        plt.pause(0.1)
    print(f"Optimal drag_coef = {drag_coef * amps[np.argmin(np.sum(I, axis=1))]:.3f}")

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"state_data": state})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
