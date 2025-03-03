"""
        RESONATOR DEPLETION TIME
This sequence is designed to measure the resonator depletion time.
It begins by sending a MW pulse to the resonator to fill it with photons via measure().
Subsequently, a Ramsey measurement is performed after allowing a variable waiting time (structured as:
wait(t) - x90 - idle_time - x90 - measurement). Given that the qubit frequency is influenced by the number of photons
in the resonator, an exponential decay should be evident in the measured I/Q quadratures.
This provides insight into the resonator depletion time, which can then be updated in the configuration.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having precisely measured the qubit frequency (ramsey).

Next steps before going to the next node:
    - Update the resonator depletion time (depletion_time) in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 1_000
ramsey_idle_time = 1 * u.us
# Time between populating the resonator and playing a Ramsey sequence in clock-cycles (4ns)
taus = np.arange(4, 1000, 1)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "ramsey_idle_time": ramsey_idle_time,
    "taus": taus,
    "config": config,
}

###################
# The QUA program #
###################
with program() as res_depletion_time:
    n = declare(int)
    n_st = declare_stream()
    t = declare(int)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(t, taus)):
            # Fill the resonator with photons
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            # Play a fixed duration Ramsey sequence after a varying time to estimate the effect of photons in the resonator
            wait(t, "resonator")
            # Align the two elements to play the Ramsey sequence after having waited for a varying time "t".
            align("qubit", "resonator")
            # Play the Ramsey sequence
            play("x90", "qubit")
            wait(ramsey_idle_time * u.ns)  # fixed time ramsey
            play("x90", "qubit")
            # Align the two elements to measure after playing the qubit pulse.
            align("qubit", "resonator")
            # Measure the state of the resonator
            measure(
                "readout",
                "resonator",
                None,
                dual_demod.full("rotated_cos", "rotated_sin", I),
                dual_demod.full("rotated_minus_sin", "rotated_cos", Q),
            )
            # Wait for the qubit to decay to the ground state
            wait(thermalization_time * u.ns, "resonator")
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(taus)).average().save("I")
        Q_st.buffer(len(taus)).average().save("Q")
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
    job = qmm.simulate(config, res_depletion_time, simulation_config)
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
    job = qm.execute(res_depletion_time)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I, Q = u.demod2volts(I, readout_len), u.demod2volts(Q, readout_len)
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Resonator depletion time")
        plt.subplot(211)
        plt.cla()
        plt.plot(4 * taus, I, ".")
        plt.ylabel("I quadrature [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot(4 * taus, Q, ".")
        plt.xlabel("Delay [ns]")
        plt.ylabel("Q quadrature [V]")
        plt.pause(0.1)
        plt.tight_layout()
    # Fit the results to extract the resonator depletion time
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        plt.figure()
        decay_fit = fit.T1(4 * taus, I, plot=True)
        depletion_time = np.round(np.abs(decay_fit["T1"][0]) / 4) * 4
        plt.xlabel("Delay [ns]")
        plt.ylabel("I quadrature [V]")
        print(f"Resonator depletion time to update in the config: depletion_time = {depletion_time:.0f} ns")
        plt.legend((f"depletion time = {depletion_time:.0f} ns",))
    except (Exception,):
        pass
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
