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
    - Update the qubit frequency, labeled as "qubit_IF", in the configuration.
    - Update the relevant flux points in the configuration.
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
n_avg = 100  # Number of averaging loops
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state
saturation_len = 10 * u.us  # In ns
saturation_amp = 0.5  # pre-factor to the value defined in the config - restricted to [-2; 2)
# Frequency sweep in Hz
span = 10 * u.MHz
df = 100 * u.kHz
frequencies = np.arange(-span, +span + 0.1, df)
# Flux amplitude sweep (as a pre-factor of the flux amplitude)
dc_min = -0.49
dc_max = 0.49
step = 0.01
flux = np.arange(dc_min, dc_max + step / 2, step)  # +da/2 to add a_max to the scan


# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "IF_frequencies": frequencies,
    "flux": flux,
    "config": config,
}


###################################
# Helper functions and QUA macros #
###################################
# Get the resonator frequency vs flux trend from the node 05_resonator_spec_vs_flux.py in order to always measure on
# resonance while sweeping the flux
def cosine_func(x, amplitude, frequency, phase, offset):
    return amplitude * np.cos(2 * np.pi * frequency * x + phase) + offset


# The fit parameters are take from the config
fitted_curve = cosine_func(flux, amplitude_fit, frequency_fit, phase_fit, offset_fit)
fitted_curve = fitted_curve.astype(int)

###################
# The QUA program #
###################
with program() as qubit_spec_2D:
    n = declare(int)  # QUA variable for the averaging index
    f = declare(int)  # QUA variable for the qubit frequency
    dc = declare(fixed)  # QUA variable for the flux dc level
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    resonator_freq = declare(int, value=fitted_curve.tolist())  # res freq vs flux table
    index = declare(int, value=0)  # index to get the right resonator freq for a given flux
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        with for_(*from_array(f, frequencies)):
            # Update the qubit frequency
            update_frequency("qubit", f + qubit_IF)
            assign(index, 0)
            with for_(*from_array(dc, flux)):
                # Update the resonator frequency to always measure on resonance
                update_frequency("resonator", resonator_freq[index] + resonator_IF)
                # Flux sweeping
                set_dc_offset("flux_line", "single", dc)
                wait(flux_settle_time * u.ns, "resonator", "qubit")
                # Play a qubit pulse on the qubit
                play("saturation" * amp(saturation_amp), "qubit", duration=saturation_len * u.ns)
                # Align the two elements to measure after playing the qubit pulse.
                # One can also measure the resonator while driving the qubit by commenting the 'align'
                align("qubit", "resonator")
                # Measure the state of the resonator
                measure(
                    "readout",
                    "resonator",
                    None,
                    dual_demod.full("cos", "sin", I),
                    dual_demod.full("minus_sin", "cos", Q),
                )
                # Wait for the qubit to decay to the ground state
                wait(thermalization_time * u.ns, "resonator")
                # Update the resonator frequency vs flux index
                assign(index, index + 1)
                # Save the 'I' & 'Q' quadratures to their respective streams
                save(I, I_st)
                save(Q, Q_st)
        # Save the averaging iteration to get the progress bar
        save(n, n_st)

    with stream_processing():
        # Cast the data into a 2D matrix, average the 2D matrices together and store the results on the OPX processor
        I_st.buffer(len(flux)).buffer(len(frequencies)).average().save("I")
        Q_st.buffer(len(flux)).buffer(len(frequencies)).average().save("Q")
        n_st.save("iteration")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, qubit_spec_2D, simulation_config)
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
    job = qm.execute(qubit_spec_2D)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  #  Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, readout_len)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # 2D spectroscopy plot
        plt.subplot(211)
        plt.suptitle(r"Qubit spectroscopy")
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$")
        plt.pcolor(flux, frequencies / u.MHz, R)
        plt.ylabel("Qubit frequency [MHz]")
        plt.xlabel("Flux level [V]")
        plt.subplot(212)
        plt.cla()
        plt.title("Phase")
        plt.pcolor(flux, frequencies / u.MHz, np.unwrap(phase))
        plt.ylabel("Qubit frequency [MHz]")
        plt.xlabel("Flux level [V]")
        plt.pause(0.1)
        plt.tight_layout()
    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
