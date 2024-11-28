"""
        RESONATOR SPECTROSCOPY INDIVIDUAL RESONATORS
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to extract the
'I' and 'Q' quadratures across varying readout intermediate frequencies.
The data is then post-processed to determine the resonator resonance frequency.
This frequency can be used to update the readout intermediate frequency in the configuration under "resonator_IF".

Prerequisites:
    - Ensure calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibrate the IQ mixer connected to the readout line (whether it's an external mixer or an Octave port).
    - Define the readout pulse amplitude and duration in the configuration.
    - Specify the expected resonator depletion time in the configuration.

Before proceeding to the next node:
    - Update the readout frequency, labeled as "resonator_IF_q1" and "resonator_IF_q2", in the configuration.
"""

from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from configuration import *
from qualang_tools.results import fetching_tool
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from scipy import signal
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################

resonator = "rr1"
resonator_LO = resonator_LO

n_avg = 200  # The number of averages
frequencies = {
    "rr1": np.arange(-50e6, +50e6, 100e3),
    "rr2": np.arange(-50e6, +50e6, 100e3),
}

save_data_dict = {
    "resonator": resonator,
    "resonator_LO": resonator_LO,
    "frequencies": frequencies,
    "n_avg": n_avg,
    "config": config,
}

###################
#   QUA Program   #
###################

with program() as PROGRAM:
    n = declare(int)  # QUA variable for the averaging loop
    f = declare(int)  # QUA variable for the readout frequency --> Hz int 32 up to 2^32
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature --> signed 4.28 [-8, 8)
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature --> signed 4.28 [-8, 8)
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(*from_array(f, frequencies[resonator])):  # QUA for_ loop for sweeping the frequency
            # Update the frequency of the digital oscillator linked to the resonator element
            update_frequency(resonator, f)
            # Measure the resonator (send a readout pulse and demodulate the signals to get the 'I' & 'Q' quadratures)
            measure(
                "readout" * amp(1),
                resonator,
                None,
                dual_demod.full("cos", "sin", I),
                dual_demod.full("minus_sin", "cos", Q),
            )
            # Wait for the resonator to deplete
            wait(depletion_time * u.ns, resonator)
            # Save the 'I' & 'Q' quadratures to their respective streams
            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        # Cast the data into a 1D vector, average the 1D vectors together and store the results on the OPX processor
        I_st.buffer(len(frequencies[resonator])).average().save("I")
        Q_st.buffer(len(frequencies[resonator])).average().save("Q")

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
    simulation_config = SimulationConfig(duration=1_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PROGRAM, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()
    plt.show(block=False)
else:
    try:
        # Open a quantum machine to execute the QUA program
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(PROGRAM)
        # Get results from QUA program
        results = fetching_tool(job, data_list=["I", "Q"])  # this one already waits for all values
        # plotting
        fig = plt.figure()
        I, Q = results.fetch_all()
        # Convert results into Volts
        S = I + 1j * Q
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        plt.suptitle(f"Resonator spectroscopy for {resonator} - LO = {resonator_LO / u.GHz} GHz")
        ax1 = plt.subplot(211)
        plt.plot((frequencies[resonator]) / u.MHz, R, ".")
        plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(212, sharex=ax1)
        plt.plot((frequencies[resonator]) / u.MHz, signal.detrend(np.unwrap(phase)), ".")
        plt.xlabel("Intermediate frequency [MHz]")
        plt.ylabel("Phase [rad]")
        plt.tight_layout()

        save_data_dict[resonator + "_I"] = I
        save_data_dict[resonator + "_Q"] = Q

        # Save results
        script_name = Path(__file__).name
        data_handler = DataHandler(root_data_folder=save_dir)
        save_data_dict.update({"fig_live": fig})
        data_handler.additional_files = {script_name: script_name, **default_additional_files}
        data_handler.save_data(data=save_data_dict, name="resonator_spectroscopy_single")

    except Exception as e:
        print(f"An exception occurred: {e}")

    finally:
        qm.close()
        print("Experiment QM is now closed")
        plt.show(block=True)
