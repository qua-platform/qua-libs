"""
        CHARGE SENSOR GATE SWEEP with the OPX
Here the voltage biasing the sensor gate is provided and being swept by the OPX connected to the DC line of the bias-tee.
A sticky element is used in order to maintain the voltage level and avoid fast voltage drops. Note that the OPX signal
can be combined with an external dc source to increase the dynamics.
The OPX is also measuring, either via dc current sensing or RF reflectometry, the response of the sensor dot.

A global average is performed (averaging on the most outer loop) and the data is extracted while the program is running
to display the full charge stability map with increasing SNR.
The script can also be easily modified to perform single point averaging instead.

Prerequisites:
    - Connect one the DC line of the bias-tee connected to the sensor dot to one OPX channel.
    - Setting the parameters of the external DC source using its driver if needed.

Before proceeding to the next node:
    - Update the config with the optimal sensing point.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
import matplotlib.pyplot as plt
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Parameters Definition
n_avg = 100  # Number of averaging loops
offsets = np.linspace(-0.2, 0.2, 101)
d_offset = np.diff(offsets)[0]

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "offsets": offsets,
    "d_offset": d_offset,
    "config": config,
}

###################
# The QUA program #
###################
with program() as charge_sensor_sweep:
    dc = declare(fixed)  # QUA variable for the voltage sweep
    n = declare(int)  # QUA variable for the averaging loop
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    with for_(n, 0, n < n_avg, n + 1):
        # Set the voltage to the 1st point of the sweep
        play("step" * amp(offsets[0] / charge_sensor_amp), "sensor_gate_sticky")
        # Wait for the voltage to settle (depends on the bias-tee cut-off frequency)
        wait(1 * u.ms, "sensor_gate_sticky")
        with for_(*from_array(dc, offsets)):
            # Play only from the second iteration
            with if_(~(dc == offsets[0])):
                play("step" * amp(d_offset / charge_sensor_amp), "sensor_gate_sticky")
                # Wait for the voltage to settle (depends on the bias-tee cut-off frequency)
                wait(1 * u.ms, "sensor_gate_sticky")
            align()
            # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
            # frequency and the integrated quadratures are stored in "I" and "Q"
            I, Q, I_st, Q_st = RF_reflectometry_macro()
            # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
            # is stored in "dc_signal"
            dc_signal, dc_signal_st = DC_current_sensing_macro()
            # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
            # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
            # process them which can cause the OPX to crash.
            wait(1_000 * u.ns, "tank_circuit")
        ramp_to_zero("sensor_gate_sticky")
        save(n, n_st)

    with stream_processing():
        I_st.buffer(len(offsets)).average().save("I")
        Q_st.buffer(len(offsets)).average().save("Q")
        dc_signal_st.buffer(len(offsets)).average().save("dc_signal")
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
    job = qmm.simulate(config, charge_sensor_sweep, simulation_config)
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
    job = qm.execute(charge_sensor_sweep)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, reflectometry_readout_length, single_demod=True)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Plot results
        plt.suptitle("Charge sensor gate sweep")
        plt.subplot(211)
        plt.cla()
        plt.plot(offsets, R)
        plt.xlabel("Sensor gate voltage [V]")
        plt.ylabel(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.subplot(212)
        plt.cla()
        plt.plot(offsets, phase)
        plt.xlabel("Sensor gate voltage [V]")
        plt.ylabel("Phase [rad]")
        plt.tight_layout()
        plt.pause(0.1)
    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)
    save_data_dict.update({"I_data": I})
    save_data_dict.update({"Q_data": Q})
    save_data_dict.update({"fig_live": fig})
    data_handler.additional_files = {script_name: script_name, **default_additional_files}
    data_handler.save_data(data=save_data_dict, name="_".join(script_name.split("_")[1:]).split(".")[0])
