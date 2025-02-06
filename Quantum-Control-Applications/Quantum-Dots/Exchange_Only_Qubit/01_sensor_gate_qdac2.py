"""
        CHARGE SENSOR GATE SWEEP with an external DC source
Here the voltage biasing the sensor gate is provided and being swept by an external DC source connected to the DC line
of the bias-tee.
The OPX is simply measuring, either via dc current sensing or RF reflectometry, the response of the sensor dot.

A single point averaging is performed (averaging on the most inner loop) and the data is extracted while the program is
running.

Prerequisites:
    - Connect one the DC line of the bias-tee connected to the sensor dot to one OPX channel.
    - Setting the parameters of the external DC source using its driver if needed.

Before proceeding to the next node:
    - Update the config with the optimal sensing point.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
import matplotlib.pyplot as plt

###################
# The QUA program #
###################

n_avg = 100  # Number of averaging loops

offsets_points = 100
offsets = np.linspace(-0.2, 0.2, offsets_points)

with program() as charge_sensor_sweep:

    n = declare(int)  # QUA integer used as an index for the averaging loop
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)
    Q = declare(fixed)
    I_st = declare_stream()
    Q_st = declare_stream()

    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop

        save(n, n_st)

        with for_(i, 0, i < offsets_points, i + 1):

            play("trigger", "qdac_trigger1")
            # Wait for the voltages to settle (depends on the voltage source bandwidth)
            wait(500 * u.us)

            align('qdac_trigger1', 'QDS')

            reset_phase('QDS')
            measure('readout', 'QDS', None, demod.full("cos", I, 'out2'), demod.full("sin", Q, 'out2'))

            save(I, I_st)
            save(Q, Q_st)

    with stream_processing():
        n_st.save("iteration")
        I_st.buffer(offsets_points).average().save("I")
        Q_st.buffer(offsets_points).average().save("Q")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

## QDAC2 section
# Create the qdac instrument
qdac = QDACII("Ethernet", IP_address=qdac_ip, port=qdac_port)  # Using Ethernet protocol
# qdac = QDACII("USB", USB_device=4)  # Using USB protocol
# Set up the qdac and load the voltage list
load_voltage_list(
    qdac,
    channel=1,
    dwell=2e-6,
    slew_rate=2e7,
    trigger_port="ext1",
    output_range="low",
    output_filter="med",
    voltage_list=offsets,
)

#######################
# Simulate or execute #
#######################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, charge_sensor_sweep, simulation_config)
    # Plot the simulated samples
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(charge_sensor_sweep)

    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the 
    
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    while results.is_processing():

        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, iteration = results.fetch_all()
        # Convert results into Volts
        S = u.demod2volts(I + 1j * Q, lock_in_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.unwrap(np.angle(S))  # Phase

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

    qm.close()