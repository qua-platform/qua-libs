"""
        CHARGE STABILITY MAP - fast and slow axes: QDAC2 set to trigger mode
The goal of the script is to acquire the charge stability map.
Here two channels of the QDAC2 are parametrized to step though two preloaded voltage lists on the event of two digital
markers provided by the OPX (connected to ext1 and ext2). This method allows the fast acquisition of a 2D voltage map
and the data can be fetched from the OPX in real time to enable live plotting.
The speed can also be further improved by removing the live plotting and increasing the QDAC2 bandwidth.

The QUA program consists in sending the triggers to the QDAC2 to increment the voltages and measure the charge of the dot
either via dc current sensing or RF reflectometry.
On top of the DC voltage sweeps, the OPX can output a continuous square wave (Coulomb pulse) through the AC line of the
bias-tee. This allows to check the coupling of the fast line to the sample and measure the lever arms between the DC and
AC lines.

A global average is performed (averaging on the most outer loop) and the data is extracted while the program is running
to display the full charge stability map with increasing SNR.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the QDAC2 and preloading the two voltages lists for the slow and fast axes.
    - Connect the two plunger gates (DC line of the bias-tee) to the QDAC2 and two digital markers from the OPX to the
      QDAC2 external trigger ports.
    - (optional) Connect the OPX to the fast line of the plunger gates for playing the Coulomb pulse and calibrate the
      lever arm.

Before proceeding to the next node:
    - Identify the different charge occupation regions.
    - Update the config with the lever-arms.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.addons.variables import assign_variables_to_element
from macros import RF_reflectometry_macro, DC_current_sensing_macro
from qdac2_driver import QDACII, load_voltage_list
import matplotlib.pyplot as plt
from macros import get_filtered_voltage
from scipy.optimize import minimize

###################
# The QUA program #
###################
n_avg = 100  # Number of averages
n_points_slow = 101  # Number of points for the slow axis
n_points_fast = 100  # Number of points for the fast axis
Coulomb_amp = 0.0  # amplitude of the Coulomb pulse
# How many Coulomb pulse periods to last the whole program
N = (int((readout_len + 1_000)/(2*bias_length)) + 1) * n_points_fast * n_points_slow * n_avg

level_empty = [-0.2, -0.2]
level_init = [0.03, 0.2]
level_readout = [0.15, 0.05]

duration_empty = 50000 // 4
duration_init = 5000 // 4
duration_readout = int(1.1 * readout_len // 4)

for j in range(2):
    S =  lambda x: np.abs(np.array(level_empty[j] + x)*duration_empty * 4 + np.array(level_init[j] + x)*duration_init * 4 + np.array(level_readout[j] + x)*duration_readout * 4)

    opt = minimize(S, x0=0, method="Nelder-Mead", options={"fatol":1e-8})
    wf=[]; wf_opt = []
    for i in range(5):
        wf += [level_empty[j]] * duration_empty * 4 + [level_init[j]] * duration_init * 4 + [level_readout[j]] * duration_readout * 4
        wf_opt += [level_empty[j] + opt.x] * duration_empty * 4 + [level_init[j] + opt.x] * duration_init * 4 + [level_readout[j] + opt.x] * duration_readout * 4

    y, y_filt = get_filtered_voltage(wf, 1e-9, 1e4)
    y_opt, y_filt_opt = get_filtered_voltage(wf_opt, 1e-9, 1e4)
    plt.figure()
    plt.subplot(211)
    plt.plot(y)
    plt.plot(y_filt)
    plt.subplot(212)
    plt.title(f"Optimum offset: {opt.x[0]:.6f} V")
    plt.plot(y_opt)
    plt.plot(y_filt_opt)
    plt.tight_layout()

    level_empty[j] += opt.x
    level_init[j] += opt.x
    level_readout[j] += opt.x
    level_empty[j] = level_empty[j][0]
    level_init[j] = level_init[j][0]
    level_readout[j] = level_readout[j][0]

# Voltages in Volt
voltage_values_slow = np.linspace(-1.5, 1.5, n_points_slow)
voltage_values_fast = np.linspace(-1.5, 1.5, n_points_fast)

with program() as charge_stability_prog:
    n = declare(int)  # QUA integer used as an index for the averaging loop
    counter = declare(int)  # QUA integer used as an index for the Coulomb pulse
    i = declare(int)  # QUA integer used as an index to loop over the voltage points
    j = declare(int)  # QUA integer used as an index to loop over the voltage points
    n_st = declare_stream()  # Stream for the iteration number (progress bar)
    I = declare(fixed)
    Q = declare(fixed)
    dc_signal = declare(fixed)

    # Ensure that the result variables are assign to the pulse processor used for readout
    assign_variables_to_element("tank_circuit", I, Q)
    assign_variables_to_element("TIA", dc_signal)
    # Play the Coulomb pulse continuously for the whole sequence
    #      ____      ____      ____      ____
    #     |    |    |    |    |    |    |    |
    # ____|    |____|    |____|    |____|    |...
    with for_(counter, 0, counter < N, counter + 1):
        # The Coulomb pulse
        play("bias" * amp(Coulomb_amp / P1_amp), "P1")
        play("bias" * amp(-Coulomb_amp / P1_amp), "P1")

    with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
        with for_(i, 0, i < n_points_slow, i + 1):
            # Trigger the QDAC2 channel to output the next voltage level from the list
            play("trigger", "qdac_trigger2")
            with for_(j, 0, j < n_points_fast, j + 1):
                # Trigger the QDAC2 channel to output the next voltage level from the list
                play("trigger", "qdac_trigger1")
                # Wait for the voltages to settle (depends on the channel bandwidth)
                # wait(300 * u.us, 'tank_circuit', "TIA")

                play("bias" * amp(level_empty[0] / P1_amp), "P1_sticky")
                wait(duration_empty, "P1_sticky")
                play("bias" * amp((level_init[0]-level_empty[0]) / P1_amp), "P1_sticky")
                wait(duration_init, "P1_sticky")
                play("bias" * amp((level_readout[0]-level_init[0]) / P1_amp), "P1_sticky")
                align("P1_sticky", "tank_circuit", "TIA")
                wait(duration_readout, "P1_sticky")
                play("bias" * amp(level_empty[1] / P1_amp), "P2_sticky")
                wait(duration_empty, "P2_sticky")
                play("bias" * amp((level_init[1] - level_empty[1]) / P1_amp), "P2_sticky")
                wait(duration_init, "P2_sticky")
                play("bias" * amp((level_readout[1] - level_init[1]) / P1_amp), "P2_sticky")
                wait(duration_readout, "P2_sticky")
                # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
                # frequency and the integrated quadratures are stored in "I" and "Q"
                I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
                # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
                # is stored in "dc_signal"
                dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)
                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
                # process them which can cause the OPX to crash.
                ramp_to_zero("P1_sticky")
                ramp_to_zero("P2_sticky")
                wait(1_000 * u.ns)  # in ns
        # Save the LO iteration to get the progress bar
        save(n, n_st)

    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(n_points_fast).buffer(n_points_slow).average().save("I")
        Q_st.buffer(n_points_fast).buffer(n_points_slow).average().save("Q")
        # DC current sensing
        dc_signal_st.buffer(n_points_fast).buffer(n_points_slow).average().save("dc_signal")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

## QDAC2 section
# Create the qdac instrument
# qdac = QDACII("Ethernet", IP_address="127.0.0.1", port=5025)  # Using Ethernet protocol
# # qdac = QDACII("USB", USB_device=4)  # Using USB protocol
# # Set up the qdac and load the voltage list
# load_voltage_list(
#     qdac,
#     channel=1,
#     dwell=2e-6,
#     slew_rate=2e7,
#     trigger_port="ext1",
#     output_range="low",
#     output_filter="med",
#     voltage_list=voltage_values_fast,
# )
# load_voltage_list(
#     qdac,
#     channel=2,
#     dwell=2e-6,
#     slew_rate=2e7,
#     trigger_port="ext2",
#     output_range="high",
#     output_filter="med",
#     voltage_list=voltage_values_slow,
# )

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=100_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, charge_stability_prog, simulation_config)
    plt.figure()
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(charge_stability_prog)
    # Get results from QUA program and initialize live plotting
    results = fetching_tool(job, data_list=["I", "Q", "dc_signal", "iteration"], mode="live")
    # Live plotting
    fig = plt.figure()
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        # Fetch the data from the last OPX run corresponding to the current slow axis iteration
        I, Q, DC_signal, iteration = results.fetch_all()
        I = u.demod2volts(I, reflectometry_readout_length)
        Q = u.demod2volts(Q, reflectometry_readout_length)
        DC_signal = u.demod2volts(DC_signal, readout_len)
        # Progress bar
        progress_counter(iteration, n_points_slow, start_time=results.start_time)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title("I quadrature [V]")
        plt.pcolor(voltage_values_fast, voltage_values_slow, I)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.subplot(122)
        plt.cla()
        plt.title("Q quadrature [V]")
        plt.pcolor(voltage_values_fast, voltage_values_slow, Q)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.tight_layout()
        plt.pause(0.1)
