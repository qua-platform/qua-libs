"""
        Pauli Spin Blockade search
The goal of the script is to find the PSB region according to the protocol described in Nano Letters 2020 20 (2), 947-952.
To do so, the charge stability map is acquired by scanning the voltages provided by the QDAC2,
to the DC part of the bias-tees connected to the plunger gates, while 2 OPX channels are stepping the voltages on the fast
lines of the bias-tees to navigate through the triangle in voltage space (empty - random initialization - measurement).

Depending on the cut-off frequency of the bias-tee, it may be necessary to adjust the barycenter (voltage offset) of each
triangle so that the fast line of the bias-tees sees zero voltage in average. Otherwise, the high-pass filtering effect
of the bias-tee will distort the fast pulses over time. A function has been written for this.

In the current implementation, the OPX is also measuring (either with DC current sensing or RF-reflectometry) during the
readout window (last segment of the triangle).
A single-point averaging is performed and the data is extracted while the program is running to display the results line-by-line.

Prerequisites:
    - Readout calibration (resonance frequency for RF reflectometry and sensor operating point for DC current sensing).
    - Setting the parameters of the QDAC2 and preloading the two voltages lists for the slow and fast axes.
    - Connect the two plunger gates (DC line of the bias-tee) to the QDAC2 and two digital markers from the OPX to the
      QDAC2 external trigger ports.
    - Connect the OPX to the fast line of the plunger gates for playing the triangle pulse sequence.

Before proceeding to the next node:
    - Identify the PSB region and update the config.
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
from macros import get_filtered_voltage, round_to_fixed
from scipy.optimize import minimize

###################
# The QUA program #
###################
n_avg = 100  # Number of averages
n_points_slow = 101  # Number of points for the slow axis
n_points_fast = 100  # Number of points for the fast axis
Coulomb_amp = 0.0  # amplitude of the Coulomb pulse
# How many Coulomb pulse periods to last the whole program
N = (int((readout_len + 1_000) / (2 * bias_length)) + 1) * n_points_fast * n_points_slow * n_avg

# Points in the charge stability map [V1, V2]
level_empty = [-0.2, -0.2]
level_init = [0.03, 0.2]
level_readout = [0.15, 0.05]

duration_empty = 5000 // 4
duration_init = 1000 // 4
duration_readout = int(1.1 * readout_len // 4)


def balance_fast_pulse_sequence(voltage_levels, durations):
    """Finds the barycenter of the triangle so that the fast line of the bias-tee sees V in average at the end of the triangle."""
    balanced_sequence = []
    dc_offset = []
    for i in range(len(np.array(voltage_levels)[0, :])):
        S = lambda x: np.abs(
            np.sum(np.array(voltage_levels)[:, i] * np.array(durations)) + x * np.sum(np.array(durations))
        )
        opt = minimize(S, x0=np.array(0), method="Nelder-Mead", options={"fatol": 1e-8})
        dc_offset.append(-opt.x[0])
        balanced_sequence.append(np.array(voltage_levels)[:, i] + opt.x)
        print(opt)
    return np.array(balanced_sequence), dc_offset


corr, off = balance_fast_pulse_sequence(
    [level_empty, level_init, level_readout], [duration_empty, duration_init, duration_readout]
)

# Visualize the pulse sequence before and after balancing
for j in range(2):
    wf = []
    wf_opt = []
    for i in range(5):
        wf += (
            [level_empty[j]] * duration_empty * 4
            + [level_init[j]] * duration_init * 4
            + [level_readout[j]] * duration_readout * 4
        )
        wf_opt += (
            [corr[j, 0]] * duration_empty * 4 + [corr[j, 1]] * duration_init * 4 + [corr[j, 2]] * duration_readout * 4
        )

    y, y_filt = get_filtered_voltage(wf, 1e-9, 5e3)
    y_opt, y_filt_opt = get_filtered_voltage(wf_opt, 1e-9, 5e3)
    plt.figure()
    plt.subplot(211)
    plt.plot(y)
    plt.plot(y_filt)
    plt.subplot(212)
    plt.title(f"Optimum offset: {off[j]:.6f} V")
    plt.plot(y_opt)
    plt.plot(y_filt_opt)
    plt.tight_layout()

# Update the triangle levels and convert them to "fixed" to reduce the accumulation of fixed point arithmetic errors
level_empty = [round_to_fixed(corr[0, 0]), round_to_fixed(corr[1, 0])]
level_init = [round_to_fixed(corr[0, 1]), round_to_fixed(corr[1, 1])]
level_readout = [round_to_fixed(corr[0, 2]), round_to_fixed(corr[1, 2])]

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

    with for_(i, 0, i < n_points_slow, i + 1):
        # Trigger the QDAC2 channel to output the next voltage level from the list
        play("trigger", "qdac_trigger2")
        with for_(j, 0, j < n_points_fast, j + 1):
            # Trigger the QDAC2 channel to output the next voltage level from the list
            play("trigger", "qdac_trigger1")
            # Wait for the voltages to settle (depends on the channel bandwidth)
            wait(300 * u.us, "tank_circuit", "TIA", "P1_sticky", "P2_sticky")
            # Play the triangle once starting from 0V (then it will start from level_readout)
            for k in range(2):
                # Empty
                play("bias" * amp((level_empty[k] - 0) / P1_amp), f"P{k + 1}_sticky")
                wait(duration_empty, f"P{k + 1}_sticky")
                # Init
                play("bias" * amp((level_init[k] - level_empty[k]) / P1_amp), f"P{k + 1}_sticky")
                wait(duration_init, f"P{k + 1}_sticky")
                # Readout
                play("bias" * amp((level_readout[k] - level_init[k]) / P1_amp), f"P{k + 1}_sticky")
                wait(duration_readout, f"P{k + 1}_sticky")

            with for_(n, 0, n < n_avg, n + 1):  # The averaging loop
                for k in range(2):
                    # Empty
                    play("bias" * amp((level_empty[k] - level_readout[k]) / P1_amp), f"P{k+1}_sticky")
                    wait(duration_empty, f"P{k+1}_sticky")
                    # Init
                    play("bias" * amp((level_init[k] - level_empty[k]) / P1_amp), f"P{k+1}_sticky")
                    wait(duration_init, f"P{k+1}_sticky")
                    # Readout
                    play("bias" * amp((level_readout[k] - level_init[k]) / P1_amp), f"P{k+1}_sticky")
                    if k == 0:
                        align("P1_sticky", "tank_circuit", "TIA")
                    wait(duration_readout, f"P{k+1}_sticky")
                # RF reflectometry: the voltage measured by the analog input 2 is recorded, demodulated at the readout
                # frequency and the integrated quadratures are stored in "I" and "Q"
                I, Q, I_st, Q_st = RF_reflectometry_macro(I=I, Q=Q)
                # DC current sensing: the voltage measured by the analog input 1 is recorded and the integrated result
                # is stored in "dc_signal"
                dc_signal, dc_signal_st = DC_current_sensing_macro(dc_signal=dc_signal)
                # Wait at each iteration in order to ensure that the data will not be transferred faster than 1 sample
                # per µs to the stream processing. Otherwise, the processor will receive the samples faster than it can
                # process them which can cause the OPX to crash.
                wait(1_000 * u.ns)  # in ns
            # Ramp the voltage down to zero at the end of the triangle (needed with sticky elements)
            ramp_to_zero("P1_sticky")
            ramp_to_zero("P2_sticky")
        # Save the LO iteration to get the progress bar
        save(i, n_st)

    # Stream processing section used to process the data before saving it.
    with stream_processing():
        n_st.save("iteration")
        # Cast the data into a 2D matrix and performs a global averaging of the received 2D matrices together.
        # RF reflectometry
        I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("I")
        Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("Q")
        # DC current sensing
        dc_signal_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(n_points_fast).save_all("dc_signal")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

## QDAC2 section
# Create the qdac instrument
qdac = QDACII("Ethernet", IP_address="127.0.0.1", port=5025)  # Using Ethernet protocol
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
    voltage_list=voltage_values_fast,
)
load_voltage_list(
    qdac,
    channel=2,
    dwell=2e-6,
    slew_rate=2e7,
    trigger_port="ext2",
    output_range="high",
    output_filter="med",
    voltage_list=voltage_values_slow,
)

###########################
# Run or Simulate Program #
###########################
simulate = False

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
        # Convert results into Volts
        min_idx = min(I.shape[0], Q.shape[0])
        S = u.demod2volts(I[:min_idx, :] + 1j * Q[:min_idx, :], reflectometry_readout_length)
        R = np.abs(S)  # Amplitude
        phase = np.angle(S)  # Phase
        DC_signal = u.demod2volts(DC_signal, readout_len)
        # Progress bar
        progress_counter(iteration, n_points_slow)
        # Plot data
        plt.subplot(121)
        plt.cla()
        plt.title(r"$R=\sqrt{I^2 + Q^2}$ [V]")
        plt.pcolor(voltage_values_fast, voltage_values_slow[:min_idx], R)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.subplot(122)
        plt.cla()
        plt.title("Phase [rad]")
        plt.pcolor(voltage_values_fast, voltage_values_slow[:min_idx], phase)
        plt.xlabel("Fast voltage axis [V]")
        plt.ylabel("Slow voltage axis [V]")
        plt.tight_layout()
        plt.pause(0.1)