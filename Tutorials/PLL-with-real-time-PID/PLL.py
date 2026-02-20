from qm.qua import *
from qm import QuantumMachinesManager, SimulationConfig
from datetime import datetime
from qualang_tools.results import fetching_tool
import matplotlib.pyplot as plt
import numpy as np
from configuration import *

# -------------
# User options
# -------------
simulate = False
total_meas_time_ns = 0.001e9
num_points = int(total_meas_time_ns / readout_len)

# PID parameters

K_p = 0.1 * 0.45
K_i = 0.1*0.54/(36*1e-6)
K_d = 0.00
alpha = 0.5

# ----------
# Functions
# ----------
def get_time_spacing(qmm, config, num_points=10, K_p = 0.0, K_i = 0.0, K_d = 0.0, alpha = 0):
    """
    Executes a short program to retrieve timestamps and verify constant spacing.

    Returns spacing in ns.
    """
    with program() as check_timestamps:

        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st = declare_stream()
        eps = declare(fixed, value=0.0001)  # error tolerance
        phi_2pi_target = declare(fixed)  # phase in cycles (phase / 2π)
        phi_2pi_curr = declare(fixed)
        phi_2pi_curr_st = declare_stream()
        error = declare(fixed)
        error_st = declare_stream()
        error_integral = declare(fixed, value=0)
        error_integral_st = declare_stream()
        error_derivative = declare(fixed, value=0.0)
        error_derivative_st = declare_stream()
        control = declare(fixed, value=0)
        previous_error = declare(fixed, value=0.0)
        update_frequency("AOM", IF_AOM + 10 * 1e3)
        play("cw", "AOM", duration=readout_len // 4)
        measure('readout', 'Detector', demod.full("cos", I), demod.full("sin", Q))
        assign(phi_2pi_target, Math.atan2_2pi(Q, I))   # = atan2(Q,I) / (2π)

        #### pulse sequence ####
        with for_(n, 0, n < num_points, n + 1):
            # frame_rotation_2pi(0.01, "AOM")
            # --- QUA PLL loop with Proportional and Integral terms ---
            play("cw", "AOM", duration=readout_len // 4)
            # Measure current phase
            measure('readout', 'Detector', demod.full("cos", I), demod.full("sin", Q))
            # Measure current phase
            assign(phi_2pi_curr, Math.atan2_2pi(Q, I))  # = atan2(Q, I) / (2π)

            # Find the proportional error
            assign(error, phi_2pi_target - phi_2pi_curr)  # phase error (in cycles)
            assign(error, Util.cond(((error > -eps) & (error < eps)), 0.0, error))
            # find the integral error
            assign(error_integral, (1 - alpha) * error_integral + alpha * error)
            # Find the derivative error
            assign(error_derivative, previous_error - error)
            assign(previous_error, error)
            # Compute PID control value
            assign(control, K_p * error + K_i * error_integral + K_d * error_derivative)

            # Apply correction via frame rotation
            frame_rotation_2pi(control, "AOM")


            # Save data
            save(phi_2pi_curr, phi_2pi_curr_st)
            save(error, error_st)
            save(error_integral, error_integral_st)
            save(error_derivative, error_derivative_st)
            save(n, n_st)

        #### stream processing ####
        with stream_processing():
            phi_2pi_curr_st.with_timestamps().save_all("phase")
            error_st.save_all("error")
            error_integral_st.save_all("error_integral")
            error_derivative_st.save_all("error_derivative")

            n_st.save("iteration")

    qm = qmm.open_qm(config, close_other_machines=True)
    job = qm.execute(check_timestamps)

    res = job.result_handles
    res.wait_for_all_values()

    phase_t = res.phase.fetch_all()["timestamp"]

    diffs = np.diff(phase_t)

    if np.allclose(diffs, diffs[0]):
        print(f"Time between measurements = {diffs[0]}")
        return diffs[0]    # in clock cycles
    else:
        raise ValueError(f"Timestamps are not equally spaced! diffs={diffs}")

def phase_correction_with_PID(num_points, K_p, K_i, K_d, alpha):
    """Phase drift measurement program."""
    with program() as PLL_prog:
        #### QUA variable decleration ####

        n = declare(int)
        I = declare(fixed)
        Q = declare(fixed)
        n_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        eps = declare(fixed, value=0.0001) # error tolerance
        phi_2pi_target = declare(fixed) # phase in cycles (phase / 2π)
        phi_2pi_curr = declare(fixed)
        phi_2pi_curr_st = declare_stream()
        error = declare(fixed)
        error_st = declare_stream()
        error_integral = declare(fixed, value=0)
        error_integral_st = declare_stream()
        error_derivative = declare(fixed, value=0.0)
        error_derivative_st = declare_stream()
        control = declare(fixed,value=0)
        previous_error = declare(fixed, value=0.0)

        update_frequency("AOM", IF_AOM + 10 * 1e3)

        play("cw", "AOM", duration=readout_len // 4)
        measure('readout', 'Detector', demod.full("cos", I), demod.full("sin", Q))
        assign(phi_2pi_target, Math.atan2_2pi(Q, I))   # = atan2(Q,I) / (2π)

        #### pulse sequence ####
        with for_(n, 0, n < num_points, n + 1):
            # --- QUA PLL loop with Proportional and Integral terms ---
            # frame_rotation_2pi(0.001, "AOM")
            play("cw", "AOM", duration=readout_len // 4)
            # Measure current phase
            measure('readout', 'Detector', demod.full("cos", I), demod.full("sin", Q))
            # Measure current phase
            assign(phi_2pi_curr, Math.atan2_2pi(Q, I))   # = atan2(Q, I) / (2π)

            # Find the proportional error
            assign(error, phi_2pi_target - phi_2pi_curr)   # phase error (in cycles)
            assign(error, Util.cond(((error > -eps) & (error < eps)), 0.0, error))
            # find the integral error
            assign(error_integral, (1 - alpha) * error_integral + alpha * error)
            # Find the derivative error
            assign(error_derivative, previous_error - error)
            assign(previous_error, error)
            # Compute PID control value
            assign(control, K_p * error + K_i * error_integral + K_d * error_derivative)

            # Apply correction via frame rotation
            frame_rotation_2pi(control, "AOM")


            # Save data
            save(phi_2pi_curr, phi_2pi_curr_st)
            save(error, error_st)
            save(error_integral, error_integral_st)
            save(error_derivative, error_derivative_st)
            save(I, I_st)
            save(Q, Q_st)
            save(n, n_st)

        #### stream processing ####
        with stream_processing():
            phi_2pi_curr_st.save_all("phase")
            error_st.save_all("error")
            error_integral_st.save_all ("error_integral")
            error_derivative_st.save_all("error_derivative")
            I_st.save_all("I")
            Q_st.save_all("Q")
            n_st.save("iteration")


    return PLL_prog

# ---------
# Plotting
# ---------
def plot_phase_pid_results(time_s, phase, error, error_integral, error_derivative, out_path=None):
        """
        Plots phase vs time, error vs time, error integral vs time, and error derivative vs time
        in four subplots (vertically stacked).

        Parameters:
            time_s (np.ndarray): Array of time points in seconds.
            phase (np.ndarray): Measured phase values.
            error (np.ndarray): Error values.
            error_integral (np.ndarray): Integral of error (typically for PID control).
            error_derivative (np.ndarray): Derivative of error (typically for PID control).
            out_path (str or Path, optional): If provided, saves plot to this path.
        """

        fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 8))

        axs[0].plot(time_s, phase, label="Phase")
        axs[0].set_ylabel("Phase [rad]")
        axs[0].set_title("Phase vs Time")
        axs[0].grid(True)
        axs[0].legend()

        axs[1].plot(time_s, error, color="C1", label="Error")
        axs[1].set_ylabel("Error [rad]")
        axs[1].set_title("Error vs Time")
        axs[1].grid(True)
        axs[1].legend()

        axs[2].plot(time_s, error_integral, color="C2", label="Error Integral")
        axs[2].set_ylabel("Error Integral [rad]")
        axs[2].set_title("Error Integral vs Time")
        axs[2].grid(True)
        axs[2].legend()

        axs[3].plot(time_s, error_derivative, color="C3", label="Error Derivative")
        axs[3].set_xlabel("Time [ms]")
        axs[3].set_ylabel("Error Derivative [rad/iteration]")
        axs[3].set_title("Error Derivative vs Time")
        axs[3].grid(True)
        axs[3].legend()

        plt.tight_layout()
        if out_path is not None:
            plt.savefig(out_path)
            print(f"Saved phase PID results plot: {out_path}")
        plt.show()

# ---------
# Saving
# ---------
def save_data(out_path, t_s, phase, error, error_integral, error_derivative, tau_s=None):
    """
    Save timetagged IQ data in an ASCII format Stable32 can read.
    Columns: time(s)  I(V)  Q(V)
    Header lines start with '#' (ignored by Stable32).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    header_lines = [
        "QM OPX exported data",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "Columns: time_s  phase  error error_integral",
    ]
    if tau_s is not None:
        header_lines.append(f"Tau: {tau_s/1e9:.12g} s")  # optional hint; Stable32 can also infer from timetags

    header = "\n".join("# " + line for line in header_lines)

    data = np.column_stack([t_s, phase, error, error_integral, error_derivative])
    np.savetxt(out_path, data, header=header, comments="", fmt="%.12e")
    print(f"Saved file: {out_path}")


# -----
# Main
# -----
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

print("-" * 70)
print("Step 21: Measuring time spacing between measurements")
print("-" * 70)
spacing = get_time_spacing(qmm, config, num_points=100, K_p=K_p, K_i=K_i, K_d=K_d, alpha=alpha)
print(f"Timestamp spacing = {spacing} ns")

print("-" * 70)
print(f"Step 3: Measuring phase drift for {total_meas_time_ns / 1e9:.2f} s")
print(f"Measurement duration = {readout_len}")
print("-" * 70)

PLL_prog  = phase_correction_with_PID(num_points, K_p=K_p, K_i=K_i, K_d=K_d, alpha=alpha)
if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=2_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, PLL_prog , simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    samples.con1.plot()
    # Get the waveform report object

    show_waveform_report = False
    if show_waveform_report:
        waveform_report = job.get_simulated_waveform_report()
        # Cast the waveform report to a python dictionary
        waveform_dict = waveform_report.to_dict()
        # Visualize and save the waveform report
        waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))

else:
    # Open the quantum machine
    qm = qmm.open_qm(config, close_other_machines=True)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(PLL_prog)
    # Creates a result handle to fetch data from the OPX
    results = fetching_tool(job, data_list=["phase", "error", "error_integral", "error_derivative", "I", "Q", "iteration"], mode="wait_for_all")

    time_ms = np.arange(num_points) * spacing * 1e-6

    phase_2pi, error, error_integral, error_derivative, I, Q, iteration = results.fetch_all()
    error *= 2 * np.pi
    error_integral *= 2 * np.pi
    error_derivative *= 2 * np.pi
    I = u.demod2volts(I, readout_len)
    Q = u.demod2volts(Q, readout_len)
    S = I + 1j * Q
    phase = np.unwrap(np.angle(S))
    # Path
    out_dir = Path(__file__).resolve().parent
    out_path = out_dir / f"PLL_with_PID_for_{total_meas_time_ns / 1e9:.0f}s.png"
    # Plotting
    plot_phase_pid_results(time_ms, phase, error, error_integral, error_derivative)

    out_dir = Path(__file__).resolve().parent
    fname = out_dir / f"PLL_with_PID_for_{total_meas_time_ns / 1e9:.0f}s.txt"
    save_data(fname, time_ms, phase, error, error_integral, error_derivative, tau_s=spacing)
    print(f"Total measurement duration: {(num_points * spacing) / 1e9} s ")
    qm.close()