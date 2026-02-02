"""
        CPMG (Carr-Purcell-Meiboom-Gill) MEASUREMENT
The program consists in playing a CPMG dynamical decoupling sequence to measure and extend the qubit's coherence time.
The sequence is: x90 - [idle_time - y180 - idle_time]xN - -x90 - measurement, where N is the number of refocusing pulses.

Unlike the standard Hahn echo which uses a single x180 pulse, CPMG uses multiple y180 pulses which provides better
protection against pulse errors and can extend coherence times by filtering out low-frequency noise.

The program sweeps both the idle time (tau) and the number of pi pulses (N) to characterize:
1. The T2_CPMG coherence time for different numbers of refocusing pulses
2. How T2 scales with N (provides information about the noise spectrum)

From the results, one can fit the exponential decay for each N and extract T2_CPMG(N).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi and pi/2 pulses (x90, x180, y180) by running qubit spectroscopy, rabi_chevron, power_rabi.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *

from configuration import *
from qualang_tools.loops import from_array
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.results.data_handler import DataHandler

##################
#   Parameters   #
##################
# Number of averages
n_avg = 1000

# Idle time (tau) sweep parameters - tau is half the time between successive pi pulses
# Sweep is in clock cycles (1 clock cycle = 4ns) - minimum is 4 clock cycles
tau_min = 4  # Minimum tau in clock cycles
tau_max = 2500  # Maximum tau in clock cycles (10 us)
d_tau = 100  # Step size in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau).astype(int)
# For logarithmic sweep, uncomment below:
# taus = np.logspace(np.log10(tau_min), np.log10(tau_max), 29).astype(int)

# Number of pi pulses (N) to sweep - CPMG order
# Common values: 1 (Hahn echo), 2, 4, 8, 16, 32...
n_pi_values = np.array([1, 2, 4, 8, 16], dtype=int)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "taus": taus,
    "n_pi_values": n_pi_values,
    "config": config,
}

###################
# The QUA program #
###################
with program() as cpmg:
    # Declare QUA variables
    n = declare(int)  # QUA variable for the averaging loop
    n_st = declare_stream()  # Stream for the averaging iteration
    tau = declare(int)  # QUA variable for the idle time (half time between pi pulses)
    n_pi = declare(int)  # QUA variable for the number of pi pulses
    i = declare(int)  # QUA variable for the CPMG refocusing loop
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature

    with for_(n, 0, n < n_avg, n + 1):
        # Sweep over number of pi pulses (CPMG order)
        with for_(*from_array(n_pi, n_pi_values)):
            # Sweep over idle times (tau)
            with for_(*from_array(tau, taus)):
                # CPMG Sequence: x90 - [tau - y180 - tau]xN - -x90 - measure

                # Initial x90 pulse to create superposition
                play("x90", "qubit")

                # CPMG refocusing loop: N repetitions of (tau - y180 - tau)
                with for_(i, 0, i < n_pi, i + 1):
                    # Wait for idle time tau
                    wait(tau, "qubit")
                    # Apply y180 refocusing pulse (Y-axis for CPMG)
                    play("y180", "qubit")
                    # Wait for idle time tau
                    wait(tau, "qubit")

                # Final -x90 pulse to project back
                play("-x90", "qubit")

                # Align qubit and resonator for measurement
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
        # Cast the data into a 2D array [n_pi_values x taus], average and save
        I_st.buffer(len(taus)).buffer(len(n_pi_values)).average().save("I")
        Q_st.buffer(len(taus)).buffer(len(n_pi_values)).average().save("Q")
        n_st.save("iteration")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name)

###########################
# Run or Simulate Program #
###########################
simulate = True

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    # Simulate blocks python until the simulation is done
    job = qmm.simulate(config, cpmg, simulation_config)
    # Get the simulated samples
    samples = job.get_simulated_samples()
    # Plot the simulated samples
    plt.figure()
    samples.con1.plot()
    plt.title("CPMG Simulated Waveforms")
    plt.tight_layout()

    # Get the waveform report object
    waveform_report = job.get_simulated_waveform_report()
    # Cast the waveform report to a python dictionary
    waveform_dict = waveform_report.to_dict()
    # Visualize and save the waveform report
    waveform_report.create_plot(samples, plot=True, save_path=str(Path(__file__).resolve()))
    plt.show()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(cpmg)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["I", "Q", "iteration"], mode="live")

    # Live plotting
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    interrupt_on_close(fig, job)  # Interrupts the job when closing the figure

    # Calculate total evolution time for each (n_pi, tau) combination
    # Total time = 2 * n_pi * tau * 4ns (convert from clock cycles to ns)
    evolution_times = 8 * np.outer(n_pi_values, taus)  # Shape: [n_pi_values, taus] in ns

    while results.is_processing():
        # Fetch results
        I, Q, iteration = results.fetch_all()
        # Convert the results into Volts
        I_volts = u.demod2volts(I, readout_len)
        Q_volts = u.demod2volts(Q, readout_len)

        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        # Plot results for each number of pi pulses
        axes[0].cla()
        axes[1].cla()

        for idx, n_pi_val in enumerate(n_pi_values):
            # Total evolution time for this n_pi value
            t_evolution = evolution_times[idx, :] * 1e-3  # Convert to microseconds
            axes[0].plot(t_evolution, I_volts[idx, :], ".-", label=f"N={n_pi_val}")
            axes[1].plot(t_evolution, Q_volts[idx, :], ".-", label=f"N={n_pi_val}")

        axes[0].set_ylabel("I quadrature [V]")
        axes[0].legend(loc="upper right")
        axes[0].set_title(f"CPMG measurement (iteration {iteration}/{n_avg})")

        axes[1].set_xlabel("Total evolution time [us]")
        axes[1].set_ylabel("Q quadrature [V]")
        axes[1].legend(loc="upper right")

        plt.tight_layout()
        plt.pause(2)

    # Fit the results to extract T2_CPMG for each number of pi pulses
    try:
        from qualang_tools.plot.fitting import Fit

        fit = Fit()
        T2_cpmg_values = []

        # Create a new figure for the fitted results
        fig_fit, axes_fit = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: I quadrature with fits for each N
        ax1 = axes_fit[0]
        for idx, n_pi_val in enumerate(n_pi_values):
            t_evolution = evolution_times[idx, :]  # in ns
            I_data = I_volts[idx, :]

            # Fit exponential decay (T1 fit works for any exponential decay)
            try:
                fit_result = fit.T1(t_evolution, I_data, plot=False)
                T2_cpmg = np.abs(fit_result["T1"][0])
                T2_cpmg_values.append(T2_cpmg)

                # Plot data and fit
                ax1.plot(t_evolution * 1e-3, I_data, "o", label=f"N={n_pi_val}")
                t_fit = np.linspace(t_evolution.min(), t_evolution.max(), 200)
                I_fit = fit_result["fit_func"](t_fit)
                ax1.plot(t_fit * 1e-3, I_fit, "-", label=f"T2={T2_cpmg*1e-3:.1f} us")
            except Exception as e:
                print(f"Fitting failed for N={n_pi_val}: {e}")
                T2_cpmg_values.append(np.nan)
                ax1.plot(t_evolution * 1e-3, I_data, "o", label=f"N={n_pi_val}")

        ax1.set_xlabel("Total evolution time [us]")
        ax1.set_ylabel("I quadrature [V]")
        ax1.set_title("CPMG: I quadrature decay")
        ax1.legend(loc="upper right", fontsize=8)

        # Plot 2: T2_CPMG vs number of pi pulses (N)
        ax2 = axes_fit[1]
        valid_mask = ~np.isnan(T2_cpmg_values)
        ax2.plot(n_pi_values[valid_mask], np.array(T2_cpmg_values)[valid_mask] * 1e-3, "o-", markersize=10)
        ax2.set_xlabel("Number of pi pulses (N)")
        ax2.set_ylabel("T2_CPMG [us]")
        ax2.set_title("T2 coherence time vs CPMG order")
        ax2.set_xscale("log", base=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Print results
        print("\n" + "=" * 50)
        print("CPMG Results Summary")
        print("=" * 50)
        for n_pi_val, T2 in zip(n_pi_values, T2_cpmg_values):
            if not np.isnan(T2):
                print(f"N = {n_pi_val:3d} pi pulses: T2_CPMG = {T2*1e-3:.2f} us ({T2:.0f} ns)")
            else:
                print(f"N = {n_pi_val:3d} pi pulses: Fit failed")
        print("=" * 50)

    except (Exception,) as e:
        print(f"Fitting module not available or fitting failed: {e}")
        T2_cpmg_values = []
        fig_fit = None

    # Save results
    script_name = Path(__file__).name
    data_handler = DataHandler(root_data_folder=save_dir)

    # Update save dictionary with measured data
    save_data_dict.update({"I_data": I_volts})
    save_data_dict.update({"Q_data": Q_volts})
    save_data_dict.update({"evolution_times_ns": evolution_times})
    save_data_dict.update({"T2_cpmg_values_ns": T2_cpmg_values})
    save_data_dict.update({"fig_live": fig})
    if fig_fit is not None:
        save_data_dict.update({"fig_fit": fig_fit})

    # Add additional files to save alongside the data
    data_handler.additional_files = {script_name: script_name, **default_additional_files}

    # Save all data
    data_handler.save_data(data=save_data_dict, name="cpmg")

    print(f"\nData saved to: {save_dir}")
