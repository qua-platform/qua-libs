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
