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
