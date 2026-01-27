"""
        XY8 DYNAMICAL DECOUPLING MEASUREMENT
The program consists in playing an XY8 dynamical decoupling sequence to measure and extend the qubit's coherence time.
The XY8 sequence uses an 8-pulse pattern (X-Y-X-Y-Y-X-Y-X) that provides superior protection against both pulse errors
and low-frequency noise compared to simpler sequences like CPMG or Hahn echo.

The sequence is: x90 - [tau - X180 - 2*tau - Y180 - 2*tau - X180 - 2*tau - Y180 - 2*tau - Y180 - 2*tau - X180 - 2*tau - Y180 - 2*tau - X180 - tau]xN - -x90 - measurement
where N is the number of XY8 repetitions (each containing 8 pi pulses).

The XY8 sequence is particularly effective because:
1. The alternating X and Y pulses cancel out systematic pulse errors
2. The symmetric pattern (X-Y-X-Y-Y-X-Y-X) provides robustness against both amplitude and phase errors
3. It acts as a bandpass filter for noise, extending coherence times in environments with low-frequency noise

The program sweeps both the idle time (tau) and the number of XY8 repetitions (N) to characterize:
1. The T2_XY8 coherence time for different numbers of XY8 cycles
2. How T2 scales with N (provides information about the noise spectrum)

From the results, one can fit the exponential decay for each N and extract T2_XY8(N).

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi and pi/2 pulses (x90, x180, y180, -x90) by running qubit spectroscopy, rabi_chevron, power_rabi.
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

# Idle time (tau) sweep parameters - tau is the time between successive pi pulses
# Sweep is in clock cycles (1 clock cycle = 4ns) - minimum is 4 clock cycles
tau_min = 4  # Minimum tau in clock cycles
tau_max = 1250  # Maximum tau in clock cycles (5 us)
d_tau = 50  # Step size in clock cycles
taus = np.arange(tau_min, tau_max + 0.1, d_tau).astype(int)
# For logarithmic sweep, uncomment below:
# taus = np.logspace(np.log10(tau_min), np.log10(tau_max), 29).astype(int)

# Number of XY8 repetitions (N) to sweep
# Each XY8 cycle contains 8 pi pulses
# Common values: 1, 2, 4, 8, 16...
n_xy8_values = np.array([1, 2, 4, 8], dtype=int)

# Data to save
save_data_dict = {
    "n_avg": n_avg,
    "taus": taus,
    "n_xy8_values": n_xy8_values,
    "config": config,
}
