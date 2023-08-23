"""
        AC STARK-SHIFT CALIBRATION WITH DRAG PULSES 1D (GOOGLE METHOD)
The sequence consists in applying an given number of x180 and -x180 pulses successively for different DRAG detunings.
Here the detuning sweep has to be performed in python, because it involves changing the DRAG waveforms in a
non-linear manner.
After such a sequence, the qubit is expected to always be in the ground state if the AC Stark shift is
properly compensated by the DRAG detuning.
One can fit the final results with an inverted parabola to precisely determined the optimal detuning and update it in
the configuration.

This protocol is described in more details in https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Having calibrated the DRAG coefficient.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update the DRAG detuning (AC_stark_detuning) in the configuration.
"""

from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import matplotlib.pyplot as plt
import numpy as np
from qm import SimulationConfig
from macros import readout_macro
import warnings

warnings.filterwarnings("ignore")

###################
# The QUA program #
###################

n_avg = 1000
number_of_pulses = 20
# Detuning to compensate for the AC STark-shift
detunings = np.arange(-3e6, 0e6, 0.1e6)

with program() as ac_stark_shift:
    n = declare(int)  # QUA variable for the averaging loop
    it = declare(int)  # QUA variable for the number of qubit pulses
    I = declare(fixed)  # QUA variable for the measured 'I' quadrature
    Q = declare(fixed)  # QUA variable for the measured 'Q' quadrature
    state = declare(bool)  # QUA variable for the qubit state
    I_st = declare_stream()  # Stream for the 'I' quadrature
    Q_st = declare_stream()  # Stream for the 'Q' quadrature
    state_st = declare_stream()  # Stream for the qubit state

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        # Loop for error amplification (perform many qubit pulses with varying DRAG coefficients)
        with for_(it, 0, it < number_of_pulses, it + 1):
            play("x180" * amp(1), "qubit")
            play("x180" * amp(-1), "qubit")
        # Align the two elements to measure after playing the qubit pulses.
        align("qubit", "resonator")
        # Measure the resonator and extract the qubit state
        state, I, Q = readout_macro(threshold=ge_threshold, state=state, I=I, Q=Q)
        # Wait for the qubit to decay to the ground state
        wait(thermalization_time * u.ns, "resonator")
        # Save the 'I' & 'Q' quadratures to their respective streams
        save(I, I_st)
        save(Q, Q_st)
        save(state, state_st)

    with stream_processing():
        # Since the Stark shift is swept in Python, we just need to stream 1 point per OPX run.
        I_st.average().save("I")
        Q_st.average().save("Q")
        state_st.boolean_to_int().average().save("state")

#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(qop_ip, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ac_stark_shift, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    xaxis = []
    I_tot = []
    Q_tot = []
    state_tot = []
    plt.figure()
    # Since the DRAG waveforms need to be changed, we have to do it in Python
    for det in detunings:
        xaxis.append(det)
        # Derive the DRAG waveforms with the new detuning
        x180_wf, x180_der_wf = np.array(
            drag_gaussian_pulse_waveforms(
                x180_amp, x180_len, x180_sigma, alpha=drag_coef, anharmonicity=anharmonicity, detuning=det
            )
        )
        x180_I_wf = x180_wf
        x180_Q_wf = x180_der_wf
        # Update the config
        config["waveforms"]["x180_I_wf"]["samples"] = x180_I_wf.tolist()
        config["waveforms"]["x180_Q_wf"]["samples"] = x180_Q_wf.tolist()
        # Open the quantum machine with the updated config
        qm = qmm.open_qm(config)
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(ac_stark_shift)
        # Get results from QUA program
        results = fetching_tool(job, data_list=["I", "Q", "state"])
        # Fetch results
        I, Q, state = results.fetch_all()
        I_tot.append(I)
        Q_tot.append(Q)
        state_tot.append(state)
        # Plot results
        plt.suptitle("AC stark shift calibration")
        plt.subplot(311)
        plt.cla()
        plt.plot(xaxis, I_tot, "o")
        plt.ylabel("I [a.u.]")
        plt.subplot(312)
        plt.cla()
        plt.plot(xaxis, Q_tot, "o")
        plt.ylabel("Q [a.u.]")
        plt.subplot(313)
        plt.cla()
        plt.plot(xaxis, state_tot, "o")
        plt.xlabel("Detuning [Hz]")
        plt.ylabel("state")
        plt.tight_layout()
        plt.pause(0.01)
