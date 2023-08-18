"""
        SINGLE QUBIT STATE TOMOGRAPHY
The goal of this program is to measure the projection of the Bloch vector of the qubit along the three axes of the Bloch 
sphere in order to reconstruct the full qubit state tomography.
The qubit state preparation is left to user to define.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having calibrated the IQ blobs to perform state discrimination.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization) for better SNR.
"""
import matplotlib.pyplot as plt
from qm import SimulationConfig
from qm.qua import *
from qm.QuantumMachinesManager import QuantumMachinesManager
from configuration import *
import numpy as np
from macros import readout_macro

###################
# The QUA program #
###################

n_avg = 100

with program() as state_tomography:
    n = declare(int)  # QUA variable for average loop
    state = declare(bool)  # QUA variable for the qubit state
    c = declare(int)  # QUA variable for switching between projections
    state_st = declare_stream()  # Stream for the qubit state
    n_st = declare_stream()  # Stream for the averaging iteration 'n'
    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(c, 0, c <= 2, c + 1):  # QUA for_ loop for switching between projections
            # Add here whatever state you want to characterize
            with switch_(c):
                with case_(0):  # projection along X
                    # Map the X-component of the Bloch vector onto the Z-axis (measurement axis)
                    play("-y90", "qubit")
                    # Align the two elements to measure after playing the qubit pulses.
                    align("qubit", "resonator")
                    # Measure the resonator and extract the qubit state
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    # Wait for the qubit to decay to the ground state
                    wait(thermalization_time * u.ns, "resonator")
                    # Save the qubit state to its stream
                    save(state, state_st)
                with case_(1):  # projection along Y
                    # Map the Y-component of the Bloch vector onto the Z-axis (measurement axis)
                    play("x90", "qubit")
                    # Align the two elements to measure after playing the qubit pulses.
                    align("qubit", "resonator")
                    # Measure the resonator and extract the qubit state
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    # Wait for the qubit to decay to the ground state
                    wait(thermalization_time * u.ns, "resonator")
                    # Save the qubit state to its stream
                    save(state, state_st)
                with case_(2):  # projection along Z
                    # Measure the Z-component of the Bloch vector
                    state, _, _ = readout_macro(threshold=ge_threshold, state=state)
                    # Wait for the qubit to decay to the ground state
                    wait(thermalization_time * u.ns, "resonator")
                    # Save the qubit state to its stream
                    save(state, state_st)
        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(3).average().save("states")

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
    job = qmm.simulate(config, state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(state_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["states", "iteration"], mode="live")
    while results.is_processing():
        # Fetch results
        state, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())
        # Converts the (0,1) -> |g>,|e> convention to (1,-1) -> |g>,|e>
        states = -2 * (state - 0.5)

    # Derive the density matrix
    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])
    # Zero order approximation
    rho = 0.5 * (I + state[0] * sigma_x + state[1] * sigma_y + state[2] * sigma_z)
    print(f"The density matrix is:\n{rho}")
