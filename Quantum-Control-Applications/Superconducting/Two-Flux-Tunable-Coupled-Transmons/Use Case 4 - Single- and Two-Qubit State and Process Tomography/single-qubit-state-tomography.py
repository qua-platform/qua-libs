#!/usr/bin/env python

"""
        SINGLE QUBIT STATE TOMOGRAPHY
The sequence consists of preparing the qubit into a chosen state using calibrated gates from the config, and
measuring the state of the qubit, by way of the readout resonator, in the X, Y and Z bases. The output,
'probs', is a triple of probability 'difference' counts for measuring the qubit in one eigenstate of the
X/Y/Z bases, minus the probability of measuring it in that eigenstate's complement.

These values are scaled into the usual Stokes parameters and used to infer the state of the qubit; see
https://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf for further details

Note that this program is similar to
qua-libs/Quantum-Control-Applications/Superconducting/Single-Fixed-Transmon
/19_state_tomography.py, which the author became aware of after having
written the current script

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy)
    - Having calibrated qubit pi and pi/2 pulses by running qubit spectroscopy, rabi_chevron, power_rabi and updating the config
    - Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR, and having
      saved the derived readout threshold value in the config
    - Set the desired flux bias in the case of flux-tunable qubits

This script implements the logic outlined in the notebook
https://github.com/bornman-nick/quantum-state-and-process-tomography. See that
notebook for a detailed explanation of standard single qubit state
tomography.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
import numpy as np


###################
# The QUA program #
###################

# qubit under test, assuming there are multiple qubits
# with XY line elements "q<qubit>_xy", flux lines
# "q<qubit>_z", and readout resonator elements "rr<qubit>"
qubit = 1

if qubit == 1:
    threshold = ge_threshold_q1
elif qubit == 2:
    threshold = ge_threshold_q2
else:
    raise ValueError(f"Incorrect qubit number chosen")


n_avg = 10_000


# subroutine to prepare desired qubit state
def prepare_state(qubit):
    # write whatever QUA code you need in order to create the
    # state to perform tomography on, from an initial
    # ground state. For example, to create the |1> state
    play("y180", f"q{qubit}_xy")


with program() as single_qubit_state_tomography:
    n = declare(int)  # QUA variable for average loop
    n_st = declare_stream()  # Stream for the averaging iteration 'n'

    state = declare(bool)  # QUA variable for the qubit state
    state_st = declare_stream()  # Stream for the qubit state

    p = declare(int)  # QUA variable for switching between projections

    I = declare(fixed)
    Q = declare(fixed)

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(p, 0, p <= 2, p + 1):  # QUA for_ loop for switching between basis changes

            prepare_state(qubit)
            align()

            with switch_(c):
                with case_(0):  # basis X

                    # Map the X-component of the Bloch vector onto the Z-axis
                    # 1/sqrt(2)(|0>+|1>) -> |0>; 1/sqrt(2)(|0>-|1>) -> |1>
                    play("-y90", f"q{qubit}_xy")

                    align(f"q{qubit}_xy", f"rr{qubit}")

                with case_(1):  # basis Y

                    # Map the Y-component of the Bloch vector onto the Z-axis
                    # 1/sqrt(2)(|0>+i|1>) -> |0>; 1/sqrt(2)(|0>-i|1>) -> |1>
                    play("x90", f"q{qubit}_xy")

                    align(f"q{qubit}_xy", f"rr{qubit}")

                with case_(2):  # basis Z

                    align(f"q{qubit}_xy", f"rr{qubit}")

                measure(
                    "readout",
                    f"rr{qubit}",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )

            # True if qubit state is |1>, False if |0>

            assign(state, I > threshold)

            wait(thermalization_time * u.ns)

            save(state, state_st)

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(3).average().save("probs")


#####################################
#  Open Communication with the QOP  #
#####################################
qmm = QuantumMachinesManager(host=qop_ip, port=qop_port, cluster_name=cluster_name, octave=octave_config)

###########################
# Run or Simulate Program #
###########################
simulate = False

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, single_qubit_state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(single_qubit_state_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["probs", "iteration"], mode="live")

    while results.is_processing():
        # Fetch results
        probs, iteration = results.fetch_all()
        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

        # Converts the (0,1) -> |g>,|e> convention, arising from the I>threshold
        # assignment, to (1,-1) -> |g>,|e>, which aligns with the Stokes parameter
        # definitions from the projector probabilities
        prob = -2 * (probs - 0.5)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # Reconstruct the density matrix
    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    rho = 0.5 * (I + prob[0] * sigma_x + prob[1] * sigma_y + prob[2] * sigma_z)
    print(f"The density matrix is:\n{rho}")
