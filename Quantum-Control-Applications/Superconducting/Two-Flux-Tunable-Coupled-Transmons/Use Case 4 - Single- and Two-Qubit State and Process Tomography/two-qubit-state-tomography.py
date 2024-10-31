#!/usr/bin/env python

"""
        TWO QUBIT STATE TOMOGRAPHY
The sequence consists of preparing the qubits into a chosen state using
calibrated gates from the config, and measuring the state of the qubits, by way
of the readout resonator, in their X, Y and Z bases. The output, 'probs', is a
list of 15 probability 'difference' counts corresponding with the 15 Stokes
parameters for two qubit state tomography. These values are used to infer the
two-qubit state; see
https://research.physics.illinois.edu/QI/Photonics/tomography-files/tomo_chapter_2004.pdf
for further details.

Prerequisites:
    - Having found the resonance frequencies of the resonators coupled to the
      qubits under study (resonator_spectroscopy)
    - Having calibrated qubits' pi and pi/2 pulses by running qubit
      spectroscopy, rabi_chevron, power_rabi and updating the config
    - Having calibrated the readout (readout_frequency, amplitude,
      duration_optimization IQ_blobs) for each qubit
      for better SNR, and having saved the derived readout threshold values in
      the config
    - Set the desired flux biases in the case of flux-tunable qubits

This script implements the logic outlined in the notebook
https://github.com/bornman-nick/quantum-state-and-process-tomography. See that
notebook for a detailed explanation of standard two-qubit state tomography
"""

import numpy as np
from scipy.linalg import sqrtm

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool

from helper_functions import rotated_multiplexed_state_discrimination


###################
# The QUA program #
###################

# qubits under test, assuming there are multiple qubits
# with XY line elements "q<qubit>_xy", flux lines
# "q<qubit>_z", and readout resonator elements "rr<qubit>"

qubit1 = 1
qubit2 = 2

if qubit1 == 1:
    threshold1 = ge_threshold_q1
elif qubit1 == 2:
    threshold1 = ge_threshold_q2
else:
    raise ValueError(f"Incorrect qubit1 number chosen")

if qubit2 == 1:
    threshold2 = ge_threshold_q1
elif qubit2 == 2:
    threshold2 = ge_threshold_q2
else:
    raise ValueError(f"Incorrect qubit2 number chosen")

if qubit1 == qubit2:
    raise ValueError(f"The value of qubit1 cannot equal that of qubit2")


n_avg = 10_000


# subroutine to prepare desired qubit state
def prepare_state(qubit1, qubit2):
    # write whatever QUA code you need in order to create the
    # state to perform tomography on, from an initial
    # ground state. For example, to create the |1> 1/sqrt(2)(|0>+i|1>)
    # state
    play("y180", f"q{qubit1}_xy")
    play("-x90", f"q{qubit2}_xy")


# matrix representation of the ideal composite qubit state from above
# (|1> 1/sqrt(2)(|0>+i|1>)) (<1| 1/sqrt(2)(<0|-i<1|))
ideal_state = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1 / 2, 1j / 2], [0, 0, -1j / 2, 1 / 2]])


with program() as two_qubit_state_tomography:

    n = declare(int)
    n_st = declare_stream()
    I = [declare(fixed) for _ in range(2)]
    Q = [declare(fixed) for _ in range(2)]
    # I_st = [declare_stream() for _ in range(2)]
    # Q_st = [declare_stream() for _ in range(2)]

    states = [
        declare(bool),
        declare(bool),
    ]  # QUA variable for the measured qubit states
    # states_st = [declare_stream(), declare_stream()]  # Stream for the qubits states

    p00 = declare(int)  # variable to track number of |0>|0> measurements
    p01 = declare(int)  # variable to track number of |0>|1> measurements
    p10 = declare(int)  # variable to track number of |1>|0> measurements
    p11 = declare(int)  # variable to track number of |1>|1> measurements

    prob_vec_results_st = declare_stream()  # Stream for the probability vector - average of states vector

    c = declare(int)  # QUA variable for switching between projections

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(c, 1, c <= 15, c + 1):  # QUA for_ loop for switching between projections

            prepare_state(qubit1, qubit2)
            align()

            assign(p00, 0)
            assign(p01, 0)
            assign(p10, 0)
            assign(p11, 0)

            with switch_(c):
                with case_(1):  # projection along Z1, X2

                    play("-y90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P10 - P01 - P11

                with case_(2):  # projection along Z1, Y2

                    play("x90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P10 - P01 - P11

                with case_(3):  # projection along Z1, Z2

                    wait(pi_len // 4, f"q{qubit1}_xy", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P10 - P01 - P11

                with case_(4):  # projection along X1, Z2

                    play("-y90", f"q{qubit1}_xy")
                    # Stokes parameter for this case: P00 + P01 - P10 - P11

                with case_(5):  # projection along X1, X2

                    play("-y90", f"q{qubit1}_xy")
                    play("-y90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(6):  # projection along X1, Y2

                    play("-y90", f"q{qubit1}_xy")
                    play("x90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(7):  # projection along X1, Z2

                    play("-y90", f"q{qubit1}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(8):  # projection along Y1, Z2

                    play("x90", f"q{qubit1}_xy")
                    # Stokes parameter for this case: P00 + P01 - P10 - P11

                with case_(9):  # projection along Y1, X2

                    play("x90", f"q{qubit1}_xy")
                    play("-y90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(10):  # projection along Y1, Y2

                    play("x90", f"q{qubit1}_xy")
                    play("x90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(11):  # projection along Y1, Z2

                    play("x90", f"q{qubit1}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(12):  # projection along Z1, Z2

                    wait(pi_len // 4, f"q{qubit1}_xy", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P01 - P10 - P11

                with case_(13):  # projection along Z1, X2

                    play("-y90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(14):  # projection along Z1, Y2

                    play("x90", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                with case_(15):  # projection along Z1, Z2

                    wait(pi_len // 4, f"q{qubit1}_xy", f"q{qubit2}_xy")
                    # Stokes parameter for this case: P00 + P11 - P10 - P01

                align()

            rotated_multiplexed_state_discrimination(
                I,
                None,
                Q,
                None,
                states,
                None,
                [qubit1, qubit2],
                [threshold1, threshold2],
            )

            align()

            # populate correct variable depending on which qubit states
            # were measured
            with if_(states[0]):
                with if_(states[1]):
                    assign(p11, 1)  # |1>|1> was measured
                with else_():
                    assign(p10, 1)  # |1>|0> was measured
            with else_():
                with if_(states[1]):
                    assign(p01, 1)  # |0>|1> was measured
                with else_():
                    assign(p00, 1)  # |0>|0> was measured

            # stream all four possible values - only a single one of these
            # four should be 1, the rest, 0
            save(p00, prob_vec_results_st)
            save(p01, prob_vec_results_st)
            save(p10, prob_vec_results_st)
            save(p11, prob_vec_results_st)

            wait(thermalization_time * u.ns)

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        prob_vec_results_st.buffer(4).buffer(15).average().save("probs")


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
    job = qmm.simulate(config, two_qubit_state_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(two_qubit_state_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["probs", "iteration"], mode="live")

    while results.is_processing():

        # Fetch results
        iteration, probs = results.fetch_all()

        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # Close the quantum machines at the end in order to put all flux biases to
    # 0 so that the fridge doesn't heat-up
    qm.close()

    # Use probs to reconstruct density matrix

    # initialise vector to contain the 15 stokes parameters
    stokes = np.zeros(15)

    # For cases 1, 2,and 3 - Stokes parameter is P00 + P10 - P01 - P11
    # For cases 4, 8 and 12 - Stokes parameter is P00 + P01 - P10 - P11
    # For cases 5, 6, 7, 9, 10, 11, 13, 14, 15 - P00 + P11 - P10 - P01
    for i in range(1, 16):
        if i in [1, 2, 3]:
            stokes[i - 1] = probs[i - 1][0] + probs[i - 1][2] - probs[i - 1][1] - probs[i - 1][3]
        elif i in [4, 8, 12]:
            stokes[i - 1] = probs[i - 1][0] + probs[i - 1][1] - probs[i - 1][2] - probs[i - 1][3]
        elif i in [5, 6, 7, 9, 10, 11, 13, 14, 15]:
            stokes[i - 1] = probs[i - 1][0] + probs[i - 1][3] - probs[i - 1][1] - probs[i - 1][2]

    # Derive the density matrix
    I = np.array([[1, 0], [0, 1]])
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])
    sigma_z = np.array([[1, 0], [0, -1]])

    # Density matrix Pauli operator basis
    II = np.kron(I, I)
    IX = np.kron(I, sigma_x)
    IY = np.kron(I, sigma_y)
    IZ = np.kron(I, sigma_z)
    XI = np.kron(sigma_x, I)
    XX = np.kron(sigma_x, sigma_x)
    XY = np.kron(sigma_x, sigma_y)
    XZ = np.kron(sigma_x, sigma_z)
    YI = np.kron(sigma_y, I)
    YX = np.kron(sigma_y, sigma_x)
    YY = np.kron(sigma_y, sigma_y)
    YZ = np.kron(sigma_y, sigma_z)
    ZI = np.kron(sigma_z, I)
    ZX = np.kron(sigma_z, sigma_x)
    ZY = np.kron(sigma_z, sigma_y)
    ZZ = np.kron(sigma_z, sigma_z)

    rho = 0.25 * (
        II
        + stokes[0] * IX
        + stokes[1] * IY
        + stokes[2] * IZ
        + stokes[3] * XI
        + stokes[4] * XX
        + stokes[5] * XY
        + stokes[6] * XZ
        + stokes[7] * YI
        + stokes[8] * YX
        + stokes[9] * YY
        + stokes[10] * YZ
        + stokes[11] * ZI
        + stokes[12] * ZX
        + stokes[13] * ZY
        + stokes[14] * ZZ
    )

    print(f"The density matrix is:\n{np.round(rho, decimals=3)}")

    sqrt_ideal_state = sqrtm(ideal_state)

    state_fidelity = (np.abs(sqrtm(sqrt_ideal_state @ rho @ sqrt_ideal_state).trace())) ** 2

    print(f"The state fidelity is: {np.round(state_fidelity, decimals=4)}")
