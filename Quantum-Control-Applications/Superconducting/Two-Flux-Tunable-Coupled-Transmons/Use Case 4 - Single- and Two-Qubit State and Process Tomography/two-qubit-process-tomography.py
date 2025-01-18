#!/usr/bin/env python

"""
        TWO QUBIT PROCESS TOMOGRAPHY
The sequence consists of preparing the qubits each into one of their six cardinal Bloch sphere
states using calibrated gates from the config, applying the operation/process - typically
ideally a unitary matrix - under investigation, and then measuring the state of the qubits,
by way of the readout resonator, in their X, Y and Z bases (specifically, measuring the
projector of one of the same six cardinal Bloch sphere states). This is repeated for
the full set of input states and measurement projectors, which is tomographically
complete. The output, 'probs', containing the 'Bloch sphere' prepared states and
measurement projectors, is mapped to the Pauli basis (giving a 'measurement vector'),
and the equation "measurement_vector = B \times chi_vector" (where B is a constant)
inverted to give the chi process matrix for the process under
investigation.

Computing PMatrix from scratch in the two-qubit process case takes a while,
so included with this project is a serialised file of Pmatrix

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubits under study (resonator_spectroscopy)
    - Having calibrated qubits' pi and pi/2 pulses by running qubit spectroscopy, rabi_chevron, power_rabi and updating the config
    - Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR, and having
      saved the derived readout threshold values in the config
    - Set the desired flux biases in the case of flux-tunable qubits

This script implements the logic outlined in the notebook
https://github.com/bornman-nick/quantum-state-and-process-tomography. See that
notebook for a detailed explanation of standard two qubit process tomography

Note that this routine carries out standard process tomography for two qubits.
Given that the measurements recorded are simply whether the final state of the
qubits are in their ground state or not for each state preparation and measurement
setting (instead of also deciding whether the qubits are perhaps in the excited
states and incrementing the counter of a complementary measurement setting),
this routine scales as 6**4, rather than (6**2)*(3**2), as is the case in
most formulations of standard process tomography. However, the routine below
is fast enough that this doesn't matter too much.
"""

from qm.qua import *
from qm import QuantumMachinesManager
from qm import SimulationConfig
from configuration import *
from qualang_tools.results import progress_counter, fetching_tool
import numpy as np
from scipy.linalg import solve

import os
import pickle

from helper_functions import (
    P_Pauli2,
    plot_process_tomography2,
    map_from_bloch_state_to_pauli_basis2,
    func_F2,
)


###################
# The QUA program #
###################

# load PMatrix if pickle file is present
# note: this takes a while to compute if you do not load the serialised
# PMatrix file
file_location = "<path to PMatrix2 file>"  # ./PMatrix2.pkl

if os.path.isfile(file_location):
    with open(file_location, "rb") as file:
        PMatrix = pickle.load(file)
else:
    # computing this takes a while
    PMatrix = np.array(
        [
            [
                P_Pauli2(
                    np.floor(v / 16).astype(int),
                    v % 16,
                    np.floor(w / 16).astype(int),
                    w % 16,
                )
                for w in range(16**2)
            ]
            for v in range(16**2)
        ]
    )

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


n_avg = 5_000


# subroutine to prepare desired gate/process
def analysed_process(qubit1, qubit2):
    # write whatever QUA code you need, here, in order to perform the
    # desired process which we want to subject to tomography.
    # For example, to analyse the X gate on qubit1 and the -Y90
    # gate on qubit2:
    play("x180", f"q{qubit1}_xy")
    play("-y90", f"q{qubit2}_xy")


ideal_gate = np.kron(func_F2(1), func_F2(3))


# subroutine to switch between preparing the tomographically-complete
# set of input states
def prepare_states(i, j, qubit1, qubit2):

    # Prepare Bloch state i on qubit1
    with switch_(i):
        with case_(0):
            wait(pi_len // 4, f"q{qubit1}_xy")
        with case_(1):
            play("x180", f"q{qubit1}_xy")
        with case_(2):
            play("y90", f"q{qubit1}_xy")
        with case_(3):
            play("-y90", f"q{qubit1}_xy")
        with case_(4):
            play("-x90", f"q{qubit1}_xy")
        with case_(5):
            play("x90", f"q{qubit1}_xy")

    # Prepare Bloch state j on qubit2
    with switch_(j):
        with case_(0):
            wait(pi_len // 4, f"q{qubit2}_xy")
        with case_(1):
            play("x180", f"q{qubit2}_xy")
        with case_(2):
            play("y90", f"q{qubit2}_xy")
        with case_(3):
            play("-y90", f"q{qubit2}_xy")
        with case_(4):
            play("-x90", f"q{qubit2}_xy")
        with case_(5):
            play("x90", f"q{qubit2}_xy")


# subroutine to switch between the tomographically-complete
# set of bases in which to measure
def measurement_basis_change(k, l, qubit1, qubit2):

    # Change basis with operation k on qubit1
    with switch_(k):
        with case_(0):
            wait(pi_len // 4, f"q{qubit1}_xy")
        with case_(1):
            play("x180", f"q{qubit1}_xy")
        with case_(2):
            play("y90", f"q{qubit1}_xy")
        with case_(3):
            play("-y90", f"q{qubit1}_xy")
        with case_(4):
            play("-x90", f"q{qubit1}_xy")
        with case_(5):
            play("x90", f"q{qubit1}_xy")

    # Change basis with operation l on qubit2
    with switch_(l):
        with case_(0):
            wait(pi_len // 4, f"q{qubit2}_xy")
        with case_(1):
            play("x180", f"q{qubit2}_xy")
        with case_(2):
            play("y90", f"q{qubit2}_xy")
        with case_(3):
            play("-y90", f"q{qubit2}_xy")
        with case_(4):
            play("-x90", f"q{qubit2}_xy")
        with case_(5):
            play("x90", f"q{qubit2}_xy")


with program() as two_qubit_process_tomography:

    n = declare(int)
    n_st = declare_stream()

    I1 = declare(fixed)
    Q1 = declare(fixed)
    I2 = declare(fixed)
    Q2 = declare(fixed)

    state = declare(bool)  # QUA variable for the measured qubits state
    state_st = declare_stream()  # Stream for the qubits state

    c1 = declare(int)  # QUA variable for switching between state preparation/creations on qubit 1
    c2 = declare(int)  # QUA variable for switching between state preparation/creations on qubit 2
    m1 = declare(int)  # QUA variable for switching between Bloch basis projections/measurements on qubit 1
    m2 = declare(int)  # QUA variable for switching between Bloch basis projections/measurements on qubit 2

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(c1, 0, c1 <= 5, c1 + 1):  # QUA for_ loop for switching between state preparations on qubit 1
            with for_(c2, 0, c2 <= 5, c2 + 1):  # QUA for_ loop for switching between state preparations on qubit 2
                with for_(
                    m1, 0, m1 <= 5, m1 + 1
                ):  # QUA for_ loop for switching between Bloch basis projections/measurements on qubit 1
                    with for_(
                        m2, 0, m2 <= 5, m2 + 1
                    ):  # QUA for_ loop for switching between Bloch basis projections/measurements on qubit 2

                        # NOTE: the following frame and phase resets may not be necessary, depending on whether you're
                        # investigating more trivial single qubit gates or more complex entangling gates where
                        # relative phases are more important

                        reset_frame(f"q{qubit1}_xy")
                        reset_frame(f"q{qubit2}_xy")

                        reset_phase(f"q{qubit1}_xy")
                        reset_phase(f"q{qubit2}_xy")

                        # prepare qubit1 and qubit 2 in one of six Bloch sphere states
                        prepare_states(c1, c2, qubit1, qubit2)

                        align()

                        # apply the process to be analysed
                        analysed_process(qubit1, qubit2)

                        align()

                        # projective measurement basis change
                        measurement_basis_change(m1, m2, qubit1, qubit2)

                        align()

                        measure(
                            "readout",
                            f"rr{qubit1}",
                            None,
                            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I1),
                            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q1),
                        )

                        measure(
                            "readout",
                            f"rr{qubit2}",
                            None,
                            dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I2),
                            dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q2),
                        )

                        align()

                        # track the number of clicks when both qubit 1 and qubit 2 are
                        # in their ground states
                        with if_((I1 < threshold1) & (I2 < threshold2)):
                            assign(state, True)
                        with else_():
                            assign(state, False)

                        save(state, state_st)

                        wait(thermalization_time * u.ns, f"rr{qubit1}", f"rr{qubit2}")

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(6).buffer(6).buffer(6).buffer(6).average().save("probs")


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
    job = qmm.simulate(config, two_qubit_process_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(two_qubit_process_tomography)
    # Get results from QUA program
    results = fetching_tool(job, data_list=["iteration", "probs"], mode="live")

    while results.is_processing():

        # Fetch results
        iteration, probs = results.fetch_all()

        # Progress bar
        progress_counter(iteration, n_avg, start_time=results.get_start_time())

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    # post-processing
    pauli_basis_measurements = lambda q, n: map_from_bloch_state_to_pauli_basis2(q, n, probs)

    measurement_vector = np.array(
        [pauli_basis_measurements(np.floor(v / 16).astype(int), v % 16) for v in range(16**2)]
    )

    chi_vector = solve(PMatrix, measurement_vector)
    chi_matrix = chi_vector.reshape(16, 16)

    plot_process_tomography2(chi_vector)
