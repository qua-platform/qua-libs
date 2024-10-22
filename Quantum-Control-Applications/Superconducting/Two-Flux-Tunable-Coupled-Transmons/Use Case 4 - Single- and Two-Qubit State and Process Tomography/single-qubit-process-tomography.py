#!/usr/bin/env python

"""
        SINGLE QUBIT PROCESS TOMOGRAPHY
The sequence consists of preparing the qubit into one of the six cardinal Bloch sphere
states using calibrated gates from the config, applying the operation/process - typically
ideally a unitary matrix - under investigation, and then measuring the state of the qubit,
by way of the readout resonator, in the X, Y and Z bases (specifically, measuring the
projector of one of the same six cardinal Bloch sphere states). This is repeated for
the full set of input states and measurement projectors, which is tomographically
complete. The output, 'probs', containing the 'Bloch sphere' prepared states and
measurement projectors, is mapped to the Pauli basis (giving a 'measurement vector'),
and the equation "measurement_vector = B \times chi_vector" (where B is a constant)
inverted to give the chi process matrix for the process under
investigation.



Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy)
    - Having calibrated qubit pi and pi/2 pulses by running qubit spectroscopy, rabi_chevron, power_rabi and updating the config
    - Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR, and having
      saved the derived readout threshold value in the config
    - Set the desired flux bias in the case of flux-tunable qubits

This script implements the logic outlined in the notebook
https://github.com/bornman-nick/quantum-state-and-process-tomography. See
that notebook for a detailed explanation of standard single qubit process
tomography.

Note that this routine carries out standard process tomography on a single qubit.
Given that the measurements recorded are simply whether the final state of the
qubit is in the ground state or not for each state preparation and measurement
setting (instead of also deciding whether the qubit is perhaps in the excited
state and incrementing the counter of a complementary measurement setting),
this routine scales as 6**2, rather than 6*3, as is the case in
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

from helper_functions import (
    P_Pauli1,
    plot_process_tomography1,
    map_from_bloch_state_to_pauli_basis1,
    func_E1,
)


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


# subroutine to prepare desired qubit gate/process
def analysed_process(qubit):
    # write whatever QUA code you need, here, in order to perform the
    # desired process which we want to subject to tomography.
    # For example, to analyse the Y gate:
    play("y180", f"q{qubit}_xy")


ideal_gate = func_E1(2)


# subroutine to switch between preparing the tomographically-complete
# set of input states
def prepare_state(i):
    with switch_(i):
        with case_(0):
            wait(pi_len // 4, f"q{qubit}_xy")
        with case_(1):
            play("x180", f"q{qubit}_xy")
        with case_(2):
            play("y90", f"q{qubit}_xy")
        with case_(3):
            play("-y90", f"q{qubit}_xy")
        with case_(4):
            play("-x90", f"q{qubit}_xy")
        with case_(5):
            play("x90", f"q{qubit}_xy")


# subroutine to switch between the tomographically-complete
# set of bases in which to measure
def measurement_basis_change(j):
    with switch_(j):
        with case_(0):
            wait(pi_len // 4, f"q{qubit}_xy")
        with case_(1):
            play("x180", f"q{qubit}_xy")
        with case_(2):
            play("y90", f"q{qubit}_xy")
        with case_(3):
            play("-y90", f"q{qubit}_xy")
        with case_(4):
            play("-x90", f"q{qubit}_xy")
        with case_(5):
            play("x90", f"q{qubit}_xy")


with program() as single_qubit_process_tomography:

    n = declare(int)
    n_st = declare_stream()

    I = declare(fixed)
    Q = declare(fixed)

    state = declare(bool)  # QUA variable for the measured qubit state
    state_st = declare_stream()  # Stream for the qubit state

    c = declare(int)  # QUA variable for switching between state preparation/creations
    m = declare(int)  # QUA variable for switching between basis projections/measurements

    with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
        with for_(c, 0, c <= 5, c + 1):  # QUA for_ loop for switching between state preparations
            with for_(m, 0, m <= 5, m + 1):  # QUA for_ loop for switching between basis projections/measurements

                # prepare qubit in one of six cardinal Bloch sphere states
                prepare_state(c)

                align()

                # the process to be analysed
                analysed_process(qubit)

                align()

                # projective measurement basis change
                measurement_basis_change(m)

                align()

                measure(
                    "readout",
                    f"rr{qubit}",
                    None,
                    dual_demod.full("rotated_cos", "out1", "rotated_sin", "out2", I),
                    dual_demod.full("rotated_minus_sin", "out1", "rotated_cos", "out2", Q),
                )

                align()

                # track number of times the qubit is found to be
                # in the ground state
                assign(state, I < threshold)

                save(state, state_st)

                wait(thermalization_time * u.ns)

        save(n, n_st)

    with stream_processing():
        n_st.save("iteration")
        state_st.boolean_to_int().buffer(6).buffer(6).average().save("probs")


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
    job = qmm.simulate(config, single_qubit_process_tomography, simulation_config)
    job.get_simulated_samples().con1.plot()

else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(single_qubit_process_tomography)
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
    PMatrix = np.array(
        [
            [
                P_Pauli1(
                    np.floor(v / 4).astype(int),
                    v % 4,
                    np.floor(w / 4).astype(int),
                    w % 4,
                )
                for w in range(16)
            ]
            for v in range(16)
        ]
    )

    pauli_basis_measurements = lambda l, k: map_from_bloch_state_to_pauli_basis1(l, k, probs)

    measurement_vector = np.array([pauli_basis_measurements(np.floor(v / 4).astype(int), v % 4) for v in range(16)])

    chi_vector = solve(PMatrix, measurement_vector)
    chi_matrix = chi_vector.reshape(4, 4)

    plot_process_tomography1(chi_vector)
