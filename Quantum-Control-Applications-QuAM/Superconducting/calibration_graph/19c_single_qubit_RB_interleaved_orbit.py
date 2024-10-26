# %%
"""
ORBIT PROTOCOL CALIBRATION FOR CONTROL PARAMETERS (AMPLITUDE, FREQUENCY, DRAG)
This program implements the ORBIT protocol to calibrate the control parameters for single-qubit gates,
specifically amplitude, frequency, and drag. The protocol involves playing random fixed-length sequences
of Clifford gates on a qubit and measuring the state of the resonator afterwards.

Each random sequence is generated on the FPGA for the depth (specified as an input) and played
for a range of control parameter values requested by the user. The sequence is truncated to the desired depth,
and each truncated sequence ends with a recovery gate, determined at each step using a preloaded lookup table
(Cayley table), to bring the qubit back to its ground state. A specific Clifford gate, chosen by the user,
is interleaved between each random gate in the sequence to evaluate the fidelity of a particular gate.

The program is re-compiled for every ORBIT value to minimize latency from real-time parameterization of the
ORBIT variable. The amplitude and drag coefficient are applied before generating the config when swept, but
the frequency is set after opening the qm to preserve the optimal mixer calibration at the original IF.

If the readout has been properly calibrated and is sufficiently accurate, state discrimination can be applied
to return only the state of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned. Each sequence is played
n_avg times for averaging, with additional averaging performed by executing different random sequences.

The data is then post-processed to extract the optimal values of the control parameters in the given range.

After finding the optimal parameters for a given depth 'm', sensitivity can be maintained by repeating the
experiment with double the depth when the error is halved.

Prerequisites:
    - Resonance frequency of the resonator coupled to the qubit under study has been determined (resonator_spectroscopy).
    - Qubit pi pulse (x180) has been calibrated through qubit spectroscopy, rabi_chevron, power_rabi, and state updates.
    - Qubit frequency has been precisely calibrated (ramsey).
    - (optional) Readout has been calibrated (readout frequency, amplitude, duration optimization, IQ blobs) for improved SNR.
    - Desired flux bias has been set.
"""
import copy
import logging

from tqdm.auto import tqdm
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List, Union
from qm import logger


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    interleaved_gate_index: int = 2
    num_random_sequences: int = 50
    num_averages: int = 1000
    circuit_depth: int = 5
    seed: int = 345324
    frequency_span_in_mhz: float = 20
    frequency_step_in_mhz: float = 4.99  #0.25
    min_amp_factor: float = 0.8
    max_amp_factor: float = 1.2
    amp_factor_step: float = 0.099  #0.005
    min_drag_coefficient_factor: float = 0.8
    max_drag_coefficient_factor: float = 1.2
    drag_coefficient_factor_step: float = 0.099  #0.02

    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "active"
    simulate: bool = False
    timeout: int = 100

node = QualibrationNode(
    name="11c_Randomized_Benchmarking_Interleaved_ORBIT",
    parameters=Parameters()
)




from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import fetching_tool
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.units import unit
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import active_reset

import matplotlib.pyplot as plt
import numpy as np



# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
# Open Communication with the QOP
qmm = machine.connect()

if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)

##############################
# Program-specific variables #
##############################
# ORBIT Parameters
dfs = np.arange(
    -node.parameters.frequency_span_in_mhz * u.MHz // 2,
    +node.parameters.frequency_span_in_mhz * u.MHz // 2,
    node.parameters.frequency_step_in_mhz * u.MHz,
dtype=np.int32)

# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step
)

# Drag coefficient sweep (as a pre-factor of the qubit drag coefficient) - must be within [-2; 2)
drag_coefficient_factors = np.arange(
    node.parameters.min_drag_coefficient_factor,
    node.parameters.max_drag_coefficient_factor,
    node.parameters.drag_coefficient_factor_step
)

orbit_variables = ["frequency", "amplitude", "drag coefficient factor"]
orbit_variable_sweeps = dict(zip(orbit_variables, [dfs, amps, drag_coefficient_factors]))

# RB Parameters
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
# %% {QUA_program}
n_avg = node.parameters.num_averages  # Number of averaging loops for each random sequence
circuit_depth = node.parameters.circuit_depth  # Maximum circuit depth
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
seed = node.parameters.seed  # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]
# index of the gate to interleave from the play_sequence() function defined below
# Correspondence table:
#  0: identity |  1: x180 |  2: y180
# 12: x90      | 13: -x90 | 14: y90 | 15: -y90 |
interleaved_gate_index = node.parameters.interleaved_gate_index


###################################
# Helper functions and QUA macros #
###################################
def get_interleaved_gate(gate_index):
    if gate_index == 0:
        return "I"
    elif gate_index == 1:
        return "x180"
    elif gate_index == 2:
        return "y180"
    elif gate_index == 12:
        return "x90"
    elif gate_index == 13:
        return "-x90"
    elif gate_index == 14:
        return "y90"
    elif gate_index == 15:
        return "-y90"
    else:
        raise ValueError(f"Interleaved gate index {gate_index} doesn't correspond to a single operation")


def power_law(power, a, b, p):
    return a * (p**power) + b


def set_orbit_value(qubit: Transmon, qubit_with_orbit_value: Transmon, orbit_variable: str, value: Union[float, int]) -> Transmon:
    interleaved_gate_operation = get_interleaved_gate(interleaved_gate_index)

    if orbit_variable == "frequency":
        # set in program to retain mixer calibration at original IF
        pass
    elif orbit_variable == "amplitude":
        qubit_with_orbit_value.xy.operations[interleaved_gate_operation].amplitude = None
        qubit_with_orbit_value.xy.operations[interleaved_gate_operation].amplitude = \
            copy.deepcopy(qubit.xy.operations[interleaved_gate_operation].amplitude * value)
    elif orbit_variable == "drag_coefficient_factor":
        qubit_with_orbit_value.xy.operations[interleaved_gate_operation].alpha = None
        qubit_with_orbit_value.xy.operations[interleaved_gate_operation].alpha = \
            copy.deepcopy(qubit.xy.operations[interleaved_gate_operation].alpha * value)
    else:
        raise ValueError(f"Orbit variable {orbit_variable} not recognized")

    return qubit_with_orbit_value

def generate_sequence(interleaved_gate_index):
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=2 * circuit_depth + 1)
    inv_gate = declare(int, size=2 * circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < 2 * circuit_depth, i + 2):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])
        # interleaved gate
        assign(step, interleaved_gate_index)
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i + 1], step)
        assign(inv_gate[i + 1], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qubit: Transmon, qubit_with_orbit_values: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):
                if interleaved_gate_index == 1:
                    qubit_with_orbit_values.xy.play("x180")
                else:
                    qubit.xy.play("x180")
            with case_(2):
                if interleaved_gate_index == 2:
                    qubit_with_orbit_values.xy.play("y180")
                else:
                    qubit.xy.play("y180")
            with case_(3):
                qubit.xy.play("y180")
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
            with case_(5):
                qubit.xy.play("x90")
                qubit.xy.play("-y90")
            with case_(6):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
            with case_(7):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
            with case_(8):
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(9):
                qubit.xy.play("y90")
                qubit.xy.play("-x90")
            with case_(10):
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(11):
                qubit.xy.play("-y90")
                qubit.xy.play("-x90")
            with case_(12):
                if interleaved_gate_index == 12:
                    qubit_with_orbit_values.xy.play("x90")
                else:
                    qubit.xy.play("x90")
            with case_(13):
                if interleaved_gate_index == 13:
                    qubit_with_orbit_values.xy.play("-x90")
                else:
                    qubit.xy.play("-x90")
            with case_(14):
                if interleaved_gate_index == 14:
                    qubit_with_orbit_values.xy.play("y90")
                else:
                    qubit.xy.play("y90")
            with case_(15):
                if interleaved_gate_index == 15:
                    qubit_with_orbit_values.xy.play("-y90")
                else:
                    qubit.xy.play("-y90")
            with case_(16):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(17):
                qubit.xy.play("-x90")
                qubit.xy.play("-y90")
                qubit.xy.play("x90")
            with case_(18):
                qubit.xy.play("x180")
                qubit.xy.play("y90")
            with case_(19):
                qubit.xy.play("x180")
                qubit.xy.play("-y90")
            with case_(20):
                qubit.xy.play("y180")
                qubit.xy.play("x90")
            with case_(21):
                qubit.xy.play("y180")
                qubit.xy.play("-x90")
            with case_(22):
                qubit.xy.play("x90")
                qubit.xy.play("y90")
                qubit.xy.play("x90")
            with case_(23):
                qubit.xy.play("-x90")
                qubit.xy.play("y90")
                qubit.xy.play("-x90")


def get_rb_interleaved_program(qubit: Transmon, qubit_with_orbit_values: Transmon):
    with program() as rb:
        depth = declare(int)  # QUA variable for the varying depth
        # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
        saved_gate = declare(int)
        m = declare(int)  # QUA variable for the loop over random sequences
        n = declare(int)  # QUA variable for the averaging loop
        I = declare(fixed)  # QUA variable for the 'I' quadrature
        Q = declare(fixed)  # QUA variable for the 'Q' quadrature
        state = declare(bool)  # QUA variable for state discrimination
        # The relevant streams
        m_st = declare_stream()
        I_st = declare_stream()
        Q_st = declare_stream()
        if state_discrimination:
            state_st = declare_stream()

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
            # Generates the RB sequence with a gate interleaved after each Clifford
            sequence_list, inv_gate_list = generate_sequence(interleaved_gate_index=interleaved_gate_index)
            # Depth_target is used to always play the gates by pairs [(random_gate-interleaved_gate)^depth/2-inv_gate]
            assign(depth, circuit_depth)
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth
            with for_(n, 0, n < n_avg, n + 1):
                if reset_type == "active":
                    active_reset(qubit)
                else:
                    wait(qubit.thermalization_time * u.ns)
                # Align the two elements to play the sequence after qubit initialization
                qubit.resonator.align(qubit.xy.name)
                # The strict_timing ensures that the sequence will be played without gaps
                if node.parameters.use_strict_timing:
                    with strict_timing_():
                        # Play the random sequence of desired depth
                        play_sequence(sequence_list, depth, qubit, qubit_with_orbit_values)
                else:
                    # Play the random sequence of desired depth
                    play_sequence(sequence_list, depth, qubit, qubit_with_orbit_values)
                # Align the elements to measure after playing the circuit.
                align()
                # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                qubit.resonator.measure("readout", qua_vars=(I, Q))
                # Make sure you updated the ge_threshold
                if state_discrimination:
                    assign(state, I > qubit.resonator.operations["readout"].threshold)
                    save(state, state_st)
                else:
                    save(I, I_st)
                    save(Q, Q_st)
            # Reset the last gate of the sequence back to the original Clifford gate
            # (that was replaced by the recovery gate at the beginning)
            assign(sequence_list[depth], saved_gate)
            # Save the counter for the progress bar
            save(m, m_st)

        with stream_processing():
            m_st.save("iteration")
            if state_discrimination:
                # saves a 2D array of depth and random pulse sequences in order to get error bars along the random sequences
                state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences).save("state")
                # returns a 1D array of averaged random pulse sequences vs depth of circuit for live plotting
                state_st.boolean_to_int().buffer(n_avg).map(FUNCTIONS.average()).average().save("state_avg")
            else:
                I_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences).save("I")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).buffer(num_of_sequences).save("Q")
                I_st.buffer(n_avg).map(FUNCTIONS.average()).average().save("I_avg")
                Q_st.buffer(n_avg).map(FUNCTIONS.average()).average().save("Q_avg")

    return rb


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    config = machine.generate_config()
    job = qmm.simulate(config, get_rb_interleaved_program(qubits[0]), simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results["figure"] = plt.gcf()
else:
    node.results = {}
    for qubit in qubits:
        # clone qubit for the interleaved gate only (todo: can't deepcopy due to inf. recursion)
        qubit_with_orbit_value_name = qubit.name + "_with_orbit_value"
        cloned_qubit_dict = qubit.to_dict(follow_references=False)
        cloned_qubit_dict["id"] = qubit_with_orbit_value_name
        machine_dict = machine.to_dict(follow_references=False)
        machine_dict["qubits"][qubit_with_orbit_value_name] = cloned_qubit_dict
        machine_with_orbit_qubit = QuAM.load(machine_dict)
        cloned_qubit = machine_with_orbit_qubit.qubits[qubit_with_orbit_value_name]
        cloned_qubit.parent = None
        machine.qubits[qubit_with_orbit_value_name] = cloned_qubit
        qubit_with_orbit_value = machine.qubits[qubit_with_orbit_value_name]

        for i, orbit_variable in enumerate(tqdm(orbit_variables, unit='ORBIT variable')):
            sweep = orbit_variable_sweeps[orbit_variable]
            # prepare empty array for state average
            node.results[orbit_variable] = np.zeros(len(sweep))
            # sweep each orbit variable one-at-a-time with fixed sequence depth
            for j, value in enumerate(tqdm(sweep, unit=f'{orbit_variable} value')):

                if orbit_variable in ["amplitude", "drag_coefficient_factor"]:
                    qubit_with_orbit_value = set_orbit_value(qubit, qubit_with_orbit_value, orbit_variable, value)

                config = machine.generate_config()
                qm = qmm.open_qm(config)

                if orbit_variable == "frequency":
                    # set the ORBIT frequency post open_qm to retain mixer calibration at original IF
                    qm.set_intermediate_frequency(
                        element=qubit.name + "_with_orbit_value.xy",
                        freq=qubit.xy.intermediate_frequency + float(value)
                    )

                # silence job INFO output since we will execute many programs
                logger.setLevel(logging.WARNING)
                # execute the program
                job = qm.execute(get_rb_interleaved_program(qubit, qubit_with_orbit_value))

                if state_discrimination:
                    results = fetching_tool(job, data_list=["state_avg", "iteration"], mode="wait_for_all")
                else:
                    results = fetching_tool(job, data_list=["I_avg", "Q_avg", "iteration"], mode="wait_for_all")

                if state_discrimination:
                    state_avg, iteration = results.fetch_all()
                    value_avg = state_avg
                else:
                    I, Q, iteration = results.fetch_all()
                    value_avg = I

                if state_discrimination:
                    results = fetching_tool(job, data_list=["state"])
                    state = results.fetch_all()[0]
                    value_avg = np.mean(state, axis=0)
                    error_avg = np.std(state, axis=0)
                else:
                    results = fetching_tool(job, data_list=["I", "Q"])
                    I, Q = results.fetch_all()
                    value_avg = np.mean(I, axis=0)
                    error_avg = np.std(I, axis=0)

                node.results[orbit_variable][j] = value_avg

            # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
            qm.close()

# %%
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
fig.suptitle(f"ORBIT at depth-{circuit_depth}")

for i, (orbit_variable, sweep) in enumerate(orbit_variable_sweeps.items()):
    ax[i].plot(sweep, node.results[orbit_variable])
    ax[i].set_xlabel(orbit_variable)
    ax[i].set_ylabel(r"$|1\rangle$-projection")

plt.tight_layout()
plt.show()
plt.savefig("test21233321.png")

node.results["figure"] = fig

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
