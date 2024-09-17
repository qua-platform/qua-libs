# %%
"""
SINGLE QUBIT RANDOMIZED BENCHMARKING (for gates >= 40ns)
The program consists in playing random sequences of Clifford gates and measuring the state of the resonator afterwards.
Each random sequence is derived on the FPGA for the maximum depth (specified as an input) and played for each depth
asked by the user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate,
found at each step thanks to a preloaded lookup table (Cayley table), that will bring the qubit back to its ground state.

If the readout has been calibrated and is good enough, then state discrimination can be applied to only return the state
of the qubit. Otherwise, the 'I' and 'Q' quadratures are returned.
Each sequence is played n_avg times for averaging. A second averaging is performed by playing different random sequences.

The data is then post-processed to extract the single-qubit gate fidelity and error per gate
.
Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the config.
    - Having the qubit frequency perfectly calibrated (ramsey).
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.
    - Set the desired flux bias.
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal

from quam_libs.trackable_object import tracked_updates

class Parameters(NodeParameters):
    qubits: Optional[str] = None
    use_state_discrimination: bool = True
    use_strict_timing: bool = False
    num_random_sequences: int = 100  # Number of random sequences
    num_averages: int = 20
    max_circuit_depth: int = 1000  # Maximum circuit depth
    delta_clifford: int = 20
    seed: int = 345324
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal['thermal', 'active'] = "thermal"
    simulate: bool = False
    multiplexed: bool = False

node = QualibrationNode(
    name="11a_Randomized_Benchmarking",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.bakery.randomized_benchmark_c1 import c1_table
from qualang_tools.units import unit
from quam_libs.components import QuAM, Transmon
from quam_libs.macros import active_reset, qua_declaration

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_decay_exp, decay_exp

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations

config = machine.generate_config()
octave_config = machine.get_octave_config()
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
num_of_sequences = node.parameters.num_random_sequences  # Number of random sequences
n_avg = node.parameters.num_averages  # Number of averaging loops for each random sequence
max_circuit_depth = node.parameters.max_circuit_depth  # Maximum circuit depth
if node.parameters.delta_clifford < 1:
    raise NotImplementedError("Delta clifford < 2 is not supported.")
delta_clifford = node.parameters.delta_clifford  #  Play each sequence with a depth step equals to 'delta_clifford - Must be > 1
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_thermal_or_active
assert (max_circuit_depth / delta_clifford).is_integer(), "max_circuit_depth / delta_clifford must be an integer."
num_depths = max_circuit_depth // delta_clifford + 1
seed = node.parameters.seed # Pseudo-random number generator seed
# Flag to enable state discrimination if the readout has been calibrated (rotated blobs and threshold)
state_discrimination = node.parameters.use_state_discrimination
strict_timing = node.parameters.use_strict_timing
# List of recovery gates from the lookup table
inv_gates = [int(np.where(c1_table[i, :] == 0)[0][0]) for i in range(24)]


###################################
# Helper functions and QUA macros #
###################################
def power_law(power, a, b, p):
    return a * (p**power) + b


def generate_sequence():
    cayley = declare(int, value=c1_table.flatten().tolist())
    inv_list = declare(int, value=inv_gates)
    current_state = declare(int)
    step = declare(int)
    sequence = declare(int, size=max_circuit_depth + 1)
    inv_gate = declare(int, size=max_circuit_depth + 1)
    i = declare(int)
    rand = Random(seed=seed)

    assign(current_state, 0)
    with for_(i, 0, i < max_circuit_depth, i + 1):
        assign(step, rand.rand_int(24))
        assign(current_state, cayley[current_state * 24 + step])
        assign(sequence[i], step)
        assign(inv_gate[i], inv_list[current_state])

    return sequence, inv_gate


def play_sequence(sequence_list, depth, qubit: Transmon):
    i = declare(int)
    with for_(i, 0, i <= depth, i + 1):
        with switch_(sequence_list[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(qubit.xy.operations["x180"].length // 4)
            with case_(1):
                qubit.xy.play("x180")
            with case_(2):
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
                qubit.xy.play("x90")
            with case_(13):
                qubit.xy.play("-x90")
            with case_(14):
                qubit.xy.play("y90")
            with case_(15):
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

average_gate_per_clifford = (
    1 * 3 + 9 * 2 + 1 * 4 + 2 * 3 + 4 * 2 + 2 * 3)/24  # = 45/24 = 1.875



###################
# The QUA program #
###################
with program() as randomized_benchmarking:
    depth = declare(int)  # QUA variable for the varying depth
    depth_target = declare(int)  # QUA variable for the current depth (changes in steps of delta_clifford)
    # QUA variable to store the last Clifford gate of the current sequence which is replaced by the recovery gate
    saved_gate = declare(int)
    m = declare(int)  # QUA variable for the loop over random sequences
    # n = declare(int)  # QUA variable for the averaging loop
    # I = declare(fixed)  # QUA variable for the 'I' quadrature
    # Q = declare(fixed)  # QUA variable for the 'Q' quadrature
    # state = declare(bool)  # QUA variable for state discrimination

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    
    # The relevant streams
    m_st = declare_stream()
    # state_st = declare_stream()
    state_st = [declare_stream() for _ in range(num_qubits)]  

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            raise ValueError(f"Unrecognized flux point {flux_point}")


    with for_(m, 0, m < num_of_sequences, m + 1):  # QUA for_ loop over the random sequences
        sequence_list, inv_gate_list = generate_sequence()  # Generate the random sequence of length max_circuit_depth

        assign(depth_target, 0)  # Initialize the current depth to 0

        with for_(depth, 1, depth <= max_circuit_depth, depth + 1):
            # Replacing the last gate in the sequence with the sequence's inverse gate
            # The original gate is saved in 'saved_gate' and is being restored at the end
            assign(saved_gate, sequence_list[depth])
            assign(sequence_list[depth], inv_gate_list[depth - 1])
            # Only played the depth corresponding to target_depth     
            with if_((depth == 1) | (depth == depth_target)):
                with for_(n, 0, n < n_avg, n + 1):
                    for i, qubit in enumerate(qubits):
                        # Bring the active qubits to the minimum frequency point
                        if flux_point == "independent":
                            machine.apply_all_flux_to_min()
                            qubit.z.to_independent_idle()
                            qubit.z.wait(1000)
                        if node.parameters.multiplexed:
                            qubit.align()
                        else:
                            align()
                        if reset_type == "active":
                            active_reset(machine, qubit.name)
                        else:
                            qubit.resonator.wait(machine.thermalization_time * u.ns)
                        # Align the two elements to play the sequence after qubit initialization
                        align(qubit.xy.name, qubit.resonator.name)
                        # The strict_timing ensures that the sequence will be played without gaps
                        if strict_timing:
                            with strict_timing_():
                                # Play the random sequence of desired depth
                                play_sequence(sequence_list, depth, qubit)
                        else:
                            play_sequence(sequence_list, depth, qubit)
                        # Align the two elements to measure after playing the circuit.
                        align(qubit.xy.name, qubit.resonator.name)
                        # Make sure you updated the ge_threshold and angle if you want to use state discrimination
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # Make sure you updated the ge_threshold
                        assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                        save(state[i], state_st[i])

                # Go to the next depth
                assign(depth_target, depth_target + delta_clifford)
            # Reset the last gate of the sequence back to the original Clifford gate
            # (that was replaced by the recovery gate at the beginning)
            assign(sequence_list[depth], saved_gate)
        # Save the counter for the progress bar
        save(m, m_st)



    with stream_processing():
        m_st.save("iteration")
        for i in range(num_qubits):
            state_st[i].buffer(n_avg).map(FUNCTIONS.average()).buffer(num_depths).buffer(num_of_sequences).save(f"state{i + 1}")

simulate = node.parameters.simulate

if simulate:
    simulation_config = SimulationConfig(duration=100_000)  # in clock cycles
    job = qmm.simulate(config, randomized_benchmarking, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results["figure"] = plt.gcf()
    node.machine = machine
    node.save()
    quit()
else:
    # Prepare data for saving
    node.results = {}
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(randomized_benchmarking)
    
    
    for i in range(num_qubits):
        print(f"Fetching results for qubit {qubits[i].name}")
        data_list = ["iteration"] 
        results = fetching_tool(job, data_list, mode="live")
    # Live plotting

        while results.is_processing():
        # Fetch results
            fetched_data = results.fetch_all()
            m = fetched_data[0]
            progress_counter(m, num_of_sequences, start_time=results.start_time)
    qm.close()   

# %%
depths = np.arange(0, max_circuit_depth + 0.1, delta_clifford)
depths[0] = 1
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"depths": depths, "sequence": np.arange(num_of_sequences)})

node.results = {}
node.results['ds'] = ds

# %%
import xarray as xr
da_state = 1 - ds['state'].mean(dim='sequence')
da_state: xr.DataArray
da_state.attrs = {'long_name': 'p(0)'}
da_state = da_state.assign_coords(depths=da_state.depths-1)
da_state = da_state.rename(depths='m')
da_state.m.attrs = {'long_name': 'no. of Cliffords'}

# if params['plot']:
#     grid = da_state.plot(col='q', marker='o', figsize=(10, 5))
da_fit = fit_decay_exp(da_state, 'm')

alpha = np.exp(da_fit.sel(fit_vals='decay'))
# EPC from here: https://qiskit.org/textbook/ch-quantum-hardware/randomized-benchmarking.html#Step-5:-Fit-the-resultsfor q in machine.active_qubits:
EPC = (1 - alpha) - (1 - alpha) / 2
EPG = EPC / average_gate_per_clifford

node.results['fit_results'] = {}
for q in qubits:
    node.results['fit_results'][q.name] = {}
    node.results['fit_results'][q.name]['EPC'] = EPC.sel(qubit=q.name).values
    node.results['fit_results'][q.name]['EPG'] = EPG.sel(qubit=q.name).values
    print(f'{q.name}: EPC={EPC.sel(qubit=q.name).values}')
    print(f'{q.name}: EPG={EPG.sel(qubit=q.name).values}')


# %%    
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    da_state_qubit = da_state.sel(qubit=qubit['qubit'])
    da_state_std = ds['state'].std(dim='sequence').sel(qubit=qubit['qubit'])
    ax.errorbar(da_state_qubit.m, da_state_qubit, yerr=da_state_std, fmt='.', capsize=2, elinewidth=0.5)
    # ax.set_yticks([0, 0.5, 1])
    ax.grid('all')
    m = da_state.m.values
    ax.set_title(qubit['qubit'], pad = 22)
    ax.set_xlabel('Circuit depth')
    fit_dict = {k: da_fit.sel(
        **qubit).sel(fit_vals=k).values for k in da_fit.fit_vals.values}
    ax.plot(m, decay_exp(m, **fit_dict), 'r--', label='fit')
    ax.text(
        0., 1.07, f'RB fidelity = {1 - EPG.sel(**qubit).values:.5f}', transform=ax.transAxes)
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()

# %%
