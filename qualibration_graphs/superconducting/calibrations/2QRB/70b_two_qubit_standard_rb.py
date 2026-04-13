"""
        TWO-QUBIT STANDARD RANDOMIZED BENCHMARKING
The program consists in playing random sequences of Clifford gates and measuring the state of the resonators afterward. 
Each random sequence is generated for the maximum depth (specified as an input) and played for each depth asked by the 
user (the sequence is truncated to the desired depth). Each truncated sequence ends with the recovery gate that will 
bring the qubits back to their ground state.

The random circuits are generated offline and transpiled to a basis gate set (default is ['rz', 'sx', 'x', 'cz']). 
The circuits are executed per two-qubit layer using a switch_case block structure, allowing for efficient execution 
of the quantum circuits.

Standard randomized benchmarking provides a measure of the average gate fidelity by fitting the survival probability 
to an exponential decay as a function of circuit depth. This gives an estimate of the overall gate error rate for 
the two-qubit system.

Key Features:
    - reduce_to_1q_cliffords: When enabled (default), the Clifford gates are sampled as 1q Cliffords per qubit 
      (this is of course a much smaller subset of the whole 2q Clifford group).

Each sequence is played multiple times for averaging, and multiple random sequences are generated for each depth to 
improve statistical significance. The data is then post-processed to extract the two-qubit Clifford fidelity.

Prerequisites:
    - Having calibrated both qubits' single-qubit gates (resonator_spectroscopy, qubit_spectroscopy, rabi_chevron, power_rabi).
    - Having calibrated the two-qubit gate (cz) that will be used in the Clifford sequences.
    - Having calibrated the readout for both qubits (readout_frequency, amplitude, duration_optimization IQ_blobs).
    - Having set the appropriate flux bias points for the qubit pair.
    - Having calibrated the qubit frequencies and coupling strength.
"""

# %%


from datetime import datetime, timedelta, timezone
from typing import List, Literal, Optional

from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from calibration_utils.two_qubit_rb.data_utils import RBResult, fetch_results_as_xarray, rb_decay_curve
import xarray as xr
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter, fetching_tool
from qualibrate import QualibrationNode, NodeParameters
from calibration_utils.two_qubit_rb.circuit_utils import layerize_quantum_circuit, process_circuit_to_integers
from calibration_utils.two_qubit_rb.qua_utils import QuaProgramHandler
from quam_builder.architecture.superconducting.qpu import FluxTunableQuam as Quam
from calibration_utils.two_qubit_rb.rb_utils import (
    StandardRB,
    get_multiplexed_pair_batches,
    validate_multiplexed_batches,
)


# Average gates per 2q layer calculation:
# - Cases with non-Z gates (X/Y via .play()): assign value 2
# - Cases with only Z gates (via .frame_rotation()): assign value 0
# - Case CZ gate: assign value 1
average_gates_per_2q_layer = 0.6757


# %% {Node_parameters}

class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["qC5-qC4"] # choose one pair or several pairs that are not overlapping or nearest neighbors to run in parallel.
    circuit_lengths: list[int] = [0,2,4,8,16,32] # in number of cliffords . Going up to 64 might exceed memory limits.
    num_circuits_per_length: int = 10 # number of random circuits . Going up to 15 or more might exceed memory limits.
    num_averages: int = 15
    basis_gates: list[str] = ['rz', 'sx', 'x', 'cz'] 
    reduce_to_1q_cliffords: bool = False 
    seed: int = 0
    
    ### hardcoded parameters. Do not change. ###
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["thermal"] = "thermal" 
    multiplexed: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 6000
    load_data_id: Optional[int] = None
    timeout: int = 1000
    targets_name = "qubit_pairs"

node = QualibrationNode[Parameters, Quam](name="70b_two_qubit_standard_rb", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}

# Instantiate the QuAM class from the state file
node.machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs_raw = node.machine.active_qubit_pairs
    qubit_pairs = get_multiplexed_pair_batches(node.machine, node.parameters, node.machine.active_qubit_pair_names)
else:
    qubit_pairs_raw = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
    qubit_pairs = get_multiplexed_pair_batches(node.machine, node.parameters, [qp.id for qp in qubit_pairs_raw])

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

# Validate multiplexed batch configuration
validate_multiplexed_batches(qubit_pairs, node.parameters.multiplexed)

# Generate the OPX and Octave configurations

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = node.machine.connect()
    # qmm.close_all_quantum_machines()


config = node.machine.generate_config()


# %% {Random circuit generation}

standard_RB = StandardRB(
    amplification_lengths=node.parameters.circuit_lengths,
    num_circuits_per_length=node.parameters.num_circuits_per_length,
    basis_gates=node.parameters.basis_gates,
    reduce_to_1q_cliffords=node.parameters.reduce_to_1q_cliffords,
    num_qubits=2,
    seed=node.parameters.seed
)

transpiled_circuits = standard_RB.transpiled_circuits
transpiled_circuits_as_ints = {}
layerized_circuits = {}
for l, circuits in transpiled_circuits.items():
    layerized_circuits[l] = [layerize_quantum_circuit(qc) for qc in circuits]
    transpiled_circuits_as_ints[l] = [process_circuit_to_integers(qc) for qc in layerized_circuits[l]]

# to calculate the average number of 2q layers per Clifford
average_layers_per_clifford = np.mean([np.mean([len(circ) for circ in circuits])/np.array(length+1) for length, circuits in transpiled_circuits_as_ints.items() if length > 0])

circuits_as_ints = []
for circuits_per_len in transpiled_circuits_as_ints.values():
    for circuit in circuits_per_len:
        circuit_with_measurement = circuit + [38] # readout
        # circuit_with_measurement = [9] * len(circuit) + [66] # readout
        circuits_as_ints.append(circuit_with_measurement)

# %% {QUA_program}

qua_program_handler = QuaProgramHandler(node, circuits_as_ints, qubit_pairs)

rb = qua_program_handler.get_qua_program()
node.namespace["qua_program"] = rb

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    samples = job.get_simulated_samples()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    
    
    with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(rb)
        
        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, node.parameters.num_averages, start_time=results.start_time)
    

 # %% {Data_fetching_and_dataset_creation}
if node.parameters.load_data_id is None:
    ds = fetch_results_as_xarray(
    job.result_handles,
    qubit_pairs,
        { "sequence": range(node.parameters.num_circuits_per_length), "depths": list(node.parameters.circuit_lengths), "shots": range(node.parameters.num_averages)},
    )
else:
    node = node.load_from_id(node.parameters.load_data_id)
    ds = node.results["ds"]
# Add the dataset to the node
node.results = {"ds": ds}
# %% {Data_analysis and plotting}

# Assume ds is your input dataset and ds['state'] is your DataArray
state = ds['state']  # shape: (qubit, shots, sequence, depths)

# Outcome labels for 2-qubit states
labels = ["00", "01", "10", "11"]

# Create a list of DataArrays: one for each outcome
probs = [state == i for i in range(4)]

# Stack along a new outcome dimension
probs = xr.concat(probs, dim='outcome')

# Assign outcome labels
probs = probs.assign_coords(outcome=("outcome", labels))

probs_00 = probs.sel(outcome="00")
probs_00 = probs_00.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
probs_00 = probs_00.transpose("qubit", "repeat", "circuit_depth", "average")


probs_00 = probs_00.astype(int)

ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

rb_result = {}

# Fit the data for all pairs first
for qp in qubit_pairs:
    rb_result[qp.id] = RBResult(
        circuit_depths=list(node.parameters.circuit_lengths),
        num_repeats=node.parameters.num_circuits_per_length,
        num_averages=node.parameters.num_averages,
        state=ds_transposed.sel(qubit=qp.name).state.data
    )
    
    # Fit the data and calculate all error and fidelity metrics
    try:
        rb_result[qp.id].fit(
            average_layers_per_clifford=average_layers_per_clifford,
            average_gates_per_2q_layer=average_gates_per_2q_layer
        )
    except Exception as e:
        print(f"Warning: Fit failed for {qp.id}: {e}")
        print("Plotting data without fit parameters.")

# Create mapping from pair name to batch index (1-indexed) and sort pairs by batch order
pair_to_batch = {}
qubit_pairs_sorted_by_batch = []
for batch_idx, batch in enumerate(qubit_pairs.batch(), start=1):
    for pair_idx, qp in batch.items():
        pair_to_batch[qp.name] = batch_idx
        qubit_pairs_sorted_by_batch.append(qp)

# Display time (fixed offset; do not use node.time_zone / node.date_time)
TIME_ZONE_HOURS = 2
run_date_time = datetime.now(timezone(timedelta(hours=TIME_ZONE_HOURS))).strftime("%Y-%m-%d %H:%M:%S")

# Create a single figure with subplots for all pairs (avoid a 1×3 grid for a single pair — too wide)
num_pairs = len(qubit_pairs_sorted_by_batch)
num_cols = min(3, num_pairs) if num_pairs > 0 else 1
num_rows = int(np.ceil(num_pairs / num_cols))

fig = plt.figure(figsize=(6 * num_cols, 4.5 * num_rows))

for idx, qp in enumerate(qubit_pairs_sorted_by_batch):
    ax = fig.add_subplot(num_rows, num_cols, idx + 1)
    
    # Get the RB result for this pair
    result = rb_result[qp.id]
    
    # Calculate error bars
    error_bars = (result.data == 0).stack(combined=("average", "repeat")).std(dim="combined").state.data / np.sqrt(result.num_repeats * result.num_averages)
    
    # Get decay curve
    decay_curve = result.get_decay_curve()
    
    # Plot experimental data
    ax.errorbar(
        result.circuit_depths,
        decay_curve,
        yerr=error_bars,
        fmt=".",
        capsize=2,
        elinewidth=0.5,
        color="blue",
        label="Experimental Data",
    )
    
    # Plot fit curve if fit parameters exist
    try:
        A = result.A
        alpha = result.alpha
        B = result.B
        circuit_depths_smooth_axis = np.linspace(result.circuit_depths[0], result.circuit_depths[-1], 100)
        ax.plot(
            circuit_depths_smooth_axis,
            rb_decay_curve(np.array(circuit_depths_smooth_axis), A, alpha, B),
            color="red",
            linestyle="--",
            label="Exponential Fit",
        )
    except AttributeError:
        # Fit parameters don't exist, skip plotting fit curve
        pass
    
    # Add fidelity title if fit was successful
    try:
        fidelity = result.fidelity
        title = f"2Q average Clifford fidelity = {fidelity * 100:.2f}%"
        ax.text(
            0.5,
            0.95,
            title,
            horizontalalignment="center",
            verticalalignment="top",
            fontdict={"fontsize": "medium", "fontweight": "bold"},
            transform=ax.transAxes,
        )
    except AttributeError:
        # Show warning if fit failed
        ax.text(
            0.5,
            0.95,
            "Fit failed - insufficient data points",
            horizontalalignment="center",
            verticalalignment="top",
            fontdict={"fontsize": "medium", "fontweight": "bold", "color": "red"},
            transform=ax.transAxes,
        )
    
    # Add average gate fidelity if it was calculated
    try:
        avg_gate_fidelity = result.average_gate_fidelity
        ax.text(
            0.5,
            0.88,
            f"Average Gate Fidelity = {avg_gate_fidelity * 100:.2f}%",
            horizontalalignment="center",
            verticalalignment="top",
            fontdict={"fontsize": "medium", "fontweight": "bold"},
            transform=ax.transAxes,
        )
    except AttributeError:
        # Average gate fidelity not available, skip
        pass
    
    ax.set_xlabel("Circuit Depth")
    ax.set_ylabel(r"Probability to recover to $|00\rangle$")
    ax.set_title(f"{qp.name}")
    ax.legend(framealpha=0)
    
    # Add batch number indicator at the bottom right (outside the plot area)
    batch_num = pair_to_batch.get(qp.name, 0)
    ax.text(0.98, -0.12, str(batch_num), transform=ax.transAxes, 
           fontsize=8, ha='right', va='top',
           bbox=dict(boxstyle='circle', facecolor='plum', edgecolor='magenta', linewidth=1.2))

# Hide unused subplots
for i in range(num_pairs, num_rows * num_cols):
    row = i // num_cols
    col = i % num_cols
    ax_unused = fig.add_subplot(num_rows, num_cols, i + 1)
    ax_unused.axis('off')

fig.suptitle(
    f"2Q Randomized Benchmarking \n {run_date_time} GMT+{TIME_ZONE_HOURS} \n "
    f"reset type = {node.parameters.reset_type}, reduce_to_1q_cliffords = {node.parameters.reduce_to_1q_cliffords}",
    y=0.995,
)
fig.subplots_adjust(top=0.75, hspace=0.4, wspace=0.3)

# Add legend explaining the batch number indicator
legend_circle = mpatches.Circle((0, 0), 0.5, facecolor='plum', edgecolor='magenta', linewidth=1.2)
fig.legend(
    [legend_circle],
    ["Batch number\n(pairs run in parallel)"],
    loc="upper right",
    fontsize=8,
    framealpha=0.9,
)

fig.show()
node.results["figure_RB_decay"] = fig

# %% {Update_state}
with node.record_state_updates():
    for qp in qubit_pairs:
        if hasattr(rb_result[qp.id], 'fidelity'):
            node.machine.qubit_pairs[qp.id].macros["cz"].fidelity["StandardRB"] = {
                "error_per_clifford": 1 - rb_result[qp.id].fidelity,
                "error_per_2q_layer": rb_result[qp.id].error_per_2q_layer if hasattr(rb_result[qp.id], 'error_per_2q_layer') else None,
                "error_per_gate": rb_result[qp.id].error_per_gate if hasattr(rb_result[qp.id], 'error_per_gate') else None,
                "average_gate_fidelity": 1 - rb_result[qp.id].error_per_gate if hasattr(rb_result[qp.id], 'error_per_gate') else None,
                "alpha": rb_result[qp.id].alpha if hasattr(rb_result[qp.id], 'alpha') else None,
                "updated_at": f"{run_date_time} GMT+{TIME_ZONE_HOURS}",
            }
        else:
            print(f"Warning: Skipping state update for {qp.id} because fit failed.")
# %% {Save_results}
node.save()

# %%
