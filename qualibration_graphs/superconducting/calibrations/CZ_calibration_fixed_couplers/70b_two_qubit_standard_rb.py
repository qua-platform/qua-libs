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
    - use_input_stream: When enabled (default), the circuit sequences are streamed to the OPX in using the
      input stream feature. This allows for dynamic circuit execution and reduces memory usage on the OPX.

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

import xarray as xr

# from iqcc_calibration_tools.qualibrate_config.qualibrate.node import NodeParameters, QualibrationNode
from calibration_utils.two_qubit_interleaved_rb.circuit_utils import (
    layerize_quantum_circuit,
    process_circuit_to_integers,
)

# from iqcc_calibration_tools.quam_config.components import Quam
# from calibration_utils.two_qubit_interleaved_rb.cloud_utils import write_sync_hook
from calibration_utils.two_qubit_interleaved_rb.data_utils import RBResult
from calibration_utils.two_qubit_interleaved_rb.plot_utils import gate_mapping
from calibration_utils.two_qubit_interleaved_rb.qua_utils import QuaProgramHandler
from calibration_utils.two_qubit_interleaved_rb.rb_utils import StandardRB
from matplotlib import pyplot as plt
from more_itertools import flatten
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibrate import QualibrationNode
from qualibrate.parameters import NodeParameters
from qualibration_libs.data import XarrayDataFetcher
from quam_config import Quam
import numpy as np
# from iqcc_calibration_tools.analysis.plot_utils import plot_samples
# from iqcc_calibration_tools.storage.save_utils import fetch_results_as_xarray


# %% {Node_parameters}


class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["qB2-qB4"]
    circuit_lengths: tuple[int] = (0, 1, 2, 4, 8, 16, 32)  # in number of cliffords
    num_circuits_per_length: int = 15
    num_averages: int = 15
    basis_gates: list[str] = ["rz", "sx", "x", "cz"]
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    reduce_to_1q_cliffords: bool = True
    use_input_stream: bool = False
    simulate: bool = False
    simulation_duration_ns: int = 10000
    load_data_id: Optional[int] = None
    timeout: int = 100
    seed: int = 0


node = QualibrationNode[Parameters, Quam](name="70b_two_qubit_standard_rb", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}

# Instantiate the QuAM class from the state file
node.machine = Quam.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = node.machine.active_qubit_pairs
else:
    qubit_pairs = [node.machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

# Generate the OPX and Octave configurations

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = node.machine.connect()

config = node.machine.generate_config()


# %% {Random circuit generation}

standard_RB = StandardRB(
    amplification_lengths=node.parameters.circuit_lengths,
    num_circuits_per_length=node.parameters.num_circuits_per_length,
    basis_gates=node.parameters.basis_gates,
    num_qubits=2,
)

transpiled_circuits = standard_RB.transpiled_circuits
transpiled_circuits_as_ints = {}
for l, circuits in transpiled_circuits.items():
    transpiled_circuits_as_ints[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]

circuits_as_ints = []
for circuits_per_len in transpiled_circuits_as_ints.values():
    for circuit in circuits_per_len:
        circuit_with_measurement = circuit + [66]  # readout
        circuits_as_ints.append(circuit_with_measurement)

# %% {QUA_program}

num_pairs = len(qubit_pairs)

qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, node.machine, qubit_pairs)

rb = node.namespace["qua_program"] = qua_program_handler.get_qua_program()

# %%

node.namespace["sweep_axes"] = {
    "sequence": xr.DataArray(np.arange(node.parameters.num_circuits_per_length)),
    "depths": xr.DataArray(np.array(node.parameters.circuit_lengths)),
    "shots": xr.DataArray(np.arange(node.parameters.num_averages)),
}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset
    called "ds_raw".
    """
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_averages,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns // 4)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    samples = job.get_simulated_samples()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")

    with qm_session(node.machine.qmm, config, timeout=node.parameters.timeout) as qm:
        if node.parameters.use_input_stream:
            num_sequences = len(qua_program_handler.sequence_lengths)
            circuits_as_ints_batched_padded = [
                batch + [0] * (qua_program_handler.max_current_sequence_length - len(batch))
                for batch in qua_program_handler.circuits_as_ints_batched
            ]

            if node.machine.network["cloud"]:
                write_sync_hook(circuits_as_ints_batched_padded)

                job = qm.execute(rb, terminal_output=True, options={"sync_hook": "sync_hook.py"})
            else:
                job = qm.execute(rb)
                for id, batch in enumerate(circuits_as_ints_batched_padded):
                    job.push_to_input_stream("sequence", batch)
                    print(f"{id}/{num_sequences}: Received ")

        else:
            job = qm.execute(rb)

        results = fetching_tool(job, ["iteration"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, node.parameters.num_averages, start_time=results.start_time)

# %% {Plot_sequence}

for num in flatten(circuits_as_ints):
    print(gate_mapping[num])

# %% {Plot and save if simulation}
if node.parameters.simulate:
    qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [
        qubit_pair.qubit_target.name for qubit_pair in qubit_pairs
    ]
    readout_lines = set([q[1] for q in qubit_names])
    fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0, 10000))

    # node.results["figure"] = fig
    # node.save()

# %% {Data_fetching_and_dataset_creation}
if node.parameters.load_data_id is None:
    ds = fetch_results_as_xarray(
        job.result_handles,
        qubit_pairs,
        {
            "sequence": range(node.parameters.num_circuits_per_length),
            "depths": list(node.parameters.circuit_lengths),
            "shots": range(node.parameters.num_averages),
        },
    )
else:
    node = node.load_from_id(node.parameters.load_data_id)
    ds = node.results["ds"]
# Add the dataset to the node
node.results = {"ds": ds}
# %% {Data_analysis and plotting}

# Assume ds is your input dataset and ds['state'] is your DataArray
state = ds["state"]  # shape: (qubit, shots, sequence, depths)

# Outcome labels for 2-qubit states
labels = ["00", "01", "10", "11"]

# Create a list of DataArrays: one for each outcome
probs = [state == i for i in range(4)]

# Stack along a new outcome dimension
probs = xr.concat(probs, dim="outcome")

# Assign outcome labels
probs = probs.assign_coords(outcome=("outcome", labels))

probs_00 = probs.sel(outcome="00")
probs_00 = probs_00.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
probs_00 = probs_00.transpose("qubit", "repeat", "circuit_depth", "average")


probs_00 = probs_00.astype(int)

ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")

for qp in qubit_pairs:

    rb_result = RBResult(
        circuit_depths=list(node.parameters.circuit_lengths),
        num_repeats=node.parameters.num_circuits_per_length,
        num_averages=node.parameters.num_averages,
        state=ds_transposed.sel(qubit=qp.name).state.data,
    )

    fig = rb_result.plot_with_fidelity()
    fig.suptitle(f"2Q Randomized Benchmarking - {qp.name}")
    node.add_node_info_subtitle(fig)
    fig.show()

# %% {Update_state}
with node.record_state_updates():
    for qp in qubit_pairs:
        node.machine.qubit_pairs[qp.id].macros["cz"].fidelity["StandardRB"] = rb_result.fidelity
        node.machine.qubit_pairs[qp.id].macros["cz"].fidelity["StandardRB_alpha"] = rb_result.alpha

# %% {Save_results}
node.save()

# %%
