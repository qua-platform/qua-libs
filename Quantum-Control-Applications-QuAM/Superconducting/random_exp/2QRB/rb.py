# %%

from datetime import datetime, timezone, timedelta
from typing import List, Literal, Optional
from more_itertools import flatten
import numpy as np
from quam_libs.experiments.rb.cloud_utils import write_sync_hook
from quam_libs.lib.data_utils import split_list_by_integer_count
import xarray as xr
from qiskit.quantum_info import Operator

from qm.qua import *
from qm.qua.lib import Cast 
from qm import SimulationConfig
from qualang_tools.multi_user import qm_session

from qualang_tools.results import progress_counter, fetching_tool

from qualibrate import NodeParameters, QualibrationNode
from quam_libs.experiments.rb.circuit_processor import process_circuit_to_integers
from quam_libs.experiments.rb.qua_utils import QuaProgramHandler, play_sequence, reset_qubits
from quam_libs.lib.debug_utils import serialize_qua_program
from quam_libs.lib.plot_utils import plot_samples
from quam_libs.lib.save_utils import fetch_results_as_xarray, get_node_id
from quam_libs.macros import active_reset, qua_declaration, readout_state
from quam_libs.lib.wiring_utils import create_controller_to_qubit_mapping

from quam_libs.components import QuAM
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.experiments.rb.rb_utils import InterleavedRB, StandardRB, layerize_quantum_circuit, quantum_circuit_to_integer_and_angle_list
from quam_libs.experiments.rb.plot_utils import gate_mapping

import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# %% {Node_parameters}

class Parameters(NodeParameters):
    qubit_pairs: Optional[List[str]] = ["qC1-qC2"]
    circuit_lengths: tuple[int] = (0,1,2,4,8,16,32) # in number of cliffords
    num_circuits_per_length: int = 20
    num_averages: int = 15
    target_gate: str = "idle_2q"
    basis_gates: dict[int, str] = ['rz', 'sx', 'x', 'cz'] # ['rz', 'sx', 'x', 'cz']
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    use_input_stream: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 10000
    load_data_id: Optional[int] = None
    timeout: int = 100
    seed: int = 0

node = QualibrationNode(name="2Q_RB", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}

# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

if len(qubit_pairs) == 0:
    raise ValueError("No qubit pairs selected")

# Generate the OPX and Octave configurations

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

config = machine.generate_config()


# %% {Random circuit generation}

# stand_inter_RB = StandardRB(
#     amplification_lengths=node.parameters.circuit_lengths,
#     num_circuits_per_length=node.parameters.num_circuits_per_length,
#     basis_gates=node.parameters.basis_gates,
#     num_qubits=2,
#     seed=node.parameters.seed
# )

stand_inter_RB = InterleavedRB(
    target_gate=node.parameters.target_gate,
    amplification_lengths=node.parameters.circuit_lengths,
    num_circuits_per_length=node.parameters.num_circuits_per_length,
    basis_gates=node.parameters.basis_gates,
    num_qubits=2,
    reduce_to_1q_cliffords=True,
    seed=node.parameters.seed
)

# def test_identity_operator(circuits_dict):
#     for length, circuits in circuits_dict.items():
#         for qc in circuits:
#             # Calculate the unitary matrix of the circuit
#             unitary = Operator(qc).data
#             # Create an identity matrix of the same size
#             identity = np.eye(unitary.shape[0])
#             # Check if the unitary is close to the identity matrix
#             assert np.allclose(np.abs(unitary), identity, atol=1e-8), f"Circuit at length {length} is not identity."

# # Run the test
# test_identity_operator(stand_inter_RB.transpiled_circuits)


transpiled_circuits = stand_inter_RB.transpiled_circuits
transpiled_circuits_as_ints = {}
for l, circuits in transpiled_circuits.items():
    transpiled_circuits_as_ints[l] = [process_circuit_to_integers(layerize_quantum_circuit(qc)) for qc in circuits]

circuits_as_ints = []
for circuits_per_len in transpiled_circuits_as_ints.values():
    for circuit in circuits_per_len:
        circuit_with_measurement = circuit + [66] # readout
        circuits_as_ints.append(circuit_with_measurement)

# %% {QUA_program}

num_pairs = len(qubit_pairs)

qua_program_handler = QuaProgramHandler(node, num_pairs, circuits_as_ints, machine, qubit_pairs)

rb = qua_program_handler.get_qua_program()

# %% {Serialize_QUA_program}
# serialize_qua_program(program=rb, program_name=node.name, config=config)

# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns//4)  # in clock cycles
    job = qmm.simulate(config, rb, simulation_config)
    samples = job.get_simulated_samples()

elif node.parameters.load_data_id is None:
    # Prepare data for saving
    node.results = {}
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        if node.parameters.use_input_stream:
            num_sequences = len(qua_program_handler.sequence_lengths)
            circuits_as_ints_batched_padded = [batch + [0] * (qua_program_handler.max_current_sequence_length - len(batch)) for batch in qua_program_handler.circuits_as_ints_batched]    
            
            if machine.network['cloud']:
                write_sync_hook(circuits_as_ints_batched_padded)

                job = qm.execute(rb,
                        terminal_output=True,options={"sync_hook": "sync_hook.py"})
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
    qubit_names = [qubit_pair.qubit_control.name for qubit_pair in qubit_pairs] + [qubit_pair.qubit_target.name for qubit_pair in qubit_pairs]
    readout_lines = set([q[1] for q in qubit_names])
    fig = plot_samples(samples, qubit_names, readout_lines=list(readout_lines), xlim=(0,10000))
    
    # node.results["figure"] = fig
    # node.machine = machine
    # node.save()

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
# %% {Data_analysis}
# Assume ds is your input dataset and ds['state'] is your DataArray
state = ds['state']  # shape: (qubit, shots, sequence, depths)

# Outcome labels for 2-qubit states
labels = ["00", "01", "10", "11"]

# Create a list of DataArrays: one for each outcome
probs = [(state == i).sum(dim='shots') for i in range(4)]

# Stack along a new outcome dimension
probs = xr.concat(probs, dim='outcome') / node.parameters.num_averages

# Assign outcome labels
probs = probs.assign_coords(outcome=("outcome", labels))

# average over the sequence dimension
probs = probs.mean(dim='sequence')

probs_00 = probs.sel(outcome="00")
# %% {Plot_probs}

from scipy.optimize import curve_fit

def exp_decay(x, A, alpha):
    return A * np.exp(-alpha * x)

# Get the data for fitting
depths = probs_00.depths.values
probabilities = probs_00.values.flatten()

# Perform the fit
popt, pcov = curve_fit(exp_decay, depths, probabilities, p0=[1.0, 0.1])
A_fit, alpha_fit = popt

#fidelity
num_qubits = 2
d = 2**num_qubits
r = 1 - alpha_fit - (1 - alpha_fit) / d
fidelity = 1 - r

# Generate points for the fit curve
x_fit = np.linspace(min(depths), max(depths), 100)
y_fit = exp_decay(x_fit, A_fit, alpha_fit)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 5))

# Plot the data points
ax.plot(depths, probabilities, 'o', label='Data', color='blue')

# Plot the fit curve
ax.plot(x_fit, y_fit, '-', label=f'Fidelity: {fidelity:.3f}', color='red')

# Add labels and title
ax.set_xlabel('Circuit Depth')
ax.set_ylabel('Probability')
ax.set_title('|00⟩ State Probability vs Circuit Depth with Exponential Fit')
ax.grid(True)
ax.legend()

# Print the fit parameters
print(f"Fit parameters:")
print(f"A = {A_fit:.3f} ± {np.sqrt(pcov[0,0]):.3f}")
print(f"α = {alpha_fit:.3f} ± {np.sqrt(pcov[1,1]):.3f}")

plt.show()
# %%

import dataclasses
import numpy as np
import xarray as xr
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


@dataclasses.dataclass
class RBResult:
    """
    Class for analyzing and visualizing the results of a Randomized Benchmarking (RB) experiment.

    Attributes:
        circuit_depths (list[int]): List of circuit depths used in the RB experiment.
        num_repeats (int): Number of repeated sequences at each circuit depth.
        num_averages (int): Number of averages for each sequence.
        state (np.ndarray): Measured states from the RB experiment.
    """

    circuit_depths: list[int]
    num_repeats: int
    num_averages: int
    state: np.ndarray

    def __post_init__(self):
        """
        Initializes the xarray Dataset to store the RB experiment data.
        """
        self.data = xr.Dataset(
            data_vars={"state": (["repeat", "circuit_depth", "average"], self.state)},
            coords={
                "repeat": range(self.num_repeats),
                "circuit_depth": self.circuit_depths,
                "average": range(self.num_averages),
            },
        )

    def plot_hist(self, n_cols=3):
        """
        Plots histograms of the N-qubit state distribution at each circuit depth.

        Args:
            n_cols (int): Number of columns in the plot grid. Adjusted if fewer circuit depths are provided.
        """
        if len(self.circuit_depths) < n_cols:
            n_cols = len(self.circuit_depths)
        n_rows = max(int(np.ceil(len(self.circuit_depths) / n_cols)), 1)
        plt.figure()
        for i, circuit_depth in enumerate(self.circuit_depths, start=1):
            ax = plt.subplot(n_rows, n_cols, i)
            self.data.state.sel(circuit_depth=circuit_depth).plot.hist(ax=ax, xticks=range(4))
        plt.tight_layout()

    def plot(self):
        """
        Plots the raw recovery probability decay curve as a function of circuit depth.
        The curve is plotted using the averaged probability and without any fitting.
        """
        recovery_probability = (self.data.state == 0).sum(("repeat", "average")) / (
            self.num_repeats * self.num_averages
        )
        recovery_probability.rename("Recovery Probability").plot.line()

    def plot_with_fidelity(self):
        """
        Plots the RB fidelity as a function of circuit depth, including a fit to an exponential decay model.
        The fitted curve is overlaid with the raw data points, and error bars are included.
        """
        A, alpha, B = self.fit_exponential()
        fidelity = self.get_fidelity(alpha)

        # Compute error bars
        error_bars = (self.data == 0).mean(dim="average").std(dim="repeat").state.data

        plt.figure()
        plt.errorbar(
            self.circuit_depths,
            self.get_decay_curve(),
            yerr=error_bars,
            fmt=".",
            capsize=2,
            elinewidth=0.5,
            color="blue",
            label="Experimental Data",
        )

        circuit_depths_smooth_axis = np.linspace(self.circuit_depths[0], self.circuit_depths[-1], 100)
        plt.plot(
            circuit_depths_smooth_axis,
            rb_decay_curve(np.array(circuit_depths_smooth_axis), A, alpha, B),
            color="red",
            linestyle="--",
            label="Exponential Fit",
        )

        plt.text(
            0.5,
            0.95,
            f"2Q Clifford Fidelity = {fidelity * 100:.2f}%",
            horizontalalignment="center",
            verticalalignment="top",
            fontdict={"fontsize": "large", "fontweight": "bold"},
            transform=plt.gca().transAxes,
        )

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to $|00\rangle$")
        plt.title("2Q Randomized Benchmarking")
        plt.legend(framealpha=0)
        plt.show()

    def plot_two_qubit_state_distribution(self):
        """
        Plot how the two-qubit state is distributed as a function of circuit-depth on average.
        """
        plt.plot(
            self.circuit_depths,
            (self.data.state == 0).mean(dim="average").mean(dim="repeat").data,
            label=r"$|00\rangle$",
            marker=".",
            color="c",
            linewidth=3,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 1).mean(dim="average").mean(dim="repeat").data,
            label=r"$|01\rangle$",
            marker=".",
            color="b",
            linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 2).mean(dim="average").mean(dim="repeat").data,
            label=r"$|10\rangle$",
            marker=".",
            color="y",
            linewidth=1,
        )
        plt.plot(
            self.circuit_depths,
            (self.data.state == 3).mean(dim="average").mean(dim="repeat").data,
            label=r"$|11\rangle$",
            marker=".",
            color="r",
            linewidth=1,
        )
        plt.axhline(0.25, color="grey", linestyle="--", linewidth=2, label="2Q mixed-state")

        plt.xlabel("Circuit Depth")
        plt.ylabel(r"Probability to recover to a given state")
        plt.title("2Q State Distribution vs. Circuit Depth")
        plt.legend(framealpha=0, title=r"2Q State $\mathbf{|q_cq_t\rangle}$", title_fontproperties={"weight": "bold"})
        plt.show()

    def fit_exponential(self):
        """
        Fits the decay curve of the RB data to an exponential model.

        Returns:
            tuple: Fitted parameters (A, alpha, B) where:
                - A is the amplitude.
                - alpha is the decay constant.
                - B is the offset.
        """
        decay_curve = self.get_decay_curve()

        popt, _ = curve_fit(rb_decay_curve, self.circuit_depths, decay_curve, p0=[0.75, 0.9, 0.25], maxfev=10000)
        A, alpha, B = popt

        return A, alpha, B

    def get_fidelity(self, alpha):
        """
        Calculates the average fidelity per Clifford based on the decay constant.

        Args:
            alpha (float): Decay constant from the exponential fit.

        Returns:
            float: Estimated average fidelity per Clifford.
        """
        n_qubits = 2  # Assuming 2 qubits as per the context
        d = 2**n_qubits
        r = 1 - alpha - (1 - alpha) / d
        fidelity = 1 - r

        return fidelity

    def get_decay_curve(self):
        """
        Calculates the decay curve from the RB data.

        Returns:
            np.ndarray: Decay curve representing the fidelity as a function of circuit depth.
        """
        return (self.data.state == 0).sum(("repeat", "average")) / (self.num_repeats * self.num_averages)


def rb_decay_curve(x, A, alpha, B):
    """
    Exponential decay model for RB fidelity.

    Args:
        x (array-like): Circuit depths.
        A (float): Amplitude of the decay.
        alpha (float): Decay constant.
        B (float): Offset of the curve.

    Returns:
        np.ndarray: Calculated decay curve.
    """
    return A * alpha**x + B


def get_interleaved_gate_fidelity(num_qubits: int, reference_alpha: float, interleaved_alpha: float):
    """
    Calculates the interleaved gate fidelity using the formula from https://arxiv.org/pdf/1210.7011.

    Args:
        num_qubits (int): Number of qubits involved.
        reference_alpha (float): Decay constant from the reference RB experiment.
        interleaved_alpha (float): Decay constant from the interleaved RB experiment.

    Returns:
        float: Estimated interleaved gate fidelity.
    """
    return 1 - ((2**num_qubits - 1) * (1 - interleaved_alpha / reference_alpha) / 2**num_qubits)

# %%
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

# %%
ds_transposed = ds.rename({"shots": "average", "sequence": "repeat", "depths": "circuit_depth"})
ds_transposed = ds_transposed.transpose("qubit", "repeat", "circuit_depth", "average")


rb_result = RBResult(
    circuit_depths=list(node.parameters.circuit_lengths),
    num_repeats=node.parameters.num_circuits_per_length,
    num_averages=node.parameters.num_averages,
    state=ds_transposed.sel(qubit="qC1-qC2").state.data
)

rb_result.plot_with_fidelity()

# %%
