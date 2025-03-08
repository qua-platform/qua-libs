# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List, ClassVar
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
import xarray as xr
from qualang_tools.data_fetcher import XarrayDataFetcher
from qualang_tools.data_dashboard import DataDashboardClient


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    qubit_pairs: Optional[List[str]] = None
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal["active", "thermal"] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    plot_raw: bool = False
    measure_leak: bool = False


node = QualibrationNode(name="IQCC_34_2Q_confusion_matrix", parameters=Parameters())

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is not None:
    qubit_pairs = []
    for qp in machine.active_qubit_pairs:
        if qp.qubit_control.name in node.parameters.qubits and qp.qubit_target.name in node.parameters.qubits:
            qubit_pairs.append(qp)
elif node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
qubits = list(set([qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]))
num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = None
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_shots = node.parameters.num_shots
flux_point = node.parameters.flux_point_joint_or_independent

# Define sweep axes for data fetcher
sweep_axes = {
    "qubit": xr.DataArray([qp.name for qp in qubit_pairs]),
    "init_state_target": xr.DataArray([0, 1]),
    "init_state_control": xr.DataArray([0, 1]),
    "N": xr.DataArray(np.arange(n_shots)),
}

with program() as CPhase_Oscillations:
    control_initial = declare(int)
    target_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]

    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)
            with for_(*from_array(control_initial, [0, 1])):
                with for_(*from_array(target_initial, [0, 1])):
                    # Reset qubits
                    if node.parameters.reset_type == "active":
                        active_reset(qp.qubit_control)
                        active_reset(qp.qubit_target)
                        qp.align()
                    else:
                        wait(5 * qp.qubit_control.thermalization_time * u.ns)
                    qp.align()

                    # setting both qubits ot the initial state
                    with if_(control_initial == 1):
                        qp.qubit_control.xy.play("x180")
                    with if_(target_initial == 1):
                        qp.qubit_target.xy.play("x180")

                    qp.align()
                    # Readout
                    readout_state(qp.qubit_control, state_control[i])
                    readout_state(qp.qubit_target, state_target[i])
                    assign(state[i], state_control[i] * 2 + state_target[i])
                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])
                    save(state[i], state_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(2).buffer(n_shots).save(f"state_target{i + 1}")
            state_st[i].buffer(2).buffer(2).buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    exit()

with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
    job = qm.execute(CPhase_Oscillations)
    data_fetcher = XarrayDataFetcher(job, sweep_axes)
    data_dashboard = DataDashboardClient()
    for ds in data_fetcher:
        progress_counter(data_fetcher["n"], n_shots, start_time=data_fetcher.t_start)

# %% {Data_fetching_and_dataset_creation}
if node.parameters.load_data_id is not None:
    ds, machine = load_dataset(node.parameters.load_data_id)

# %% {Data_analysis}
states = [0, 1, 2, 3]
confusions = {}

for qp in qubit_pairs:
    conf = []
    for state in states:
        row = []
        for q1 in [0, 1]:
            for q0 in [0, 1]:
                row.append(
                    (ds.sel(qubit=qp.name).state.sel(init_state_target=q0, init_state_control=q1) == state).sum().values
                )
        conf.append(row)
    confusions[qp.name] = np.array(conf) / node.parameters.num_shots

# %% {Plotting}
grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
grid = QubitPairGrid(grid_names, qubit_pair_names)
for ax, qubit_pair in grid_iter(grid):
    conf = confusions[qubit_pair["qubit"]]
    ax.pcolormesh(["00", "01", "10", "11"], ["00", "01", "10", "11"], conf)
    for i in range(4):
        for j in range(4):
            if i == j:
                ax.text(i, j, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="k")
            else:
                ax.text(i, j, f"{100 * conf[i][j]:.1f}%", ha="center", va="center", color="w")
    ax.set_ylabel("prepared")
    ax.set_xlabel("measured")
    ax.set_title(qubit_pair["qubit"])
plt.tight_layout()
node.results["figure_confusion"] = grid.fig

# %% {Update_state}
if node.parameters.load_data_id is None:
    with node.record_state_updates():
        for qp in qubit_pairs:
            qp.confusion = confusions[qp.name].tolist()

# %% {Save_results}
node.results = {
    "ds": ds,
    "confusions": confusions,
    "figure_confusion": node.results["figure_confusion"],
    "initial_parameters": node.parameters.model_dump(),
}
node.outcomes = {q.name: "successful" for q in qubits}
node.machine = machine
node.save()

# %%
