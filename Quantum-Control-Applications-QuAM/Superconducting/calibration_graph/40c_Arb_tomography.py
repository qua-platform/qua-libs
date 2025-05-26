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
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, save_node
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="40c_Arb_tomography", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################
from matplotlib.colors import LinearSegmentedColormap

def plot_3d_hist_with_frame(data,ideal, title = ''):
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), subplot_kw={'projection': '3d'})
    # Create a grid of positions for the bars
    xpos, ypos = np.meshgrid(np.arange(4) + 0.5, np.arange(4) + 0.5, indexing="ij")
    xpos = xpos.ravel()
    ypos = ypos.ravel()
    zpos = np.zeros_like(xpos)
    # Create a custom colormap with two distinct colors for positive and negative values
    colors = [(0.1, 0.1, 0.6), (0.55, 0.55, 1.0)]  # Light blue for positive, dark blue for negative
    cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    # finding global min,max
    gmin = np.min([np.min(np.real(data)),np.min(np.imag(data)),np.min(np.real(ideal)),np.min(np.imag(ideal))])
    gmax = np.max([np.max(np.real(data)),np.max(np.imag(data)),np.max(np.real(ideal)),np.max(np.imag(ideal))])

    # Use the bar3d function with the 'color' parameter to color the bars
    for i in range(2):
        if i == 0:
            dz = np.real(data).ravel()
            dzi = np.real(ideal).ravel()
            axs[i].set_title('real')
        else:
            dz = np.imag(data).ravel()
            dzi = np.imag(ideal).ravel()
            axs[i].set_title('imaginary')            
        axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dz, alpha= 1, color=cmap(np.sign(dz)))
        axs[i].bar3d(xpos, ypos, zpos, dx=0.4, dy=0.4, dz=dzi, alpha= 0.1, edgecolor = 'k')
        # Set tick labels for x and y axes
        axs[i].set_xticks(np.arange(1, 5))
        axs[i].set_yticks(np.arange(1, 5))
        axs[i].set_xticklabels(['00', '01', '10', '11'])
        axs[i].set_yticklabels(['00', '01', '10', '11'])
        axs[i].set_xticklabels(['00', '01', '10', '11'], rotation=45)
        axs[i].set_yticklabels(['00', '01', '10', '11'], rotation=45)
        axs[i].set_zlim([gmin,gmax])
    fig.suptitle(title)
    # Show the plot
    
    return fig    

def flatten(data):
    if isinstance(data, tuple):
        if len(data) == 0:
            return ()
        else:
            return flatten(data[0]) + flatten(data[1:])
    else:
        return (data,)
    
def generate_pauli_basis(n_qubits):    
    pauli = np.array([0,1,2,3])
    paulis = pauli
    for i in range(n_qubits-1):
        new_paulis = []
        for ps in paulis:
            for p in pauli:
                new_paulis.append(flatten((ps, p)))
        paulis = new_paulis
    return paulis
        
def gen_inverse_hadamard(n_qubits):
    H = np.array([[1,1],[1,-1]])/2
    for _ in range(n_qubits-1):
        H = np.kron(H, H)
    return np.linalg.inv(H)

def get_pauli_data(da):

    pauli_basis = generate_pauli_basis(2)

    inverse_hadamard = gen_inverse_hadamard(2)

    # Create an xarray Dataset with dimensions and coordinates based on pauli_basis
    paulis_data = xr.Dataset(
        {
            "value": (["pauli_op"], np.zeros(len(pauli_basis))),
            "appearances": (["pauli_op"], np.zeros(len(pauli_basis), dtype=int))
        },
        coords={'pauli_op': [','.join(map(str, op)) for op in pauli_basis]}
    )

    for tomo_axis in da.coords['tomo_axis'].values:
        tomo_data = da.sel(tomo_axis = tomo_axis)
        pauli_data = inverse_hadamard @ tomo_data.data
        paulis = ["0,0", f"{tomo_axis[0]+1},0", f"0,{tomo_axis[1]+1}", f"{tomo_axis[0]+1},{tomo_axis[1]+1}"]
        for i, pauli in enumerate(paulis):
            paulis_data.value.loc[{'pauli_op': pauli}] += pauli_data[i]
            paulis_data.appearances.loc[{'pauli_op': pauli}] += 1
        
    paulis_data = xr.where(paulis_data.appearances != 0, paulis_data.value / paulis_data.appearances, paulis_data.value)
    
    return paulis_data


def get_density_matrix(paulis_data):
    # 2Q
    # Define the Pauli matrices
    I = np.array([[1, 0], [0, 1]])
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])

    # Create a vector of the Pauli matrices
    pauli_matrices = [I, X, Y, Z]

    rho = np.zeros((4,4))

    for i, pauli_i in enumerate(pauli_matrices):
        for j, pauli_j in enumerate(pauli_matrices):
            rho = rho + 0.25*paulis_data.sel(pauli_op = f"{i},{j}").values * np.kron(pauli_i, pauli_j)
    
    return rho

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as CPhase_Oscillations:
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    tomo_axis_control = declare(int)
    tomo_axis_target = declare(int)
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp.qubit_control)

        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st) 
            with for_(tomo_axis_control, 0, tomo_axis_control < 3, tomo_axis_control + 1):
                with for_(tomo_axis_target, 0, tomo_axis_target < 3, tomo_axis_target + 1):
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            active_reset(qp.qubit_target)
                    else:
                        wait(5*qp.qubit_control.thermalization_time * u.ns)
                    qp.align()
                    # Bell state
                    qp.qubit_control.xy.play("-y90")
                    qp.qubit_target.xy.play("-y90")
                    qp.gates['Cz'].execute()
                    # qp.align()
                    # qp.gates['Cz'].execute()
                    # qp.align()
                    # qp.qubit_control.xy.play("y90")
                    # qp.qubit_target.xy.play("y180")                    
                    # qp.qubit_control.xy.play("y90")
                    qp.align()
                    # tomography pulses
                    with if_(tomo_axis_control == 0): #X axis
                        qp.qubit_control.xy.play("y90")
                    with elif_(tomo_axis_control == 1): #Y axis
                        qp.qubit_control.xy.play("x90")
                    with if_(tomo_axis_target == 0): #X axis
                        qp.qubit_target.xy.play("y90")
                    with elif_(tomo_axis_target == 1): #Y axis
                        qp.qubit_target.xy.play("x90")
                    qp.align()            
                    # readout
                    readout_state(qp.qubit_control, state_control[i])
                    readout_state(qp.qubit_target, state_target[i])
                    assign(state[i], state_control[i]*2 + state_target[i])
                    save(state_control[i], state_st_control[i])
                    save(state_target[i], state_st_target[i])
                    save(state[i], state_st[i])
                align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_control{i + 1}")
            state_st_target[i].buffer(3).buffer(3).buffer(n_shots).save(f"state_target{i + 1}")
            state_st[i].buffer(3).buffer(3).buffer(n_shots).save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_shots, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"tomo_axis_target": [0,1,2], "tomo_axis_control": [0,1,2], "N": np.linspace(1, n_shots, n_shots)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %%
import xarray as xr
if not node.parameters.simulate:
    states = [0,1,2,3]

    results = []
    for state in states:
        results.append((ds.state == state).sum(dim = "N") / node.parameters.num_shots)
        
results_xr = xr.concat(results, dim=xr.DataArray(states, name="state"))
results_xr = results_xr.rename({"dim_0": "state"})
results_xr = results_xr.stack(
        tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

corrected_results = []
for qp in qubit_pairs:
    corrected_results_qp = [] 
    for tomo_axis_control in [0,1,2]:
        corrected_results_control = []
        for tomo_axis_target in [0,1,2]:
            results = results_xr.sel(tomo_axis_control = tomo_axis_control, tomo_axis_target = tomo_axis_target, 
                                     qubit = qp.name)
            results = np.linalg.inv(qp.confusion) @ results.data

            results = results * (results > 0)
            results = results / results.sum()
            corrected_results_control.append(results)
        corrected_results_qp.append(corrected_results_control)
    corrected_results.append(corrected_results_qp)

    # %%

    # Convert corrected_results to an xarray DataArray
    corrected_results_xr = xr.DataArray(
        corrected_results,
        dims=['qubit', 'tomo_axis_control', 'tomo_axis_target', 'state'],
        coords={
            'qubit': [qp.name for qp in qubit_pairs],
            'tomo_axis_control': [0, 1, 2],
            'tomo_axis_target': [0, 1, 2],
            'state': ['00', '01', '10', '11']
        }
    )
    corrected_results_xr = corrected_results_xr.stack(
            tomo_axis=['tomo_axis_target', 'tomo_axis_control'])

    # Store the xarray in the node results

    # %%

    paulis_data = {}
    rhos = {}
    for qp in qubit_pairs:
        paulis_data[qp.name] = get_pauli_data(corrected_results_xr.sel(qubit = qp.name))
        rhos[qp.name] = get_density_matrix(paulis_data[qp.name])
        
    # %%

# %%
if not node.parameters.simulate:
    
    for qp in qubit_pairs:
        ideal_dat = np.array([[1,0,0,1],[0,0,0,0],[0,0,0,0],[1,0,0,1]])/2
        fig = plot_3d_hist_with_frame(rhos[qp.name], ideal_dat, title = qp.name)
        node.results[f"{qp.name}_figure_city"] = fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        rho = np.real(rhos[qubit_pair['qubit']])
        ax.pcolormesh(rho, vmin = -0.5, vmax = 0.5, cmap = "RdBu")
        # plt.colorbar(ax.pcolormesh(rho), ax=ax)
        for i in range(4):
            for j in range(4):
                if np.abs(rho[i][j]) < 0.1:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                else:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
        ax.set_title(qubit_pair['qubit'])
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Pauli Operators')
        ax.set_xticks(range(4), ['00', '01', '10', '11'])
        ax.set_yticks(range(4), ['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
        ax.set_yticklabels(['00', '01', '10', '11'])
    grid.fig.suptitle(f"Bell state tomography (real part)")
    node.results["figure_rho_real"] = grid.fig
        
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        rho = np.imag(rhos[qubit_pair['qubit']])
        ax.pcolormesh(rho, vmin = -0.1, vmax = 0.1, cmap = "RdBu")
        # plt.colorbar(ax.pcolormesh(rho), ax=ax)
        for i in range(4):
            for j in range(4):
                if np.abs(rho[i][j]) < 0.1:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="k")
                else:
                    ax.text(i+0.5, j+0.5, f"{ rho[i][j]:.2f}", ha="center", va="center", color="w")
        ax.set_title(qubit_pair['qubit'])
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Pauli Operators')
        ax.set_xticks(range(4), ['00', '01', '10', '11'])
        ax.set_yticks(range(4), ['00', '01', '10', '11'])
        ax.set_xticklabels(['00', '01', '10', '11'], rotation=45, ha='right')
        ax.set_yticklabels(['00', '01', '10', '11'])
    grid.fig.suptitle(f"Bell state tomography (imaginary part)")
    node.results["figure_rho_imag"] = grid.fig

    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        # Extract the values and labels for plotting
        values = paulis_data[qubit_pair['qubit']].values
        labels = paulis_data[qubit_pair['qubit']].coords['pauli_op'].values

        # Create a bar plot
        bars = ax.bar(range(len(values)), values)

        # Customize the plot
        ax.set_xlabel('Pauli Operators')
        ax.set_ylabel('Value')
        ax.set_title(qubit_pair['qubit'])
        ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right')

        # Add value labels on top of each bar
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom')

# Adjust layout and display the plot
    plt.tight_layout()
    plt.show()
    node.results["figure_paulis"] = grid.fig
# %%

# %%

# %% {Update_state}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    # save_node(node)
        
# %%
