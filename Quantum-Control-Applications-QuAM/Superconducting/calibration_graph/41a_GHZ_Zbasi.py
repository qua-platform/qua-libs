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

    qubit_triplets: List[List[str]] = [["qubitC2","qubitC4","qubitC3"],["qubitC2","qubitC1","qubitC3"]]# list of lists of thwe qubits making up the GHZ state
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None


node = QualibrationNode(
    name="41a_GHZ_Zbasis", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

# %%

class qubit_triplet:
    def __init__(self, qubit_A, qubit_B, qubit_C):
        self.qubit_A = qubit_A
        self.qubit_B = qubit_B
        self.qubit_C = qubit_C
        
        for qp in machine.qubit_pairs:
            if machine.qubit_pairs[qp].qubit_control in [qubit_A, qubit_B] and machine.qubit_pairs[qp].qubit_target in [qubit_A, qubit_B]:
                self.qubit_pair_AB = machine.qubit_pairs[qp]
            if machine.qubit_pairs[qp].qubit_control in [qubit_B, qubit_C] and machine.qubit_pairs[qp].qubit_target in [qubit_B, qubit_C]:
                self.qubit_pair_BC = machine.qubit_pairs[qp]
                
        self.name = f"{qubit_A.name}-{qubit_B.name}-{qubit_C.name}"

qubit_triplets = [[machine.qubits[qubit] for qubit in triplet] for triplet in node.parameters.qubit_triplets]
qubit_triplets = [qubit_triplet(qubit_triplets[0], qubit_triplets[1], qubit_triplets[2]) for qubit_triplets in qubit_triplets]
num_qubit_triplets = len(qubit_triplets)

####################
# Helper functions #
####################


# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as CPhase_Oscillations:
    n = declare(int)
    n_st = declare_stream()
    state_A = [declare(int) for _ in range(num_qubit_triplets)]
    state_B = [declare(int) for _ in range(num_qubit_triplets)]
    state_C = [declare(int) for _ in range(num_qubit_triplets)]
    state = [declare(int) for _ in range(num_qubit_triplets)]
    state_st_A = [declare_stream() for _ in range(num_qubit_triplets)]
    state_st_B = [declare_stream() for _ in range(num_qubit_triplets)]
    state_st_C = [declare_stream() for _ in range(num_qubit_triplets)]
    state_st = [declare_stream() for _ in range(num_qubit_triplets)]
    
    for i, qubit_triplet in enumerate(qubit_triplets):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qubit_triplet.qubit_A)
        align()
        
        with for_(n, 0, n < n_shots, n + 1):
            save(n, n_st)         
            # reset
            if node.parameters.reset_type == "active":
                active_reset(qubit_triplet.qubit_A)
                active_reset(qubit_triplet.qubit_B)
                active_reset(qubit_triplet.qubit_C)

            else:
                wait(5*qubit_triplet.qubit_A.thermalization_time * u.ns)
            align()
            # Bell state AB
            # qubit_triplet.qubit_A.xy.play("y90")
            # qubit_triplet.qubit_B.xy.play("y90")
            # qubit_triplet.qubit_pair_AB.gates['Cz'].execute()
            # qubit_triplet.qubit_A.xy.play("-y90")
            
            # Bell state BC
            # qubit_triplet.qubit_C.xy.play("y90")
            # qubit_triplet.qubit_B.xy.play("y90")
            # qubit_triplet.qubit_pair_BC.gates['Cz'].execute()
            # qubit_triplet.qubit_C.xy.play("-y90")
            
            # GHZ
            qubit_triplet.qubit_B.xy.play("y90")
            qubit_triplet.qubit_A.xy.play("y90")
            qubit_triplet.qubit_pair_AB.gates['Cz'].execute()
            qubit_triplet.qubit_A.xy.play("-y90")
            align()
            qubit_triplet.qubit_C.xy.play("y90")
            qubit_triplet.qubit_pair_BC.gates['Cz'].execute()
            qubit_triplet.qubit_C.xy.play("-y90")
            
            align()
      
            readout_state(qubit_triplet.qubit_A, state_A[i])
            readout_state(qubit_triplet.qubit_B, state_B[i])
            readout_state(qubit_triplet.qubit_C, state_C[i])
            assign(state[i], state_A[i]*4 + state_B[i]*2 + state_C[i])
            save(state_A[i], state_st_A[i])
            save(state_B[i], state_st_B[i])
            save(state_C[i], state_st_C[i])
            save(state[i], state_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_triplets):
            state_st_A[i].buffer(n_shots).save(f"state_A{i + 1}")
            state_st_B[i].buffer(n_shots).save(f"state_B{i + 1}")
            state_st_C[i].buffer(n_shots).save(f"state_C{i + 1}")
            state_st[i].buffer(n_shots).save(f"state{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_triplets, {"N": np.linspace(1, n_shots, n_shots)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}
    
# %%
if not node.parameters.simulate:
    states = [0,1,2,3,4,5,6,7]

    results = {}
    corrected_results = {}
    for qubit_triplet in qubit_triplets:
        results[qubit_triplet.name] = []
        for state in states:
            results[qubit_triplet.name].append((ds.sel(qubit = qubit_triplet.name).state == state).sum().values)
        results[qubit_triplet.name] = np.array(results[qubit_triplet.name])/node.parameters.num_shots
        conf_mat = np.array([[1]])
        for q in [qubit_triplet.qubit_A, qubit_triplet.qubit_B, qubit_triplet.qubit_C]:
            conf_mat = np.kron(conf_mat, q.resonator.confusion_matrix)
        # conf_mat = qp.confusion
        corrected_results[qubit_triplet.name] = np.linalg.inv(conf_mat) @ results[qubit_triplet.name]
        # corrected_results[qubit_triplet.name] = results[qubit_triplet.name]
        print(f"{qubit_triplet.name}: {corrected_results[qubit_triplet.name]}")



# %%
if not node.parameters.simulate:
    if num_qubit_triplets == 1:
        f,axs = plt.subplots(1,figsize=(6,3))
    else:
        f,axs = plt.subplots(num_qubit_triplets,1,figsize=(6,3*num_qubit_triplets))
    
    for i, qubit_triplet in enumerate(qubit_triplets):
        if num_qubit_triplets == 1:
            ax = axs
        else:
            ax = axs[i]
        ax.bar(['000', '001', '010', '011', '100', '101', '110', '111'], corrected_results[qubit_triplet.name], color='skyblue', edgecolor='navy')
        ax.set_ylim(0, 1)
        for i, v in enumerate(corrected_results[qubit_triplet.name]):
            ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        ax.set_ylabel('Probability')
        ax.set_xlabel('State')
        ax.set_title(qubit_triplet.name)
    plt.show()
    node.results["figure"] = f
    
    # grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    # grid = QubitPairGrid(grid_names, qubit_pair_names)
    # for ax, qubit_pair in grid_iter(grid):
    #     print(qubit_pair['qubit'])
    #     corrected_res = corrected_results[qubit_pair['qubit']]
    #     ax.bar(['00', '01', '10', '11'], corrected_res, color='skyblue', edgecolor='navy')
    #     ax.set_ylim(0, 1)
    #     for i, v in enumerate(corrected_res):
    #         ax.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    #     ax.set_ylabel('Probability')
    #     ax.set_xlabel('State')
    #     ax.set_title(qubit_pair['qubit'])
    # plt.show()
    # node.results["figure"] = grid.fig
# %%

# %% {Update_state}

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%
