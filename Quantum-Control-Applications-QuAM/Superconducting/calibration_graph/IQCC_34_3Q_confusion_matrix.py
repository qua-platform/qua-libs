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

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["qubitC2", "qubitC1", "qubitC3"]
    num_shots: int = 2000
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "thermal"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    plot_raw : bool = False
    measure_leak : bool = False


node = QualibrationNode(
    name="IQCC_34_3Q_confusion_matrix", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

qubits = [machine.qubits[q] for q in node.parameters.qubits]

assert len(qubits) == 3

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_shots = node.parameters.num_shots  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as CPhase_Oscillations:
    a_initial = declare(int)
    b_initial = declare(int)
    c_initial = declare(int)

    n = declare(int)
    n_st = declare_stream()

    states = [declare(int) for _ in range(3)]
    states_st = [declare_stream() for _ in range(3)]

    state = declare(int)
    state_st = declare_stream()
    
    # Bring the active qubits to the minimum frequency point
    if flux_point == "independent":
        machine.apply_all_flux_to_min()
        # qp.apply_mutual_flux_point()
    elif flux_point == "joint":
        machine.apply_all_flux_to_joint_idle()
    else:
        machine.apply_all_flux_to_zero()
    wait(1000)

    with for_(n, 0, n < n_shots, n + 1):
        save(n, n_st)
        with for_(*from_array(a_initial, [0,1])):
            with for_(*from_array(b_initial, [0,1])):
                with for_(*from_array(c_initial, [0, 1])):
                    wait(5*machine.thermalization_time * u.ns)
                    # setting both qubits ot the initial state
                    qubits[0].xy.play("x180", condition=a_initial==1)
                    qubits[1].xy.play("x180", condition=b_initial==1)
                    qubits[2].xy.play("x180", condition=c_initial==1)

                    align()

                    for i in range(3):
                        readout_state(qubits[i], states[i])

                    assign(state, 4*states[0] + 2*states[1] + states[2])
                    save(state, state_st)

                    for i in range(3):
                        save(states[i], states_st[i])
        align()

    with stream_processing():
        n_st.save("n")
        # for i in range(3):
        #     states_st[i].buffer(2).buffer(2).buffer(2).buffer(n_shots).save(f"states{i + 1}")
        state_st.buffer(2).buffer(2).buffer(2).buffer(n_shots).save(f"states1")

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
        ds = fetch_results_as_xarray(job.result_handles, [qubits[0]],
                                     {"init_state_a": [0,1],
                                      "init_state_b": [0,1],
                                      "init_state_c": [0,1],
                                      "N": np.linspace(1, n_shots, n_shots)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

# %%
if not node.parameters.simulate:
    states = np.arange(8)

    conf = []
    for state in states:
        row = []
        for q2 in [0,1]:
            for q1 in [0,1]:
                for q0 in [0,1]:
                    row.append(
                        (ds.sel(qubit = qubits[0].name).states.sel(
                            init_state_a=q0,
                            init_state_b=q1,
                            init_state_c=q2,
                        ) == state).sum().values)
        conf.append(row)

    confusion = np.array(conf)/node.parameters.num_shots

# %%
if not node.parameters.simulate:
    fig, ax = plt.subplots(1,1)
    ax.pcolormesh(['000','001','010','011','100','101','110','111'],['000','001','010','011','100','101','110','111'],conf)
    for i in range(8):
        for j in range(8):
            if i==j:
                ax.text(i, j, f"{100 * confusion[i][j]:.1f}%", ha="center", va="center", color="k")
            else:
                ax.text(i, j, f"{100 * confusion[i][j]:.1f}%", ha="center", va="center", color="w")
    ax.set_ylabel('prepared')
    ax.set_xlabel('measured')
    plt.show()
    node.results["figure_confusion"] = fig
    node.results["confusion"] = confusion
    print(confusion)

with node.record_state_updates():
    machine.qC1_qC2_qC3_confusion_matrix = confusion.tolist()

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
