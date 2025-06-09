#%% imports
import os
import json
from iqcc_cloud_client import IQCC_Cloud
import h5py
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id
from quam_libs.trackable_object import tracked_updates
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
from quam_libs.macros import active_reset
#%%
quantum_computer_backend = "arbel"  # for example qc_qwfix
qc = IQCC_Cloud(quantum_computer_backend=quantum_computer_backend)
#%%
# Get the latest state and wiring files
latest_wiring = qc.state.get_latest("wiring")
latest_state = qc.state.get_latest("state")
# Get the state folder path from environment variable
quam_state_folder_path = os.environ["QUAM_STATE_PATH"]
# Save the files
with open(os.path.join(quam_state_folder_path, "wiring.json"), "w") as f:
    json.dump(latest_wiring.data, f, indent=4)
with open(os.path.join(quam_state_folder_path, "state.json"), "w") as f:
    json.dump(latest_state.data, f, indent=4)
#%%
# Define the output file path
# output_file = os.path.expanduser("~/Desktop/spam_experiment_data.h5")
# %% {Node_parameters - node is a part of the QPU which will be used: qubits, shots etc.}- parms to use in QUA program, i.e. circuit op's.
class Parameters(NodeParameters):
    simulate: bool = False  # simulation flag
    simulation_duration_ns: int = 2500
    timeout: int = 10000  # wait time for job until cancel
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    qubits: Optional[List[str]] = ["qC3"] # manually define quantum register
    num_averages: int = 20000  # num_shots for the circuit
# The node is a QualibrationNode() object with inputs: name of the exp and Parameters()
node = QualibrationNode(name="reset_to_zero_exp", parameters=Parameters())
node_id = get_node_id()
# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# checks whether the register was set manually or use all active qubits
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)  # takes the number of qubits in the set register
# %% {QUA_program}{QUA can take parameters beforehand or to have vars defined in program using =declare(<type of var>)}
n_avg = node.parameters.num_averages  # The number of averages (circuit runs\shots) from node parameters.
flux_point = "joint"
with program() as reset_to_0_exp:
    I, _, Q, _, n, n_st = qua_declaration(num_qubits=num_qubits)
    mitigation_order = 3 # changes manually
    post_select_meas = 3 # changes manually
    total_meas = post_select_meas+ (mitigation_order + 1) * (3*mitigation_order + 2) // 2
    j = declare(int)
    # state(states) is the data structure holding the results
    # >> an array of booleans that holds the counts for each state >> size is the state size in string form (boolean)
    # >> e.g. 011 >> |011> - #if there is more than a single qubit, multiple states needed
    state = [declare(bool, size=total_meas) for _ in range(num_qubits)]
    state_stream = [declare_stream() for _ in range(num_qubits)]
    #temp_meas = declare(int) #temp place holder for the j measurements without saving
# Circuit operations
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        qubit.align()
        active_reset(qubit)
        wait(qubit.resonator.depletion_time // 4)
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(j, 0, j < total_meas, j + 1):
                qubit.align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i][j], I[i] > qubit.resonator.operations["readout"].threshold)
                save(state[i][j], state_stream[i])
                wait(qubit.resonator.depletion_time // 4)
    # Stream processing for saving results
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_stream[i].boolean_to_int().buffer(total_meas).buffer(n_avg).save(f"state{i + 1}")
# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, reset_to_0_exp, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(reset_to_0_exp)
        results = fetching_tool(job, ["n"] ,mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)
# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # get all shots
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"readout_number": range(total_meas), "n": range(n_avg)})
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}
    node.save()
#%%
# # Extract the results from the node
ds = node.results["ds"]
#%%
# Extract the measurement results from the "state" variable
# Assuming num_qubits and n_avg are defined
data = ds["state"].values
#%%
# Prepare the output file
with h5py.File(output_file, "w") as f:
    f.create_dataset("real_data", data=data)
    # Save experiment metadata # Convert to native int to avoid HDF5 error#######
    f.attrs["num_qubits"] = num_qubits
    f.attrs["n_avg"] = n_avg
    f.attrs["maxiaml_mitigation_order"] = mitigation_order
    f.attrs["post_select_meas"] = post_select_meas
    f.attrs["total_meas"] = total_meas
#post processing in the oustide file.
# %%
# Print the xarray dimensions and shape
print("Dimensions:", ds["state"].dims)
print("Shape:     ", ds["state"].shape)
# %%
