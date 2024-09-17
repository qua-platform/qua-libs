# %%
"""

"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 200
    operation: str = "x180"
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 3000
    wait_time_step_in_ns: int = 20
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="MW_crosstalk",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, readout_state
from quam.components import pulses
import copy
import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation, oscillation
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

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

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)


        
###################
# The QUA program #
###################

operation = node.parameters.operation  # The qubit operation to play, can be switched to "x180" when the qubits are found.
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# tracked_qubits = []
# max_amp = 0.49
# for i, qubit in enumerate(qubits):
#     with tracked_updates(qubit, auto_revert=False, dont_assign_to_none=True) as qubit:
#         qubit.xy.operations[operation].amplitude = max_amp
#         qubit.xy.operations[operation].length = 80
#         tracked_qubits.append(qubit)
# config = machine.generate_config()
# for tracked_qubit in tracked_qubits:
#     tracked_qubit.revert_changes()


# Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)
with program() as time_rabi:
    n = declare(int)
    n_st = declare_stream()
    state = [declare(int) for _ in range(num_qubits * (num_qubits ))]
    state_stream = [declare_stream() for _ in range(num_qubits * (num_qubits ))]
    
    t = declare(int)  

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        for j, qubit2 in enumerate(qubits):
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)            
                with for_(*from_array(t, idle_times)):   
                    qubit.resonator.wait(machine.thermalization_time * u.ns)
                    align()
                    # qubit2.xy.update_frequency(qubit.xy.intermediate_frequency)
                    qubit2.xy.play(operation, duration = t)
                    # qubit2.xy.update_frequency(qubit2.xy.intermediate_frequency)
                    align()
                    readout_state(qubit, state[i*num_qubits + j - 1])
                    save(state[i*num_qubits + j - 1], state_stream[i*num_qubits + j - 1])


            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            for j in range(num_qubits):
                state_stream[i*num_qubits + j-1].buffer(len(idle_times)).average().save(f"state{j+1}_{i+1}")


###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, time_rabi, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    quit()
else:
    # Open the quantum machine
    qm = qmm.open_qm(config)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(time_rabi, flags=['auto-element-thread'])
    # Get results from QUA program
    data_list = ["n"] 
    results = fetching_tool(job, data_list, mode="live")
    # Live plotting
    # fig = plt.figure()
    # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
    while results.is_processing():
        fetched_data = results.fetch_all()
        n = fetched_data[0]
        progress_counter(n, n_avg, start_time=results.start_time)

    # Close the quantum machines at the end in order to put all flux biases to 0 so that the fridge doesn't heat-up
    qm.close()

    plt.show()

# %%
handles = job.result_handles
# %%
import xarray as xr

def extract_string(input_string):
    # Find the index of the first occurrence of a digit in the input string
    index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)
    
    if index is not None:
        # Extract the substring from the start of the input string to the index
        extracted_string = input_string[:index]
        return extracted_string
    else:
        return None
    
stream_handles = handles.keys()
meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
values = [handles.get(f'{meas_vars[0]}{j+1}_{i+1}').fetch_all() for i, qubit in enumerate(qubits) for j, qubit in enumerate(qubits)]

ds = xr.Dataset(
    {
        f"{meas_vars[0]}": (["qubit_pair", "time"], np.array(values).reshape(-1, len(idle_times)))
    },
    coords={
        "time": 4*idle_times,
        "qubit_pair": [f"q{i+1}_q{j+1}" for i in range(len(qubits)) for j in range(len(qubits))]
    }
)


# %%

fit = fit_oscillation_decay_exp(ds.state, 'time')
fit_evals = oscillation_decay_exp(ds.time,fit.sel(fit_vals = 'a'),
                                  fit.sel(fit_vals = 'f'),fit.sel(fit_vals = 'phi'),
                                  fit.sel(fit_vals = 'offset'),fit.sel(fit_vals = 'decay'))

# Add fit_evals to the dataset
ds['fit'] = (('qubit_pair', 'time'), fit_evals.values)

# Update the plotting code to include the fit
fig, axs = plt.subplots(num_qubits, num_qubits, figsize=(4*num_qubits, 4*num_qubits))

# Flatten the axs array if it's not already 1D (happens when num_qubits > 1)
if num_qubits > 1:
    axs = axs.flatten()
else:
    axs = [axs]  # Convert to list if there's only one subplot

# Iterate over all qubit pairs
for i in range(num_qubits):
    for j in range(num_qubits):
        qubit_pair = f"q{i+1}_q{j+1}"
        ax = axs[i*num_qubits + j]
        
        # Plot the data and fit for this qubit pair
        ds[meas_vars[0]].sel(qubit_pair=qubit_pair).plot(ax=ax, label='Data')
        ds['fit'].sel(qubit_pair=qubit_pair).plot(ax=ax, label='Fit')
        
        ax.set_title(f"Qubit Pair: {qubit_pair}")
        ax.set_xlabel("Time (ns)")
        ax.set_ylabel(meas_vars[0])
        ax.legend()

plt.tight_layout()
plt.show()

# Save the updated figure to the node results
node.results['crosstalk_matrix_with_fit'] = fig

# %%

# Create a crosstalk matrix
crosstalk_matrix = np.zeros((num_qubits, num_qubits))

# Calculate the crosstalk values
for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            qubit_pair = f"q{i+1}_q{j+1}"
            fit_params = fit.sel(qubit_pair=qubit_pair)
            amplitude = fit_params.sel(fit_vals='a').values
            frequency = fit_params.sel(fit_vals='f').values
            if np.abs(amplitude) > 0.3:
                crosstalk_matrix[i, j] = 0.5/frequency

crosstalk_matrix
# %%
# Normalize the crosstalk matrix by the length of the xy.operations[operation] for each qubit
for i in range(num_qubits):
    for j in range(num_qubits):
        if i != j:
            # Get the length of the operation for the control qubit (i)
            control_qubit = qubits[i]
            operation_length = control_qubit.xy.operations[operation].length
            
            # Normalize the crosstalk value
            if operation_length > 0:
                crosstalk_matrix[i, j] = operation_length / crosstalk_matrix[i, j]
        else:
            crosstalk_matrix[i, j] = 1

# Find the maximal value off the diagonal
off_diagonal_mask = ~np.eye(crosstalk_matrix.shape[0], dtype=bool)
max_off_diagonal = np.max(crosstalk_matrix[off_diagonal_mask])
min_off_diagonal = np.min(crosstalk_matrix[off_diagonal_mask])

# Set the diagonal elements to 1
np.fill_diagonal(crosstalk_matrix, 1)

# %%

# Print the crosstalk matrix
print("Crosstalk Matrix:")
print(crosstalk_matrix)

# Plot the crosstalk matrix
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.matshow(crosstalk_matrix, cmap='viridis', vmax = max_off_diagonal*1.1, vmin = min_off_diagonal*0.9)
fig.colorbar(cax)

# Add text annotations with values
for i in range(num_qubits):
    for j in range(num_qubits):
        value = crosstalk_matrix[i, j]
        text_color = 'white' if value < min_off_diagonal + (max_off_diagonal - min_off_diagonal)/2 else 'black'
        ax.text(j, i, f'{value:.2f}', ha='center', va='center', color=text_color)

# Set axis labels
ax.set_xticks(np.arange(num_qubits))
ax.set_yticks(np.arange(num_qubits))
ax.set_xticklabels([f"Q{i+1}" for i in range(num_qubits)])
ax.set_yticklabels([f"Q{i+1}" for i in range(num_qubits)])
plt.xlabel('Qubit')
plt.ylabel('Qubit')
plt.title('Crosstalk Matrix')

plt.show()

# Save the crosstalk matrix and figure to the node results
node.results['crosstalk_matrix'] = crosstalk_matrix.tolist()
node.results['crosstalk_matrix_figure'] = fig


# %%
for q in qubits:
    with node.record_state_updates():
        pass
# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%


