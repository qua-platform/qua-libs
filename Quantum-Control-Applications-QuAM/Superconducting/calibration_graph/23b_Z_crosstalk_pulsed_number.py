# %%
"""

"""

# %% {Imports}
from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam.components import pulses
import copy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, save_node
from quam_libs.lib.fit import fit_oscillation, oscillation
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from qualang_tools.multi_user import qm_session
import xarray as xr

# %% {Node_parameters}

class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["qubitC1", "qubitC2", "qubitC3"]
    operation: Literal["z180"] = "z180"
    timeout: int = 100
    num_averages: int = 2500
    max_num_of_pulses: int = 1000
    num_of_pulses_step: int = 10
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    simulate: bool = False
    load_data_id: Optional[int] = None
   
node = QualibrationNode(name="23_Z_crosstalk_pulsed", parameters=Parameters())



# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}

n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

num_of_pulses = np.arange(
    2,
    node.parameters.max_num_of_pulses,
    node.parameters.num_of_pulses_step,
)

with program() as cross_talk_sequential:
    n = declare(int)
    n_st = declare_stream()
    num = declare(int)
    state = [declare(int) for _ in range(num_qubits * (num_qubits ))]
    state_stream = [declare_stream() for _ in range(num_qubits * (num_qubits ))]
    
    t = declare(int)  

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
            
        align()
        
        for j, qubit2 in enumerate(qubits):
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)            
                with for_(*from_array(num, num_of_pulses)):   
                    if node.parameters.reset_type_thermal_or_active == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    qubit.align()
                    qubit.xy.play("x90")
                    align()
                    with for_(t, 0, t < num, t + 1):
                        qubit2.z.wait(10)
                        qubit2.z.play(node.parameters.operation)
                        qubit2.z.wait(10)
                    align()
                    qubit.xy.play("x90")
                    align()
                    readout_state(qubit, state[i*num_qubits + j - 1])
                    save(state[i*num_qubits + j - 1], state_stream[i*num_qubits + j - 1])

            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            for j in range(num_qubits):
                state_stream[i*num_qubits + j-1].buffer(len(num_of_pulses)).average().save(f"state{j+1}_{i+1}")


# with program() as cross_talk_simultaneous:
#     n = declare(int)
#     n_st = declare_stream()
#     num = declare(int)
#     state = [declare(int) for _ in range(num_qubits * (num_qubits ))]
#     state_stream = [declare_stream() for _ in range(num_qubits * (num_qubits ))]
    
#     t = declare(int)  

#     for j, qubit2 in enumerate(qubits):
#         # Bring the active qubits to the minimum frequency point
#         machine.set_all_fluxes(flux_point=flux_point, target=qubit)
#         align()        
#         with for_(n, 0, n < n_avg, n + 1):
#             save(n, n_st)            
#             with for_(*from_array(num, num_of_pulses)):  
#                  for i, qubit in enumerate(qubits):
#                     if node.parameters.reset_type_thermal_or_active == "active":
#                         active_reset(qubit, "readout")
#                     else:
#                         qubit.wait(qubit.thermalization_time * u.ns)
#                     qubit.align()
#                     qubit.xy.play("x90")
#                     align()
#                     with for_(t, 0, t < num, t + 1):
#                         qubit2.z.wait(10)
#                         qubit2.z.play(node.parameters.operation)
#                         qubit2.z.wait(10)
#                     align()
#                     qubit.xy.play("x90")
#                     align()
#                     readout_state(qubit, state[i*num_qubits + j - 1])
#                     save(state[i*num_qubits + j - 1], state_stream[i*num_qubits + j - 1])

#             align()

#     with stream_processing():
#         n_st.save("n")
#         for i in range(num_qubits):
#             for j in range(num_qubits):
#                 state_stream[i*num_qubits + j-1].buffer(len(num_of_pulses)).average().save(f"state{j+1}_{i+1}")



# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, cross_talk_sequential, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i + 1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(cross_talk_sequential)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)



# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    handles = job.result_handles

    
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
            f"{meas_vars[0]}": (["qubit_pair", "repeats"], np.array(values).reshape(-1, len(num_of_pulses)))
        },
        coords={
            "repeats": num_of_pulses,
            "qubit_pair": [f"q{i+1}_q{j+1}" for i in range(len(qubits)) for j in range(len(qubits))]
        }
    )

    # %%

    fit = fit_oscillation_decay_exp(ds.state, 'repeats')
    fit_evals = oscillation_decay_exp(ds.repeats,fit.sel(fit_vals = 'a'),
                                    fit.sel(fit_vals = 'f'),fit.sel(fit_vals = 'phi'),
                                    fit.sel(fit_vals = 'offset'),fit.sel(fit_vals = 'decay'))

    # Add fit_evals to the dataset
    ds['fit'] = (('qubit_pair', 'repeats'), fit_evals.values)

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
            ax.set_xlabel("Repeats")
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
                if np.abs(amplitude) > 0.1:
                    crosstalk_matrix[i, j] = 0.5/frequency
                else:
                    crosstalk_matrix[i, j] = np.inf

    crosstalk_matrix
    # %%
    # Normalize the crosstalk matrix by the length of the xy.operations[operation] for each qubit
    for i in range(num_qubits):
        for j in range(num_qubits):
            if i != j:
                # Get the length of the operation for the control qubit (i)
                operation_amp_ratio = qubits[i].z.operations[node.parameters.operation].amplitude/qubits[j].z.operations[node.parameters.operation].amplitude
                
                # Normalize the crosstalk value

                crosstalk_matrix[i, j] = operation_amp_ratio / (crosstalk_matrix[i, j])
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
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=text_color)

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
    inv_crosstalk_matrix = np.linalg.inv(crosstalk_matrix)
    with node.record_state_updates():
        for i, qubit in enumerate(qubits):
            qubit.z.opx_output.crosstalk = {}
            for j, qubit2 in enumerate(qubits):
                if i != j:
                    
                    qubit.z.opx_output.crosstalk[qubit2.z.opx_output.port_id] = inv_crosstalk_matrix[j, i]

    # %%
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)
# %%
