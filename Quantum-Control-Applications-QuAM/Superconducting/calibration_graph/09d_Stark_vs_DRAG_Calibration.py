"""
        2D STARK DETUNING AND DRAG CALIBRATION (COMBINED METHOD)
This sequence performs a 2D scan over both detuning and DRAG parameters simultaneously.
The sequence consists in applying an increasing number of x180 and -x180 pulses successively 
for different combinations of DRAG detunings and DRAG coefficients.
After such a sequence, the qubit is expected to always be in the ground state if both the 
AC Stark shift is properly compensated by the DRAG detuning and the DRAG coefficient is optimal.

This protocol combines the methods described in:
https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.117.190503

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR and state discrimination.
    - Set the desired flux bias.

Next steps before going to the next node:
    - Update both the DRAG detuning and DRAG coefficient (alpha) in the state.
"""


# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, get_node_id, save_node
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


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['qC5']
    num_averages: int = 10
    operation: str = "x180"
    # Detuning parameters
    frequency_span_in_mhz: float = 10
    frequency_step_in_mhz: float = 0.2
    # DRAG parameters
    min_amp_factor: float = -1.99
    max_amp_factor: float = 1.99
    amp_factor_step: float = 0.1
    # General parameters
    max_number_pulses_per_sweep: int = 20
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "thermal"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    # Initial values for optimization
    initial_detuning_mhz: float = 0.0
    initial_alpha: float = 0.1
    # State discrimination
    use_state_discrimination: bool = True

node = QualibrationNode(name="09d_Stark_DRAG_2D_Calibration", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
operation = node.parameters.operation  # The qubit operation to play

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_qubits = []
for q in qubits:
    with tracked_updates(q, auto_revert=False) as q:
        q.xy.operations[operation].detuning = node.parameters.initial_detuning_mhz * 1e6  # Convert MHz to Hz
        q.xy.operations[operation].alpha = node.parameters.initial_alpha
        tracked_qubits.append(q)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
reset_type = node.parameters.reset_type_thermal_or_active  # "active" or "thermal"

# Pulse frequency sweep (detuning)
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)

# DRAG amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
amps = np.arange(
    node.parameters.min_amp_factor,
    node.parameters.max_amp_factor,
    node.parameters.amp_factor_step,
)

# Number of applied Rabi pulses sweep
N_pi = node.parameters.max_number_pulses_per_sweep  # Maximum number of qubit pulses
N_pi_vec = np.linspace(1, N_pi, N_pi).astype("int")

with program() as stark_drag_2d_calibration:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    if node.parameters.use_state_discrimination:
        state = [declare(bool) for _ in range(num_qubits)]
        state_stream = [declare_stream() for _ in range(num_qubits)]
    df = declare(int)  # QUA variable for the qubit drive detuning
    a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
    npi = declare(int)  # QUA variable for the number of qubit pulses
    count = declare(int)  # QUA variable for counting the qubit pulses

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(npi, N_pi_vec)):
                with for_(*from_array(df, dfs)):
                    with for_(*from_array(a, amps)):
                        # Initialize the qubits
                        if reset_type == "active":
                            active_reset(qubit, "readout")
                        else:
                            qubit.wait(qubit.thermalization_time * u.ns)

                        # Update the qubit frequency after initialization for active reset
                        update_frequency(qubit.xy.name, df + qubit.xy.intermediate_frequency)
                        
                        # Loop for error amplification (perform many qubit pulses)
                        with for_(count, 0, count < npi, count + 1):
                            if operation == "x180":
                                play(operation * amp(1, 0, 0, a), qubit.xy.name)
                                play(operation * amp(-1, 0, 0, -a), qubit.xy.name)
                            elif operation == "x90":
                                play(operation * amp(1, 0, 0, a), qubit.xy.name)
                                play(operation * amp(1, 0, 0, a), qubit.xy.name)
                                play(operation * amp(-1, 0, 0, -a), qubit.xy.name)
                                play(operation * amp(-1, 0, 0, -a), qubit.xy.name)

                        # Update the qubit frequency back to the resonance frequency for active reset
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
                        qubit.align()
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        # State discrimination
                        if node.parameters.use_state_discrimination:
                            assign(state[i], I[i] > qubit.resonator.operations["readout"].threshold)
                            save(state[i], state_stream[i])
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            if node.parameters.use_state_discrimination:
                state_stream[i].boolean_to_int().buffer(len(amps)).buffer(len(dfs)).buffer(N_pi).average().save(f"state{i + 1}")
            I_stream = I_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(N_pi).average().save(f"I{i + 1}")
            Q_stream = Q_st[i].buffer(len(amps)).buffer(len(dfs)).buffer(N_pi).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, stark_drag_2d_calibration, simulation_config)
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
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(stark_drag_2d_calibration)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amp": amps, "freq": dfs, "N": N_pi_vec})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Add the qubit pulse absolute alpha coefficient to the dataset
        ds = ds.assign_coords(
        {
            "alpha": (
                ["qubit", "amp"],
                np.array([q.xy.operations[operation].alpha * amps for q in qubits]),
            )
        }
        )
        # Calculate IQ magnitude if not using state discrimination
        if not node.parameters.use_state_discrimination:
            ds["IQ_abs"] = np.sqrt(ds.I**2 + ds.Q**2)
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        if not node.parameters.use_state_discrimination and "IQ_abs" not in ds:
            ds["IQ_abs"] = np.sqrt(ds.I**2 + ds.Q**2)
            
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Get the average along the number of pulses axis to identify the best parameters
    if node.parameters.use_state_discrimination:
        state_n = ds.state.mean(dim="N")
    else:
        # If not using state discrimination, we'll use the IQ magnitude
        # The qubit should be in the ground state when the parameters are optimal
        state_n = ds.IQ_abs.mean(dim="N")
    
    # Find the minimum state value (ground state) for each qubit
    fit_results = {}
    for qubit in qubits:
        qubit_data = state_n.sel(qubit=qubit.name)
        
        # Find the global minimum
        min_idx = np.unravel_index(qubit_data.argmin(), qubit_data.shape)
        
        # Get the corresponding alpha and detuning values directly from the dataset coordinates
        best_alpha = float(ds.alpha.sel(qubit=qubit.name).values[min_idx[1]])
        best_detuning = float(ds.freq.values[min_idx[0]])
        
        fit_results[qubit.name] = {
            "alpha": best_alpha,
            "detuning": best_detuning,
            "min_state_value": float(qubit_data.min().values)
        }
        
        print(f"Optimal parameters for {qubit.name}:")
        print(f"  DRAG coefficient (alpha): {best_alpha}")
        print(f"  Detuning: {best_detuning} Hz ({best_detuning/1e6:.3f} MHz)")
        print(f"  Minimum state value: {fit_results[qubit.name]['min_state_value']:.3f}")
    
    node.results["fit_results"] = fit_results

    # Revert the change done at the beginning of the node
    for qubit in tracked_qubits:
        qubit.revert_changes()
    
    # %% {Plotting}
    qubit_names_to_qubits = {q.name: q for q in qubits}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            qubit_data = ds.loc[qubit].state.mean(dim="N")
            plot_label = 'State'
        else:
            qubit_data = ds.loc[qubit].IQ_abs.mean(dim="N")
            plot_label = 'IQ magnitude'
        
        # Get the coordinate meshgrid for plotting
        freq_mesh, alpha_mesh = np.meshgrid(
            qubit_data.freq.values * 1e-6,  # Convert to MHz
            ds.alpha.sel(qubit=qubit["qubit"]).values
        )
        
        # Create 2D plot
        im = ax.contourf(
            freq_mesh,
            alpha_mesh,
            qubit_data.values.T,
            levels=20,
            cmap='viridis'
        )
        
        # Mark the optimal point
        best_freq = fit_results[qubit["qubit"]]["detuning"] * 1e-6  # Convert to MHz
        best_alpha = fit_results[qubit["qubit"]]["alpha"]
        ax.plot(best_freq, best_alpha, 'r*', markersize=10, label='Optimal')
        
        # Mark the current point
        qb = qubit_names_to_qubits[qubit["qubit"]]
        current_freq = qb.xy.operations[operation].detuning * 1e-6  # Convert to MHz
        current_alpha = qb.xy.operations[operation].alpha
        ax.plot(current_freq, current_alpha, 'g*', markersize=10, label='Current')
        
        ax.set_xlabel("Detuning [MHz]")
        ax.set_ylabel(r"DRAG coeff $\alpha$")
        ax.set_title(f"{qubit['qubit']}")
        ax.legend()
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label=plot_label)
    
    grid.fig.suptitle(f"2D Stark-DRAG Calibration\n{date_time} GMT+3 #{node_id}\nmultiplexed = {node.parameters.multiplexed} reset Type = {node.parameters.reset_type_thermal_or_active}\nState Discrimination = {node.parameters.use_state_discrimination}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    
    # Update the state with optimal parameters
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.xy.operations[operation].detuning = fit_results[qubit.name]["detuning"]
                qubit.xy.operations[operation].alpha = fit_results[qubit.name]["alpha"]

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    save_node(node) 
# %%
