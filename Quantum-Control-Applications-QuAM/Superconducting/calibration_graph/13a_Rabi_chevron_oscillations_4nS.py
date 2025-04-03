"""
        RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing gates on resonance as opposed to the detuned Ramsey.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit spectroscopy, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency and T2_ramsey in the state.
    - Save the current state
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
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

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 250
    num_time_points: int = 100
    detuning_step_size_in_mhz: float = 4
    detuning_range_in_mhz: float = 100
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type_thermal_or_active: Literal["thermal", "active"] = "active"
    use_state_discrimination: bool = True
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(name="13a_Rabi_chevron_oscillations", parameters=Parameters())


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = (
    np.linspace(
        node.parameters.min_wait_time_in_ns,
        node.parameters.max_wait_time_in_ns,
        node.parameters.num_time_points,
    )
    // 4
).astype(int)


# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency

flux_point = node.parameters.flux_point_joint_or_independent
detuning_axis = np.arange(-node.parameters.detuning_range_in_mhz * 1e6 // 2, 
                         node.parameters.detuning_range_in_mhz * 1e6 // 2+1, 
                         node.parameters.detuning_step_size_in_mhz * 1e6,
                         dtype=np.int32)
with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the idle time
    detuning_ax = declare(int)
    sign = declare(int)  # QUA variable to change the sign of the detuning
    # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    phi = declare(fixed)
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_each_(detuning_ax, detuning_axis):
                qubit.xy.update_frequency(detuning_ax + qubit.xy.intermediate_frequency)
                with for_each_(t, idle_times):
                    if node.parameters.reset_type_thermal_or_active == "active":
                        active_reset(qubit, "readout")
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation

                    qubit.align()
                    # # Strict_timing ensures that the sequence will be played without gaps
                    # with strict_timing_():
                    qubit.xy.play("x180",duration=t)

                    # Align the elements to measure after playing the qubit pulse.
                    qubit.align()
                    # Measure the state of the resonators and save data
                    if node.parameters.use_state_discrimination:
                        readout_state(qubit, state[i])
                        save(state[i], state_st[i])
                    else:
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(idle_times)).buffer(len(detuning_axis)).average().save(f"state{i + 1}")
            else:
                I_st[i].buffer(len(idle_times)).buffer(len(detuning_axis)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(idle_times)).buffer(len(detuning_axis)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
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
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": idle_times, "detuning": detuning_axis})

        # Add the absolute time to the dataset
        ds = ds.assign_coords({"time": (["time"], 4 * idle_times)})
        ds.time.attrs["long_name"] = "idle_time"
        ds.time.attrs["units"] = "ns"
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}


    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        if node.parameters.use_state_discrimination:
            ds.loc[qubit].state.plot(
                ax=ax
            )
        else:
            (ds.loc[qubit].I * 1e3).plot(
                ax=ax
            )
        ax.set_xlim(node.parameters.min_wait_time_in_ns, node.parameters.max_wait_time_in_ns)
        ax.set_xlabel("Idle time [ns]")
        ax.set_title(qubit["qubit"])
    grid.fig.suptitle("Ramsey : I vs. idle time")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig


    # %% {Update_state}

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()

# %%
