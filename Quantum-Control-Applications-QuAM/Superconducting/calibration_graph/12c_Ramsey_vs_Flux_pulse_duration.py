"""
RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
    - Save the current state by calling machine.save("quam")
"""


# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state
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
    frequency_detuning_in_mhz: float = 4.0
    min_wait_pre_time_in_ns: int = 100
    max_wait_pre_time_in_ns: int = 30000
    wait_pre_time_step_in_ns: int = 1000    
    min_wait_time_in_ns: int = 50
    max_wait_time_in_ns: int = 1000
    wait_time_step_in_ns: int = 4
    flux_pulse_amp : float = 0.04
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = False
    net_zero: bool = False

node = QualibrationNode(name="12c_Ramsey_vs_Flux_pulse_duration", parameters=Parameters())


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
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

pre_idle_times = np.arange(
    node.parameters.min_wait_pre_time_in_ns // 4,
    node.parameters.max_wait_pre_time_in_ns // 4,
    node.parameters.wait_pre_time_step_in_ns // 4,
)

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    t = declare(int)  # QUA variable for the idle time
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    pre_time = declare(int) 
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            assign(init_state[i], 0)
            with for_(*from_array(pre_time, pre_idle_times)):
                with for_(*from_array(t, idle_times)):
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                    # TODO: this has gaps and the Z rotation is not derived properly, is it okay still?
                    # Ramsey sequence
                    qubit.align()

                    qubit.xy.wait(pre_time)
                    if node.parameters.net_zero:
                        qubit.z.play("const", amplitude_scale=-node.parameters.flux_pulse_amp / qubit.z.operations["const"].amplitude, 
                                        duration=pre_time /2 )
                        qubit.z.play("const", amplitude_scale=node.parameters.flux_pulse_amp / qubit.z.operations["const"].amplitude, 
                                        duration=pre_time/2 )
                    else:
                        qubit.z.play("const", amplitude_scale=node.parameters.flux_pulse_amp / qubit.z.operations["const"].amplitude, 
                                        duration=pre_time )

                    qubit.xy.play("x180", amplitude_scale=0.5)
                    qubit.xy.frame_rotation_2pi(phi)
                    qubit.z.wait(duration=qubit.xy.operations["x180"].length)
                    
                    qubit.xy.wait(t+1)
                    qubit.z.play("const", amplitude_scale=node.parameters.flux_pulse_amp / qubit.z.operations["const"].amplitude, 
                                    duration=t)
                    
                    qubit.xy.play("x180", amplitude_scale=0.5)

                    qubit.align()
                    # Measure the state of the resonators
                    readout_state(qubit, state[i])
                    assign(final_state[i], init_state[i] ^ state[i])
                    save(final_state[i], state_st[i])
                    assign(init_state[i], state[i])

                    # Reset the frame of the qubits in order not to accumulate rotations
                    reset_frame(qubit.xy.name)

        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(idle_times)).buffer(len(pre_idle_times)).average().save(f"state{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
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
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "pre_time": pre_idle_times})
        # Add the absolute time in µs to the dataset
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / 1e3)
        ds = ds.assign_coords(pre_time=4 * ds.pre_time / 1e3)
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
        ds.pre_time.attrs = {"long_name": "pre time", "units": "µs"}
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # TODO: explain the data analysis
    fit_data = fit_oscillation_decay_exp(ds.state, "idle_time")
    fit_data.attrs = {"long_name": "time", "units": "µs"}
    fitted = oscillation_decay_exp(
        ds.state.idle_time,
        fit_data.sel(fit_vals="a"),
        fit_data.sel(fit_vals="f"),
        fit_data.sel(fit_vals="phi"),
        fit_data.sel(fit_vals="offset"),
        fit_data.sel(fit_vals="decay"),
    )

    frequency = fit_data.sel(fit_vals="f")
    frequency.attrs = {"long_name": "frequency", "units": "MHz"}

    decay = fit_data.sel(fit_vals="decay")
    decay.attrs = {"long_name": "decay", "units": "nSec"}

    tau = 1 / fit_data.sel(fit_vals="decay")
    tau.attrs = {"long_name": "T2*", "units": "uSec"}

    frequency = frequency.where(frequency > 0, drop=True)

    # %% {Plotting}
    grid_names = [q.grid_location for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Idle_time (uS)")
        ax.set_ylabel("Pulse duration (us)")
    grid.fig.suptitle("Ramsey freq. Vs. flux")
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig

    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        frequency.sel(qubit=qubit["qubit"]).plot(marker=".", linewidth=0, ax=ax)
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Pulse duration (us)")
    grid.fig.suptitle("Ramsey freq. Vs. pulse duration")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}



    # %% {Save_results}
    if node.parameters.load_data_id is None:        
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        node.save()

# %%
