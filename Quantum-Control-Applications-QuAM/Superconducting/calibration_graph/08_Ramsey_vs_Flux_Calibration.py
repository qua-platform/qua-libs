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
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, get_node_id, load_dataset, save_node
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
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 2000
    wait_time_step_in_ns: int = 20
    flux_span: float = 0.02
    flux_step: float = 0.001
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True

node = QualibrationNode(name="08_Ramsey_vs_Flux_Calibration", parameters=Parameters())
node_id = get_node_id()

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

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
fluxes = np.arange(
    -node.parameters.flux_span / 2, node.parameters.flux_span / 2 + 0.001, step=node.parameters.flux_step
)

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    t = declare(int)  # QUA variable for the idle time
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    flux = declare(fixed)  # QUA variable for the flux dc level

    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            assign(init_state[i], 0)
            with for_(*from_array(flux, fluxes)):
                with for_(*from_array(t, idle_times)):
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                    # TODO: this has gaps and the Z rotation is not derived properly, is it okay still?
                    # Ramsey sequence
                    qubit.align()
                    # with strict_timing_():
                    qubit.xy.play("x180", amplitude_scale=0.5)
                    qubit.xy.frame_rotation_2pi(phi)
                    qubit.z.wait(duration=qubit.xy.operations["x180"].length)
                    
                    qubit.xy.wait(t+1)
                    qubit.z.play("const", amplitude_scale=flux / qubit.z.operations["const"].amplitude, duration=t)
                    
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
            state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")


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
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
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
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux": fluxes})
        # Add the absolute time in µs to the dataset
        ds = ds.assign_coords(idle_time=4 * ds.idle_time / 1e3)
        ds.flux.attrs = {"long_name": "flux", "units": "V"}
        ds.idle_time.attrs = {"long_name": "idle time", "units": "µs"}
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

    fitvals = frequency.polyfit(dim="flux", deg=2)
    flux = frequency.flux
    a = {}
    flux_offset = {}
    freq_offset = {}
    for q in qubits:
        a[q.name] = float(-1e6 * fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values)
        flux_offset[q.name] = float(
            (
                -0.5
                * fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients
                / fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients
            ).values
        )
        freq_offset[q.name] = 1e6 * (flux_offset[q.name]**2 * float(fitvals.sel(qubit=q.name, degree=2).polyfit_coefficients.values) +
                                     flux_offset[q.name] * float(fitvals.sel(qubit=q.name, degree=1).polyfit_coefficients.values) + 
                                     float(fitvals.sel(qubit=q.name, degree=0).polyfit_coefficients.values)) - detuning

    # Save fitting results
    node.results["fit_results"] = {}
    for q in qubits:
        node.results["fit_results"][q.name] = {}
        node.results["fit_results"][q.name]["flux_offset"] = flux_offset[q.name]
        node.results["fit_results"][q.name]["freq_offset"] = freq_offset[q.name]
        node.results["fit_results"][q.name]["quad_term"] = a[q.name]

    # %% {Plotting}
    grid_names = [q.grid_location for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit=qubit["qubit"]).state.plot(ax=ax)
        ax.set_title(qubit["qubit"])
        ax.set_xlabel("Idle_time (uS)")
        ax.set_ylabel(" Flux (V)")
    grid.fig.suptitle(f"{date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig

    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        fitted_freq = (
            fitvals.sel(qubit=qubit["qubit"], degree=2).polyfit_coefficients * flux**2
            + fitvals.sel(qubit=qubit["qubit"], degree=1).polyfit_coefficients * flux
            + fitvals.sel(qubit=qubit["qubit"], degree=0).polyfit_coefficients
        )
        frequency.sel(qubit=qubit["qubit"]).plot(marker=".", linewidth=0, ax=ax)
        ax.plot(flux, fitted_freq)
        ax.set_title(qubit["qubit"])
        ax.set_xlabel(" Flux (V)")
        print(f"The quad term for {qubit['qubit']} is {a[qubit['qubit']]/1e9:.3f} GHz/V^2")
        print(f"Flux offset for {qubit['qubit']} is {flux_offset[qubit['qubit']]*1e3:.1f} mV")
        print(f"Freq offset for {qubit['qubit']} is {freq_offset[qubit['qubit']]/1e6:.3f} MHz")
        print()
    grid.fig.suptitle(f"{date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qubit in qubits:
                qubit.xy.intermediate_frequency -= freq_offset[qubit.name]
                if flux_point == "independent":
                    qubit.z.independent_offset += flux_offset[qubit.name]
                elif flux_point == "joint":
                    qubit.z.joint_offset += flux_offset[qubit.name]
                else:
                    raise RuntimeError(f"unknown flux_point")
                qubit.freq_vs_flux_01_quad_term = float(a[qubit.name])

        # %% {Save_results}
        node.outcomes = {q.name: "successful" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)

# %%
