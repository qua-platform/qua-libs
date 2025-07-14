"""
Qubit Spectroscopy vs Flux Time - Long Time Cryoscope
===================================================

This sequence involves doing a qubit spectroscopy as a function of time after a flux pulse. 
The instantaneous frequency shift is used to extract the actual flux seen by the qubit as a function of time.
The flux response is then fitted to a single or double exponential decay.
The experiment is based on the description at https://arxiv.org/pdf/2503.04610.

Key Features:
------------
- Measures qubit frequency shift over time after flux pulse application
- Supports both single and double exponential decay fitting
- Provides visualization of raw data, frequency shifts, and flux response
- Updates exponential filter parameters in the system state

Prerequisites:
-------------
- Rough calibration of a pi-pulse, preferably with Gaussian envelope ("Power_Rabi_general_operation").
- Calibration of XY-Z delay.
- Identification of the approximate qubit frequency ("qubit_spectroscopy").
- The quadratic dependence of the qubit frequency on the flux ("Ramsey_vs_flux").

Before proceeding to the next node:
    - Update the exponential filter in the state.
"""

# %% {Imports}
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Any, Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import time
import quam_libs.lib.cryoscope_tools as cryoscope_tools
start = time.time()
# %% {Node_parameters}
class Parameters(NodeParameters):
    """Configuration parameters for the spectroscopy experiment.
    
    Attributes:
        qubits (List[str]): List of qubits to measure
        num_averages (int): Number of measurement averages
        operation (str): Qubit operation to perform (default: "x180_Gaussian")
        operation_amplitude_factor (float): Scaling factor for operation amplitude
        duration_in_ns (int): Total measurement duration in nanoseconds
        frequency_span_in_mhz (float): Frequency sweep range in MHz
        frequency_step_in_mhz (float): Frequency step size in MHz
        flux_amp (float): Amplitude of flux pulse
        update_lo (bool): Whether to update local oscillator frequency
        fit_single_exponential (bool): Use single vs double exponential fit
        update_state (bool): Update system state with fit results
        flux_point_joint_or_independent (str): Flux point handling method
        simulate (bool): Run in simulation mode
        simulation_duration_ns (int): Simulation duration
        timeout (int): Operation timeout in seconds
        load_data_id (int): ID of data to load (optional)
        multiplexed (bool): Use multiplexed vs sequential measurement
        reset_type_active_or_thermal (str): Reset method to use
    """

    qubits: Optional[List[str]] = None
    num_averages: int = 10
    operation: str = "x180_Gaussian"
    operation_amplitude_factor: Optional[float] = 1
    duration_in_ns: Optional[int] = 5000
    time_axis: Literal["linear", "log"] = "linear"
    time_step_in_ns: Optional[int] = 48 # for linear time axis
    time_step_num: Optional[int] = 200 # for log time axis
    frequency_span_in_mhz: float = 150
    frequency_step_in_mhz: float = 0.45
    flux_amp : float = 0.06
    update_lo: bool = True
    fitting_base_fractions: List[float] = [0.4, 0.15, 0.07] # fraction of times from which to fit each exponential
    update_state: bool = False
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'


node = QualibrationNode(name="97b_Pi_vs_flux_time", parameters=Parameters())
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

# Modify the lo frequency to allow for maximum detuning 
tracked_qubits = []
if node.parameters.update_lo:
    for q in qubits:
        with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
            lo_band: Any = q.xy.opx_output.band
            rf_frequency = q.xy.intermediate_frequency + q.xy.opx_output.upconverter_frequency
            lo_frequency = rf_frequency - 400e6
            if (lo_band == 3) and (lo_frequency < 6.5e9):
                lo_frequency = 6.5e9
            elif (lo_band == 2) and (lo_frequency < 4.5e9):
                lo_frequency = 4.5e9
            
            q.xy.intermediate_frequency = rf_frequency - lo_frequency
            q.xy.opx_output.upconverter_frequency = lo_frequency
            tracked_qubits.append(q)

# Generate the OPX and Octave configurations
config = machine.generate_config()

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
operation = node.parameters.operation  # The qubit operation to play
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
# Flux bias sweep
if node.parameters.time_axis == "linear":
    times = np.arange(4, node.parameters.duration_in_ns // 4, node.parameters.time_step_in_ns // 4, dtype=np.int32)
elif node.parameters.time_axis == "log":
    times = np.logspace(np.log10(4), np.log10(node.parameters.duration_in_ns // 4), node.parameters.time_step_num, dtype=np.int32)
    # Remove repetitions from times
    times = np.unique(times)

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
detuning = [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    df = declare(int)  # QUA variable for frequency scan
    t_delay = declare(int)  # QUA variable for delay time scan
    duration = node.parameters.duration_in_ns * u.ns
    
    if flux_point == "joint":
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                
                # with for_(*from_array(t_delay, times)):
                with for_each_(t_delay, times):
                    if node.parameters.reset_type_active_or_thermal == "active":
                        active_reset(qubit)
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)                    # Flux sweeping for a qubit
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detuning[i])
                    # Bring the qubit to the desired point during the saturation pulse
                    qubit.align()
                    qubit.z.play("const", amplitude_scale=node.parameters.flux_amp / qubit.z.operations["const"].amplitude, duration=t_delay+200)
                    # Apply saturation pulse to all qubits
                    # qubit.xy.wait(qubit.z.settle_time * u.ns)
                    qubit.xy.wait(t_delay)
                    qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp
                    )
                    # qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    qubit.align()
                    qubit.wait(200)
                    # QUA macro to read the state of the active resonators
                    readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                    
                    # Wait for the resonator to deplete of photons
                    qubit.resonator.wait(machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i+1)
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
        job = qm.execute(multi_qubit_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        # Load previously saved data and configuration
        node = node.load_from_id(node.parameters.load_data_id)
        machine = node.machine
        ds = xr.Dataset({"state": node.results["ds"].state})
        times = ds.time.values
        qubits = [machine.qubits[q] for q in ds.qubit.values]
    else:
        # Fetch new data and create dataset with proper coordinates
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": times*4, "freq": dfs})
        ds = ds.assign_coords(
            {
                "freq_full": (  # Full frequency including RF and flux-induced shifts
                    ["qubit", "freq"],
                    np.array([dfs + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term*node.parameters.flux_amp**2 for q in qubits]),
                ),
                "detuning": (  # Frequency shift due to flux
                    ["qubit", "freq"],
                    np.array([dfs + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]),
                ),
                "flux": (  # Applied flux values
                    ["qubit", "freq"],
                    np.array([np.sqrt(dfs / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp**2) for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"

    node.results = {"ds": ds}
    end = time.time()
    print(f"Script runtime: {end - start:.2f} seconds")

# %%  {Data_analysis}
# Extract frequency points and reshape data for analysis
freqs = ds['freq'].values

# Transpose to ensure ('qubit', 'time', 'freq') order for analysis
stacked = ds.transpose('qubit', 'time', 'freq')

# Fit Gaussian to each spectrum to find center frequencies
center_freqs = xr.apply_ufunc(
    lambda states: cryoscope_tools.fit_gaussian(freqs, states),
    stacked,
    input_core_dims=[['freq']],
    output_core_dims=[[]],  # no dimensions left after fitting
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
).rename({"state": "center_frequency"})

# Add flux-induced frequency shift to center frequencies
center_freqs = center_freqs.center_frequency + np.array([q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 * np.ones_like(times) for q in qubits])

# Calculate flux response from frequency shifts
flux_response = np.sqrt(center_freqs / xr.DataArray([q.freq_vs_flux_01_quad_term for q in qubits], coords={"qubit": center_freqs.qubit}, dims=["qubit"]))

# Store results in dataset
ds['center_freqs'] = center_freqs
ds['flux_response'] = flux_response

# Perform exponential fitting for each qubit
fit_results = {}
for q in qubits:
    fit_results[q.name] = {}
    t_data = flux_response.sel(qubit=q.name).time.values
    y_data = flux_response.sel(qubit=q.name).values
    fit_successful, best_fractions, best_components, best_a_dc, best_rms = cryoscope_tools.optimize_start_fractions(
        t_data, y_data, node.parameters.fitting_base_fractions, bounds_scale=0.5
        )

    fit_results[q.name]["fit_successful"] = fit_successful
    fit_results[q.name]["best_fractions"] = best_fractions
    fit_results[q.name]["best_components"] = best_components
    fit_results[q.name]["best_a_dc"] = best_a_dc
    fit_results[q.name]["best_rms"] = best_rms

node.results["fit_results"] = fit_results

# %% {Plotting}
# Create grid for raw spectroscopy data plots
grid = QubitGrid(ds, [q.grid_location for q in qubits])

# Plot raw spectroscopy data for each qubit
for ax, qubit in grid_iter(grid):
    # Plot state vs frequency and time
    im = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
        ax=ax, add_colorbar=False, x="time", y="freq_GHz"
    )
    ax.set_ylabel("Freq (GHz)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    # Add colorbar showing qubit state
    cbar = grid.fig.colorbar(im, ax=ax)
    cbar.set_label("Qubit State")
grid.fig.suptitle(f"Qubit spectroscopy vs time after flux pulse \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_raw"] = grid.fig

# Create grid for frequency shift plots
grid = QubitGrid(ds, [q.grid_location for q in qubits])

# Plot frequency shifts over time for each qubit
for ax, qubit in grid_iter(grid):
    # Plot center frequency vs time
    (ds.loc[qubit].center_freqs / 1e9).plot(ax=ax)
    ax.set_ylabel("Freq (GHz)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    # ax.set_xscale('log')
grid.fig.suptitle(f"Qubit frequency shift vs time after flux pulse \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_freqs_shift"] = grid.fig

# Create grid for flux response plots
grid = QubitGrid(ds, [q.grid_location for q in qubits])

# Plot flux response and fitted curves for each qubit
for ax, qubit in grid_iter(grid):
    # Plot measured flux response
    ds.loc[qubit].flux_response.plot(ax=ax)
    # flux_response_norm = ds.loc[qubit].flux_response / ds.loc[qubit].flux_response.values[-1]
    # flux_response_norm.plot(ax=ax)
    
    # Plot fitted curves and parameters if fits were successful    
    if fit_results[qubit["qubit"]]["fit_successful"]:
        best_a_dc = fit_results[qubit["qubit"]]["best_a_dc"]
        t_offset = t_data - t_data[0]
        y_fit = np.ones_like(t_data, dtype=float) * best_a_dc  # Start with fitted constant
        fit_text = f'a_dc = {best_a_dc:.3f}\n'
        for i, (amp, tau) in enumerate(fit_results[qubit["qubit"]]["best_components"]):
            y_fit += amp * np.exp(-t_offset/tau)
            fit_text += f'a{i+1} = {amp / best_a_dc:.3f}, τ{i+1} = {tau:.0f}ns\n'

        ax.plot(t_data, y_fit, color='r', label='Full Fit', linewidth=2) # Plot full fit
        ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=8)

    ax.set_ylabel("Flux (V)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    # ax.set_xscale('log')
grid.fig.suptitle(f"Flux response vs time \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_flux_response"] = grid.fig

# %% {Update_state}
# Revert any temporary changes to tracked qubits
for q in tracked_qubits:
    q.revert_changes()

if node.parameters.load_data_id is None:
    if node.parameters.update_state:
        with node.record_state_updates():
            for q in qubits:
                if fit_results[q.name]["fit_successful"]:
                    q.z.opx_output.exponential_filter = [[amp / best_a_dc, tau] 
                        for amp, tau in fit_results[q.name]["best_components"]
                        ]
                    print("updated the exponential filter")

# %% {Save_results}
# Store final results and metadata
node.results["ds"] = ds
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
save_node(node)