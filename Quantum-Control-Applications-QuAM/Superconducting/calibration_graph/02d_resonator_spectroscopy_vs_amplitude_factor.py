"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE FACTOR
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitude factors.
Based on the results, one can determine if a qubit is coupled to the resonator by noting the resonator frequency
splitting. This information can then be used to adjust the readout amplitude, choosing a readout amplitude value
just before the observed frequency splitting.

Prerequisites:
    - Calibration of the time of flight, offsets, and gains (referenced as "time_of_flight").
    - Calibration of the IQ mixer connected to the readout line (be it an external mixer or an Octave port).
    - Identification of the resonator's resonance frequency (referred to as "resonator_spectroscopy").
    - Configuration of the readout pulse amplitude (the pulse processor will sweep up to twice this value) and duration.
    - Specification of the expected resonator depletion time in the state.

Before proceeding to the next node:
    - Update the readout frequency, in the state.
    - Adjust the readout amplitude, in the state.
    - Save the current state
"""


# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V, subtract_slope, apply_angle
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
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

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_span_in_mhz: float = 15
    frequency_step_in_mhz: float = 0.1
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    initial_readout_amp: float = 0.05
    min_amp_factor: float = 0.1
    max_amp_factor: float = 1.99
    num_power_points: int = 100
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    derivative_crossing_threshold_in_hz_per_amp: int = int(-50e3)
    derivative_smoothing_window_num_points: int = 30
    moving_average_filter_window_num_points: int = 30
    multiplexed: bool = True # We get compilation error for this true and 15 qubits (5 OK)
    load_data_id: Optional[int] = None

node = QualibrationNode(name="02d_Resonator_Spectroscopy_vs_Amplitude_Factor", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
resonators = [qubit.resonator for qubit in qubits]
prev_amps = [rr.operations["readout"].amplitude for rr in resonators]
num_qubits = len(qubits)

# Generate the OPX and Octave configurations
config = machine.generate_config()

tracked_qubits = []
if node.parameters.load_data_id is None:
    for q in qubits:
        with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
            q.resonator.operations.readout.amplitude = node.parameters.initial_readout_amp
            tracked_qubits.append(q)

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amp_min = node.parameters.min_amp_factor
amp_max = node.parameters.max_amp_factor

amps = np.linspace(amp_min, amp_max, node.parameters.num_power_points)

# The frequency sweep around the resonator resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span / 2, +span / 2, step)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

with program() as multi_res_spec_vs_amp:
    # Declare 'I' and 'Q' and the corresponding streams for the two resonators.
    # For instance, here 'I' is a python list containing two QUA fixed variables.
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    a = declare(fixed)  # QUA variable for the readout amplitude pre-factor
    df = declare(int)  # QUA variable for the readout frequency

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
        
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        # resonator of this qubit
        rr = qubit.resonator

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)

            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                rr.wait(qubit.resonator.depletion_time * u.ns)
                # QUA for_ loop for sweeping the readout amplitude
                with for_(*from_array(a, amps)):
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                    # wait for the resonator to relax
                    rr.wait(qubit.resonator.depletion_time * u.ns)
                    # save data
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(amps)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_res_spec_vs_amp, simulation_config)
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
        job = qm.execute(multi_res_spec_vs_amp)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # ds = fetch_results_as_xarray(job.result_handles, qubits, {"amps": amps, "freq": dfs})
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"amps": amps, "freq": dfs})
        # Convert IQ data into volts
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        ds.amps.attrs["long_name"] = "Amplitude Factor"
        ds.amps.attrs["units"] = ""

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Generate 1D dataset tracking the minimum IQ value, as a proxy for resonator frequency
    ds["rr_min_response"] = ds.IQ_abs.idxmin(dim="freq")
    rr_min_response = ds.IQ_abs.idxmin(dim="freq")
    # Calculate the derivative along the amps axis
    ds["rr_min_response_diff"] = ds.rr_min_response.differentiate(coord="amps").dropna("amps")
    # Calculate the moving average of the derivative
    ds["rr_min_response_diff_avg"] = ds.rr_min_response_diff.rolling(
        amps=node.parameters.derivative_smoothing_window_num_points,  # window size in points
        center=True
    ).mean().dropna("amps")
    # Apply a filter to scale down the initial noisy values in the moving average if needed
    for j in range(node.parameters.moving_average_filter_window_num_points):
        ds.rr_min_response_diff_avg.isel(amps=j).data /= (node.parameters.moving_average_filter_window_num_points - j)
    # Find the first position where the moving average crosses below the threshold
    below_threshold = ds.rr_min_response_diff_avg < node.parameters.derivative_crossing_threshold_in_hz_per_amp
    # Get the first occurrence below the derivative threshold
    rr_optimal_amps = dict()  # Initialize as empty dictionary
    rr_optimal_amps_absolute = dict()  # Initialize as empty dictionary for absolute amplitudes
    rr_optimal_frequencies = dict()  # Initialize as empty dictionary
    for qubit in qubits:
        if below_threshold.sel(qubit=qubit.name).any():
            optimal_amp = below_threshold.sel(qubit=qubit.name).idxmax(dim="amps")
            rr_optimal_amps[qubit.name] = float(optimal_amp.values)  # Convert to float
            # Calculate absolute amplitude by multiplying factor with current readout amplitude
            current_readout_amp = qubit.resonator.operations.readout.amplitude
            rr_optimal_amps_absolute[qubit.name] = rr_optimal_amps[qubit.name] * current_readout_amp
            print(f"Optimal amplitude factor for {qubit.name}: {rr_optimal_amps[qubit.name]}")
            print(f"Absolute optimal amplitude for {qubit.name}: {rr_optimal_amps_absolute[qubit.name]} V")
        else:
            rr_optimal_amps[qubit.name] = np.nan
            rr_optimal_amps_absolute[qubit.name] = np.nan

        if not np.isnan(rr_optimal_amps[qubit.name]):
            fit, fit_eval = fit_resonator(
                s21_data=ds.sel(amps=rr_optimal_amps[qubit.name]).sel(qubit=qubit.name),
                frequency_LO_IF=qubit.resonator.RF_frequency,
                print_report=True
            )
            rr_optimal_frequencies[qubit.name] = int(fit.params["omega_r"].value)
        else:
            rr_optimal_frequencies[qubit.name] = np.nan

    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        # Plot the data
        ds.loc[qubit].IQ_abs.plot(
            ax=ax,
            add_colorbar=False,
            x="freq_full",
            y="amps",
            robust=True,
        )
        ax.set_ylabel("Amplitude Factor")
        
        # Create secondary y-axis for absolute amplitude values
        ax2 = ax.twinx()
        # Get the readout amplitude for this qubit
        readout_amp = machine.qubits[qubit['qubit']].resonator.operations.readout.amplitude
        # Set the secondary y-axis limits and ticks
        ax2.set_ylim(amps[0] * readout_amp, amps[-1] * readout_amp)
        ax2.set_ylabel(f"Absolute Amplitude (V)")
        
        # Plot the resonance frequency for each amplitude
        ax.plot(
            ds.rr_min_response.loc[qubit],
            ds.amps,
            color="orange",
            linewidth=0.5,
        )
        
        # Plot where the optimum readout amplitude was found
        if not np.isnan(rr_optimal_amps[qubit['qubit']]):
            ax.axhline(
                y=rr_optimal_amps[qubit['qubit']],
                color="r",
                linestyle="--",
            )
        if not np.isnan(rr_optimal_frequencies[qubit['qubit']]):
            ax.axvline(
                x=rr_optimal_frequencies[qubit['qubit']] + machine.qubits[qubit['qubit']].resonator.RF_frequency,
                color="blue",
                linestyle="--",
            )
        
        # Add optimal amplitude to subplot title
        current_title = ax.get_title()
        if not np.isnan(rr_optimal_amps_absolute[qubit['qubit']]):
            optimal_amp_str = f" (Opt: {rr_optimal_amps_absolute[qubit['qubit']]:.3f})"
            ax.set_title(current_title + optimal_amp_str)
    
    grid.fig.suptitle(f"Resonator spectroscopy VS. amplitude factor at base \n {date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    
    # Revert the change done at the beginning of the node
    for qubit in tracked_qubits:
        qubit.revert_changes()
    
    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        
        if not node.parameters.load_data_id:
            with node.record_state_updates():
                if not np.isnan(rr_optimal_amps_absolute[q.name]):
                    q.resonator.operations.readout.amplitude = rr_optimal_amps_absolute[q.name]
                if not np.isnan(rr_optimal_frequencies[q.name]):
                    q.resonator.intermediate_frequency += rr_optimal_frequencies[q.name]
                    
        fit_results[q.name]["RO_frequency"] = q.resonator.RF_frequency
        fit_results[q.name]["optimal_amplitude_factor"] = rr_optimal_amps[q.name]
        fit_results[q.name]["optimal_amplitude_absolute"] = rr_optimal_amps_absolute[q.name]
    node.results["fit_results"] = fit_results

    # %% {Save_results}
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)


# %% 