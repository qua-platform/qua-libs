"""
        RESONATOR SPECTROSCOPY VERSUS READOUT AMPLITUDE
This sequence involves measuring the resonator by sending a readout pulse and demodulating the signals to
extract the 'I' and 'Q' quadratures for all resonators simultaneously.
This is done across various readout intermediate dfs and amplitudes.
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
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.lib.fit_utils import fit_resonator
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V, subtract_slope, apply_angle
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
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
    max_power_dbm: int = -30
    min_power_dbm: int = -50
    num_power_points: int = 100
    max_amp: float = 0.1
    flux_point_joint_or_independent: Literal["joint", "independent"] = "independent"
    ro_line_attenuation_dB: float = 0
    derivative_crossing_threshold_in_hz_per_dbm: int = int(-50e3)
    derivative_smoothing_window_num_points: int = 30
    moving_average_filter_window_num_points: int = 30
    multiplexed: bool = False
    load_data_id: Optional[int] = None

node = QualibrationNode(name="02c_Resonator_Spectroscopy_vs_Amplitude", parameters=Parameters())


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

# Update the readout power to match the desired range, this change will be reverted at the end of the node.
tracked_resonators = []
for i, qubit in enumerate(qubits):
    with tracked_updates(qubit.resonator, auto_revert=False, dont_assign_to_none=True) as resonator:
        resonator.set_output_power(
            power_in_dbm=node.parameters.max_power_dbm,
            max_amplitude=node.parameters.max_amp
        )
        tracked_resonators.append(resonator)

# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# The readout amplitude sweep (as a pre-factor of the readout amplitude) - must be within [-2; 2)
amp_min = resonators[0].calculate_voltage_scaling_factor(
    fixed_power_dBm=node.parameters.max_power_dbm,
    target_power_dBm=node.parameters.min_power_dbm
)
amp_max = 1

amps = np.geomspace(amp_min, amp_max, node.parameters.num_power_points)

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

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)
        
        # resonator of this qubit
        rr = qubit.resonator

        with for_(n, 0, n < n_avg, n + 1):  # QUA for_ loop for averaging
            save(n, n_st)

            with for_(*from_array(df, dfs)):  # QUA for_ loop for sweeping the frequency
                # Update the resonator frequencies for all resonators
                update_frequency(rr.name, df + rr.intermediate_frequency)
                rr.wait(machine.depletion_time * u.ns)
                # QUA for_ loop for sweeping the readout amplitude
                with for_(*from_array(a, amps)):
                    # readout the resonator
                    rr.measure("readout", qua_vars=(I[i], Q[i]), amplitude_scale=a)
                    # wait for the resonator to relax
                    rr.wait(machine.depletion_time * u.ns)
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
        ds, machine, json_data, qubits, node.parameters = load_dataset(node.parameters.load_data_id, parameters = node.parameters)
    else:
        power_dbm = np.linspace(
            node.parameters.min_power_dbm,
            node.parameters.max_power_dbm,
            node.parameters.num_power_points
        ) - node.parameters.ro_line_attenuation_dB
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"power_dbm": power_dbm, "freq": dfs})
        # Convert IQ data into volts
        ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": subtract_slope(apply_angle(ds.I + 1j * ds.Q, dim="freq"), dim="freq")})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        RF_freq = np.array([dfs + q.resonator.RF_frequency for q in qubits])
        ds = ds.assign_coords({"freq_full": (["qubit", "freq"], RF_freq)})
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
        ds.power_dbm.attrs["long_name"] = "Power"
        ds.power_dbm.attrs["units"] = "dBm"

        # Normalize the IQ_abs with respect to the amplitude axis
        ds = ds.assign({"IQ_abs_norm": ds["IQ_abs"] / ds.IQ_abs.mean(dim=["freq"])})

    # Add the dataset to the node
    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Generate 1D dataset tracking the minimum IQ value, as a proxy for resonator frequency
    ds["rr_min_response"] = ds.IQ_abs_norm.idxmin(dim="freq")
    rr_min_response = ds.IQ_abs_norm.idxmin(dim="freq")
    # Calculate the derivative along the power_dbm axis
    ds["rr_min_response_diff"] = ds.rr_min_response.differentiate(coord="power_dbm").dropna("power_dbm")
    # Calculate the moving average of the derivative
    ds["rr_min_response_diff_avg"] = ds.rr_min_response_diff.rolling(
        power_dbm=node.parameters.derivative_smoothing_window_num_points,  # window size in points
        center=True
    ).mean().dropna("power_dbm")
    # Apply a filter to scale down the initial noisy values in the moving average if needed
    for j in range(node.parameters.moving_average_filter_window_num_points):
        ds.rr_min_response_diff_avg.isel(power_dbm=j).data /= (node.parameters.moving_average_filter_window_num_points - j)
    # Find the first position where the moving average crosses below the threshold
    below_threshold = ds.rr_min_response_diff_avg < node.parameters.derivative_crossing_threshold_in_hz_per_dbm
    # Get the first occurrence below the derivative threshold
    rr_optimal_power_dbm = {}
    rr_optimal_frequencies = {}
    for qubit in qubits:
        if below_threshold.sel(qubit=qubit.name).any():
            rr_optimal_power_dbm[qubit.name] = below_threshold.sel(qubit=qubit.name).idxmax(dim="power_dbm")  # Get the first occurrence
        else:
            rr_optimal_power_dbm[qubit.name] = np.nan

        if not np.isnan(rr_optimal_power_dbm[qubit.name]):
            fit, fit_eval = fit_resonator(
                s21_data=ds.sel(power_dbm=rr_optimal_power_dbm[qubit.name].data).sel(qubit=qubit.name),
                frequency_LO_IF=qubit.resonator.RF_frequency,
                print_report=True
            )
            rr_optimal_frequencies[qubit.name] = int(fit.params["omega_r"].value)
        else:
            rr_optimal_frequencies[qubit.name] = np.nan


    # %% {Plotting}
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        # Plot the data using the secondary y-axis
        ds.loc[qubit].IQ_abs_norm.plot(
            ax=ax,
            add_colorbar=False,
            x="freq_full",
            y="power_dbm",
            robust=True,
        )
        ax.set_ylabel("Power (dBm)")
        # Plot the resonance frequency  for each amplitude
        ax.plot(
            ds.rr_min_response.loc[qubit],
            ds.power_dbm,
            color="orange",
            linewidth=0.5,
        )
        # Plot where the optimum readout power was found
        if not np.isnan(rr_optimal_power_dbm[qubit['qubit']]):
            ax.axhline(
                y=rr_optimal_power_dbm[qubit['qubit']],
                color="r",
                linestyle="--",
            )
        if not np.isnan(rr_optimal_frequencies[qubit['qubit']]):
            ax.axvline(
                x=rr_optimal_frequencies[qubit['qubit']] + machine.qubits[qubit['qubit']].resonator.RF_frequency,
                color="blue",
                linestyle="--",
            )

    grid.fig.suptitle("Resonator spectroscopy VS. power at base")
    plt.tight_layout()
    plt.show()
    node.results["figure"] = grid.fig

    # %% {Update_state}
    # Revert the change done at the beginning of the node
    for tracked_resonator in tracked_resonators:
        tracked_resonator.revert_changes()

    # Save fitting results
    fit_results = {}
    for q in qubits:
        fit_results[q.name] = {}
        if not node.parameters.load_data_id:
            with node.record_state_updates():
                if not np.isnan(rr_optimal_power_dbm[q.name]):
                    power_settings = q.resonator.set_output_power(
                        power_in_dbm=rr_optimal_power_dbm[q.name].item(),
                        max_amplitude=0.1
                    )
                if not np.isnan(rr_optimal_frequencies[q.name]):
                    q.resonator.intermediate_frequency += rr_optimal_frequencies[q.name]
        fit_results[q.name] = power_settings
        fit_results[q.name]["RO_frequency"] = q.resonator.RF_frequency
    node.results["fit_results"] = fit_results

    # %% {Save_results}
    if node.parameters.load_data_id is not None:
        if node.storage_manager is not None:
            node.storage_manager.active_machine_path = None
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()


# %%
