"""
        2D QUBIT SPECTROSCOPY WITH READOUT AMPLITUDE OPTIMIZATION
This sequence extends the standard qubit spectroscopy by adding a readout amplitude sweep.
It performs a 2D scan over qubit drive frequency and readout amplitude to find the optimal 
readout amplitude that gives the most pronounced spectroscopy signal.

The sequence involves:
1. Scanning over readout amplitudes and qubit drive frequencies
2. Finding the optimal readout amplitude that gives the strongest spectroscopy signal
3. Performing standard qubit spectroscopy analysis at the optimal amplitude
4. Updating both the qubit frequency and optimal readout amplitude in the state

Prerequisites:
    - Same as standard qubit spectroscopy
    - Additional consideration for readout power range to avoid any damage

Before proceeding to the next node:
    - Updates both qubit frequency and optimal readout amplitude in the state
    - Save the current state
"""

# %% {Imports}
from datetime import datetime, timezone, timedelta
from qualibrate import QualibrationNode, NodeParameters

from quam_libs.components import QuAM
from quam_libs.lib.instrument_limits import instrument_limits
from quam_libs.macros import qua_declaration
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 500
    operation: str = "saturation"
    operation_amplitude_factor: Optional[float] = 0.1
    operation_len_in_ns: Optional[int] = None
    frequency_span_in_mhz: float = 50
    frequency_step_in_mhz: float = 0.25
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    target_peak_width: Optional[float] = 2e6
    # New parameters for readout amplitude sweep
    readout_amp_start: float = 0.75  # Starting amplitude scaling factor
    readout_amp_stop: float = 1.5   # Ending amplitude scaling factor
    readout_amp_steps: int = 10     # Number of amplitude points
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True


node = QualibrationNode(name="03d_Qubit_Spectroscopy_vs_readout_amp", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
u = unit(coerce_to_integer=True)
machine = QuAM.load()
config = machine.generate_config()
if node.parameters.load_data_id is None:
    qmm = machine.connect()
    
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


# %% {QUA_program}
operation = node.parameters.operation
n_avg = node.parameters.num_averages
operation_len = node.parameters.operation_len_in_ns
operation_amp = node.parameters.operation_amplitude_factor if node.parameters.operation_amplitude_factor else 1.0

# Frequency sweep parameters
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, +span // 2, step, dtype=np.int32)

# Readout amplitude sweep parameters
ro_amps = np.linspace(
    node.parameters.readout_amp_start,
    node.parameters.readout_amp_stop,
    node.parameters.readout_amp_steps
)

flux_point = node.parameters.flux_point_joint_or_independent
qubit_freqs = {q.name: q.xy.RF_frequency for q in qubits}

# Set the qubit frequency for a given flux point. TODO : arb_flux_bias_offset. deprecated should be removed
arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
detunings = {q.name: 0.0 for q in qubits}

target_peak_width = node.parameters.target_peak_width or 3e6

with program() as qubit_spec_2d:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    df = declare(int)
    ro_amp = declare(fixed)

    if flux_point == "joint":
        # Bring the active qubits to the desired frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubits[0])
    
    for i, qubit in enumerate(qubits):
        # Set flux point
        if flux_point != "joint":
            machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(ro_amp, ro_amps)):
                with for_(*from_array(df, dfs)):
                    # Update qubit frequency
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detunings[qubit.name])
                    qubit.align()
                    
                    duration = operation_len * u.ns if operation_len is not None else (qubit.xy.operations[operation].length + qubit.z.settle_time) * u.ns
                    # Set flux bias
                    qubit.z.play("const", amplitude_scale=arb_flux_bias_offset[qubit.name] / qubit.z.operations["const"].amplitude, duration=duration)
                    # Play saturation pulse
                    qubit.xy.wait(qubit.z.settle_time * u.ns)
                    qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp,
                        duration=duration,
                    )
                    qubit.align()

                    # Readout with variable amplitude
                    qubit.resonator.measure(
                        "readout",
                        amplitude_scale=ro_amp,
                        qua_vars=(I[i], Q[i])
                    )
                    qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                    
                    save(I[i], I_st[i])
                    save(Q[i], Q_st[i])

        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            I_st[i].buffer(len(dfs)).buffer(len(ro_amps)).average().save(f"I{i + 1}")
            Q_st[i].buffer(len(dfs)).buffer(len(ro_amps)).average().save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)
    job = qmm.simulate(config, qubit_spec_2d, simulation_config)
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now(timezone(timedelta(hours=3))).strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(qubit_spec_2d)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_avg, start_time=results.start_time)

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # Fetch data and create dataset with both frequency and amplitude dimensions
        ds = fetch_results_as_xarray(
            job.result_handles,
            qubits,
            {"freq": dfs, "ro_amp": ro_amps}
        )
        ds = convert_IQ_to_V(ds, qubits)
        ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        ds = ds.assign({"phase": np.arctan2(ds.Q, ds.I)})
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + qubit_freqs[q.name] + detunings[q.name] for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"

    node.results = {"ds": ds}

    # %% {Data_analysis}
    # Find optimal readout amplitude for each qubit
    optimal_ro_amps = {}
    optimal_ro_amps_scaling = {}
    fit_results = {}
    
    # First pass - find optimal amplitudes
    for q in qubits:
        # Calculate contrast at each amplitude
        qubit_data = ds.sel(qubit=q.name)
        contrast = (qubit_data.IQ_abs.max(dim='freq') / qubit_data.IQ_abs.mean(dim='freq'))
        
        opt_amp_scale = ro_amps[contrast.argmax()]
        optimal_ro_amps_scaling[q.name] = opt_amp_scale
        current_ro_amp = q.resonator.operations["readout"].amplitude
        optimal_ro_amp = float(opt_amp_scale * current_ro_amp)
        optimal_ro_amps[q.name] = optimal_ro_amp
    
    # Create optimized dataset with only the optimal amplitude data
    optimal_slices = []
    for q in qubits:
        slice_data = ds.sel(qubit=q.name, ro_amp=optimal_ro_amps_scaling[q.name])
        # Calculate rotation angle for this qubit
        shift = np.abs((slice_data.IQ_abs - slice_data.IQ_abs.mean(dim='freq'))).idxmax(dim='freq')
        angle = np.arctan2(
            slice_data.sel(freq=shift).Q - slice_data.Q.mean(dim='freq'),
            slice_data.sel(freq=shift).I - slice_data.I.mean(dim='freq')
        )
        # Rotate the data
        I_rot = slice_data.I * np.cos(angle) + slice_data.Q * np.sin(angle)
        slice_data = slice_data.assign(I_rot=I_rot)
        optimal_slices.append(slice_data)
    
    optimized_ds = xr.concat(optimal_slices, dim='qubit')
    
    # Find peaks for all qubits
    result = peaks_dips(optimized_ds.I_rot, dim='freq', prominence_factor=5)
    
    # The resonant RF frequency of the qubits
    abs_freqs = dict(
        [
            (
                q.name,
                optimized_ds.freq_full.sel(freq=result.position.sel(qubit=q.name).values).sel(qubit=q.name).values,
            )
            for q in qubits if not np.isnan(result.sel(qubit=q.name).position.values)
        ]
    )

    for q in qubits:
        fit_results[q.name] = {}
        if not np.isnan(result.sel(qubit=q.name).position.values):
            fit_results[q.name]["fit_successful"] = True
            fit_results[q.name]["optimal_ro_amp"] = optimal_ro_amps[q.name]
            fit_results[q.name]["drive_freq"] = result.sel(qubit=q.name).position.values + q.xy.RF_frequency
            fit_results[q.name]["peak_width"] = float(result.sel(qubit=q.name).width.values)
            fit_results[q.name]["readout_angle"] = float(angle)
            
            # Calculate amplitude scaling factors
            Pi_length = q.xy.operations["x180"].length
            used_amp = q.xy.operations["saturation"].amplitude * operation_amp
            factor_cw = float(target_peak_width / result.sel(qubit=q.name).width.values)
            factor_pi = np.pi / (result.sel(qubit=q.name).width.values * Pi_length * 1e-9)
            
            fit_results[q.name]["factor_cw"] = factor_cw
            fit_results[q.name]["factor_pi"] = factor_pi
            
            print(f"\nResults for {q.name}:")
            print(f"Optimal readout amplitude: {optimal_ro_amps[q.name]:.3f}")
            print(f"Drive frequency: {fit_results[q.name]['drive_freq']/1e9:.6f} GHz")
            print(f"Peak width: {result.sel(qubit=q.name).width.values/1e6:.2f} MHz")
            print(f"Readout angle: {angle:.4f}")
        else:
            fit_results[q.name]["fit_successful"] = False
            print(f"\nFailed to find a peak for {q.name}")

    # %% {Plotting}
    # Create two separate figures using QubitGrid
    
    # Figure 1: 2D heatmaps
    grid_2d = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_2d):
        qubit_data = ds.sel(qubit=qubit["qubit"])
        im = ax.pcolormesh(
            qubit_data.freq_full/1e9,
            ro_amps,
            qubit_data.IQ_abs,
            shading='auto'
        )
        if fit_results[qubit["qubit"]]["fit_successful"]:
            ax.axhline(y=optimal_ro_amps_scaling[qubit["qubit"]], color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('Frequency (GHz)')
        ax.set_ylabel('Readout Amplitude Scale')
        ax.set_ylim(0.98*node.parameters.readout_amp_start, 1.02*node.parameters.readout_amp_stop)
        ax.set_title(f'{qubit["qubit"]}')
    
    grid_2d.fig.suptitle(f"Qubit Spectroscopy vs Readout Amplitude Optimization \n {date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Optimal slices
    grid_slice = QubitGrid(optimized_ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid_slice):
        if fit_results[qubit["qubit"]]["fit_successful"]:
            qubit_slice = optimized_ds.sel(qubit=qubit["qubit"])
            freq_ghz = qubit_slice.freq_full / 1e9
            ax.plot(freq_ghz, qubit_slice.I_rot * 1e3, 'b-', label='Data')
            
            # Plot Lorentzian fit
            qubit_result = result.sel(qubit=qubit["qubit"])
            freq_points = qubit_slice.freq
            approx_peak = qubit_result.base_line + qubit_result.amplitude / (1 + ((freq_points - qubit_result.position) / (qubit_result.width/2)) ** 2)
            
            ax.plot(freq_ghz, approx_peak * 1e3, 
                    linewidth=2, linestyle="--", color='r', label='fit')
            
            # Plot peak point
            peak_freq = abs_freqs[qubit["qubit"]] / 1e9
            peak_val = qubit_slice.I_rot.sel(freq=qubit_result.position) * 1e3
            ax.plot(peak_freq, peak_val, "or", markersize=5, label='Peak')
            
            ax.set_xlabel('Frequency (GHz)')
            ax.set_ylabel('Rotated I (mV)')
            ax.set_title(f'{qubit["qubit"]} (amp={optimal_ro_amps[qubit["qubit"]]:.3f})')
            ax.legend(fontsize=9, loc='upper right')
            
            # Set reasonable x-axis limits around the peak
            peak_idx = np.abs(freq_ghz - peak_freq).argmin()
            width_pts = int(len(freq_ghz) * 0.2)  # Show ~20% of the data around peak
            ax.set_xlim(freq_ghz[max(0, peak_idx - width_pts)], 
                        freq_ghz[min(len(freq_ghz)-1, peak_idx + width_pts)])
    
    grid_slice.fig.suptitle(f"Qubit Spectroscopy vs Readout Amplitude Optimization \n {date_time} GMT+3 #{node_id} \n multiplexed = {node.parameters.multiplexed}")
    plt.tight_layout()
    plt.show()
    
    # Save the figures to node results
    node.results["figure_optimal_amp"] = grid_2d.fig
    node.results["figure"] = grid_slice.fig
    
    # %% {Update_state}
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for q in qubits:
                if fit_results[q.name]["fit_successful"]:
                    # Update frequency
                    q.xy.intermediate_frequency += float(result.position.sel(qubit=q.name).values)

                    # Update readout amplitude and angle
                    q.resonator.operations["readout"].amplitude = optimal_ro_amps[q.name]
                    prev_angle = q.resonator.operations["readout"].integration_weights_angle
                    if not prev_angle:
                        prev_angle = 0.0
                    q.resonator.operations["readout"].integration_weights_angle = (
                        prev_angle + fit_results[q.name]["readout_angle"]
                    ) % (2 * np.pi)

                    # Update operation amplitudes
                    limits = instrument_limits(q.xy)
                    factor_cw = fit_results[q.name]["factor_cw"]
                    factor_pi = fit_results[q.name]["factor_pi"]
                    used_amp = q.xy.operations["saturation"].amplitude * operation_amp

                    if factor_cw * used_amp / operation_amp < limits.max_wf_amplitude:
                        q.xy.operations["saturation"].amplitude = factor_cw * used_amp / operation_amp
                    else:
                        q.xy.operations["saturation"].amplitude = limits.max_wf_amplitude

                    if factor_pi * used_amp < limits.max_x180_wf_amplitude:
                        q.xy.operations["x180"].amplitude = factor_pi * used_amp
                    elif factor_pi * used_amp >= limits.max_x180_wf_amplitude:
                        q.xy.operations["x180"].amplitude = limits.max_x180_wf_amplitude

        node.results["ds"] = ds
        node.results["fit_results"] = fit_results

        # %% {Save_results}
        node.outcomes = {q.name: "successful" if fit_results[q.name]["fit_successful"] else "failed" for q in qubits}
        node.results["initial_parameters"] = node.parameters.model_dump()
        node.machine = machine
        save_node(node)

# %% 