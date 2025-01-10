# %%
"""
Calibration of the Controlled-Phase (CPhase) of the CZ Gate

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The amount of leakage to the |f> state when the control qubit is in the excited state

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Measuring the phase of the target qubit for both control qubit states
3. Calculating the phase difference
4. Measuring the population in the |f> state to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

Outcomes:
- Optimal CZ gate amplitude for achieving a π phase shift
- Leakage characteristics across the amplitude range
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = None
    num_averages: int = 500
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    amp_range : float = 0.020
    amp_step : float = 0.001
    num_frames: int = 10
    load_data_id: Optional[int] = None # 92417 
    plot_raw : bool = False
    measure_leak : bool = True


node = QualibrationNode(
    name="32a_Cz_phase_calibration_frame", parameters=Parameters()
)
assert not (node.parameters.simulate and node.parameters.load_data_id is not None), "If simulate is True, load_data_id must be None, and vice versa."

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubit_pairs is None or node.parameters.qubit_pairs == "":
    qubit_pairs = machine.active_qubit_pairs
else:
    qubit_pairs = [machine.qubit_pairs[qp] for qp in node.parameters.qubit_pairs]
# if any([qp.q1.z is None or qp.q2.z is None for qp in qubit_pairs]):
#     warnings.warn("Found qubit pairs without a flux line. Skipping")

num_qubit_pairs = len(qubit_pairs)

# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
# %%

####################
# Helper functions #
####################

def tanh_fit(x, a, b, c, d):
    return a * np.tanh(b * x + c) + d


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# Loop parameters
amplitudes = np.arange(1-node.parameters.amp_range, 1+node.parameters.amp_range, node.parameters.amp_step)
frames = np.arange(0, 1, 1/node.parameters.num_frames)

with program() as CPhase_Oscillations:
    amp = declare(fixed)   
    frame = declare(fixed)
    control_initial = declare(int)
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            # qp.apply_mutual_flux_point()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(amp, amplitudes)):
                with for_(*from_array(frame, frames)):
                    with for_(*from_array(control_initial, [0,1])):
                        # reset
                        if node.parameters.reset_type == "active":
                            active_reset_gef(qp.qubit_control)
                            active_reset(qp.qubit_target)
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                        qp.align()
                        reset_frame(qp.qubit_target.xy.name)
                        reset_frame(qp.qubit_control.xy.name)                   
                        # setting both qubits ot the initial state
                        qp.qubit_control.xy.play("x180", condition=control_initial==1)
                        qp.qubit_target.xy.play("x90")
                        qp.align()

                        #play the CZ gate
                        qp.gates['Cz'].execute(amplitude_scale = amp)
                        
                        #rotate the frame
                        frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                        
                        # return the target qubit before measurement
                        qp.qubit_target.xy.play("x90")                        
                            
                        # measure both qubits
                        if node.parameters.measure_leak:
                            readout_state_gef(qp.qubit_control, state_control[i])
                        else:
                            readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])  
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(n_avg).save(f"state_control{i + 1}")
            state_st_target[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(n_avg).save(f"state_target{i + 1}")

# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, CPhase_Oscillations, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(CPhase_Oscillations)

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"control_axis": [0,1], "frame": frames, "amp": amplitudes, "N": np.linspace(1, n_avg, n_avg)})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)

        
    node.results = {"ds": ds}

# %% {Data_analysis}
if not node.parameters.simulate:
    def abs_amp(qp, amp):
        return amp * qp.gates['Cz'].flux_pulse_control.amplitude

    def detuning(qp, amp):
        return -(amp * qp.gates['Cz'].flux_pulse_control.amplitude)**2 * qp.qubit_control.freq_vs_flux_01_quad_term
    
    ds = ds.assign_coords(
        {"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qubit", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))}
    )
# %% Analysis
if not node.parameters.simulate:

    phase_diffs = {}
    optimal_amps = {}
    leaks = {}
    fitted = {}
    for qp in qubit_pairs:
        ds_qp = ds.sel(qubit=qp.name)
        fit_data = fit_oscillation(ds_qp.state_target.mean(dim = 'N'), "frame")
        
        ds_qp = ds_qp.assign({'fitted': oscillation(ds_qp.frame,
                                                    fit_data.sel(fit_vals="a"),
                                                    fit_data.sel(fit_vals="f"),
                                                    fit_data.sel(
                                                        fit_vals="phi"),
                                                    fit_data.sel(fit_vals="offset"))})
        if node.parameters.plot_raw:
            plt.figure()
            ds_qp.mean(dim = 'N').to_array()\
                .sel(variable=["state_target", "fitted"])\
                .stack(control_axis_fit=("control_axis", "variable"))\
                .plot.line(x='frame', col='amp', col_wrap=4)
            plt.show()
            
        phase = fix_oscillation_phi_2pi(fit_data)    
        phase_diff = (phase.sel(control_axis=0)-phase.sel(control_axis=1)) % 1
        try:
            fit_params, _ = curve_fit(tanh_fit, phase_diff.amp, phase_diff, p0=[-0.5,100,-100,0.5])
            optimal_amp = ( np.arctanh((0.5 - fit_params[3])/fit_params[0]) - fit_params[2])/fit_params[1]
            fitted[qp.name] = tanh_fit(phase_diff.amp, *fit_params)
        except:
            print(f"Fitting failed for {qp.name}")
            optimal_amp = float(np.abs(phase_diff - 0.5).idxmin("amp"))    
        
        phase_diffs[qp.name] = phase_diff
        optimal_amps[qp.name] = optimal_amp * qp.gates['Cz'].flux_pulse_control.amplitude
        
        print(f"parameters for {qp.name}: amp={optimal_amps[qp.name]}")
        
        if node.parameters.measure_leak:
            all_counts = (ds_qp.state_control < 3).sum(dim = 'N').sel(control_axis = 1).sum(dim = 'frame')
            leak_counts = (ds_qp.state_control == 2).sum(dim = 'N').sel(control_axis = 1).sum(dim = 'frame')
            leaks[qp.name] = leak_counts / all_counts  

# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        phase_diffs[qubit_pair['qubit']].plot.line(ax=ax, x = "amp_full")
        if qubit_pair['qubit'] in fitted:
            ax.plot(phase_diffs[qubit_pair['qubit']].amp_full, fitted[qubit_pair['qubit']])
        ax.plot([optimal_amps[qubit_pair['qubit']]], [0.5], marker = 'o', color = 'red')
        ax.axhline(y=0.5, color='red', linestyle='--',lw=0.5)
        ax.axvline(x=optimal_amps[qubit_pair['qubit']], color='red', linestyle='--',lw=0.5)
        # Add secondary x-axis for detuning in MHz
        def amp_to_detuning_MHz(amp):
            return -(amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term / 1e6  # Convert Hz to MHz

        def detuning_MHz_to_amp(detuning_MHz):
            return np.sqrt(-detuning_MHz * 1e6 / qp.qubit_control.freq_vs_flux_01_quad_term)

        secax = ax.secondary_xaxis('top', functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
        secax.set_xlabel('Detuning (MHz)')
        ax.set_title(qubit_pair['qubit'])
        ax.set_xlabel('Amplitude (V)')
        ax.set_ylabel('Phase difference')
        
    plt.suptitle('Cz phase calibration', y=0.95)
    plt.tight_layout()
    plt.show()
    node.results["figure_phase"] = grid.fig
    
    if node.parameters.measure_leak:
        grid = QubitPairGrid(grid_names, qubit_pair_names)
        for ax, qubit_pair in grid_iter(grid):
            leaks[qubit_pair['qubit']].plot(ax=ax, x = 'amp_full')
            ax.axvline(optimal_amps[qubit_pair['qubit']],color = 'r', linestyle='--',lw=0.5)
            ax.set_title(qubit_pair['qubit'])
            ax.set_xlabel('Amplitude (V)')
            ax.set_ylabel('Leak probability')
            
            # Add secondary x-axis for detuning in MHz
            def amp_to_detuning_MHz(amp):
                return -(amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term / 1e6  # Convert Hz to MHz

            def detuning_MHz_to_amp(detuning_MHz):
                return np.sqrt(-detuning_MHz * 1e6 / qp.qubit_control.freq_vs_flux_01_quad_term)

            secax = ax.secondary_xaxis('top', functions=(amp_to_detuning_MHz, detuning_MHz_to_amp))
            secax.set_xlabel('Detuning (MHz)')
                    
        plt.suptitle('F state probability', y=0.95)
        plt.tight_layout()    
        plt.show()
        node.results['figure_leak'] = grid.fig    

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.gates['Cz'].flux_pulse_control.amplitude = optimal_amps[qp.name]

                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%