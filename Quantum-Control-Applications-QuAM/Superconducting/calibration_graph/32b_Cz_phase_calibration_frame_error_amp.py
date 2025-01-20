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
    amp_range : float = 0.01
    amp_step : float = 0.0002
    num_frames: int = 10
    num_repeats: int = 10
    load_data_id: Optional[int] = None
    measure_leak : bool = True


node = QualibrationNode(
    name="32b_Cz_phase_calibration_frame_error_amp", parameters=Parameters()
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

repeats = np.arange(1, node.parameters.num_repeats, 2)

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
    leakage_control = [declare(fixed) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    leakage_control_st = [declare_stream() for _ in range(num_qubit_pairs)]
    n_repeats = declare(int)
    count = declare(int)
    
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
                        with for_(*from_array(n_repeats, repeats)):
                            # reset
                            if node.parameters.reset_type == "active":
                                active_reset_gef(qp.qubit_control)
                                qp.align()
                                active_reset(qp.qubit_target)
                                qp.align()
                            else:
                                wait(qp.qubit_control.thermalization_time * u.ns)
                            qp.align()
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)                   
                            # setting both qubits ot the initial state
                            qp.qubit_control.xy.play("x180", condition=control_initial==1)
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            with for_(count, 0, count < n_repeats, count + 1):
                                #play the CZ gate
                                qp.gates['Cz'].execute(amplitude_scale = amp)
                                qp.align()
                                qp.qubit_control.z.wait(50)
                                qp.align()
                                
                            #rotate the frame
                            frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                            qp.align()
                            
                            # return the target qubit before measurement
                            qp.qubit_target.xy.play("x90")                        
                                
                            # measure both qubits
                            readout_state_gef(qp.qubit_control, state_control[i])
                            readout_state(qp.qubit_target, state_target[i])
                            assign(leakage_control[i], Cast.to_fixed( state_control[i] == 2))
                            save(state_control[i], state_st_control[i])
                            save(state_target[i], state_st_target[i])  
                            save(leakage_control[i], leakage_control_st[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(repeats)).buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(repeats)).buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"state_target{i + 1}")
            leakage_control_st[i].buffer(len(repeats)).buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"leakage_control{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"repeats": repeats, "control_axis": [0,1], "frame": frames, "amp": amplitudes})
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
    for qp in qubit_pairs:
        ds_qp = ds.sel(qubit=qp.name)
        fit_data = fit_oscillation(ds_qp.state_target, "frame")
        
        ds_qp = ds_qp.assign({'fitted': oscillation(ds_qp.frame,
                                                    fit_data.sel(fit_vals="a"),
                                                    fit_data.sel(fit_vals="f"),
                                                    fit_data.sel(
                                                        fit_vals="phi"),
                                                    fit_data.sel(fit_vals="offset"))})
        phase = fix_oscillation_phi_2pi(fit_data)    
        phase_diff = (phase.sel(control_axis=0)-phase.sel(control_axis=1)) % 1 
        optimal_amps[qp.name] = phase_diff.amp_full[np.abs(phase_diff-0.5).mean(dim = 'repeats').argmin(dim = 'amp')]
        phase_diffs[qp.name] = phase_diff

    # %%
    (phase_diff-0.5).plot(x = "repeats", y = "amp_full")
    phase_diff.amp_full[np.abs(phase_diff-0.5).mean(dim = 'repeats').argmin(dim = 'amp')]
    # %%
    ds.leakage_control.mean(dim = ("frame","control_axis")).plot()

# %%

# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        
        data_to_plot = phase_diffs[qubit_pair['qubit']].assign_coords(detuning_MHz = 1e-6*phase_diffs[qubit_pair['qubit']].detuning)-0.5
        plot = data_to_plot.plot(x = "repeats", y = "detuning_MHz", add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Phase')

        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
        print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")
        
        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax.axhline(y=1e6*flux_to_detuning(optimal_amps[qubit_pair['qubit']], quad), color='k', linestyle='--', lw = 0.5)
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')

        
    plt.suptitle('Cz phase calibration', y=0.95)
    plt.tight_layout()
    plt.show()
    node.results["figure_phase"] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        
        data_to_plot = ds.sel(qubit = qubit_pair['qubit']).leakage_control.mean(dim = ("frame","control_axis")).assign_coords(detuning_MHz = 1e-6*ds.sel(qubit = qubit_pair['qubit']).detuning)
        plot = data_to_plot.plot(x = "repeats", y = "detuning_MHz", add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='leakage')

        quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
        print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")
        
        def detuning_to_flux(det, quad = quad):
            return 1e3 * np.sqrt(-1e6 * det / quad)

        def flux_to_detuning(flux, quad = quad):
            return -1e-6 * (flux/1e3)**2 * quad
        
        ax2 = ax.secondary_yaxis('right', functions=(detuning_to_flux, flux_to_detuning))
        ax.axhline(y=1e6*flux_to_detuning(optimal_amps[qubit_pair['qubit']], quad), color='r', linestyle='--', lw = 0.5)
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')

        
    plt.suptitle('Cz phase calibration', y=0.95)
    plt.tight_layout()
    plt.show()
    node.results['figure_leak'] = grid.fig    

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                qp.gates['Cz'].flux_pulse_control.amplitude = float(optimal_amps[qp.name].values)

                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%