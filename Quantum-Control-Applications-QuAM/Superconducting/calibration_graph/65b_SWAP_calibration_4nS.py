# %%
"""
Two-Qubit Readout Confusion Matrix Measurement

This sequence measures the readout error when simultaneously measuring the state of two qubits. The process involves:

1. Preparing the two qubits in all possible combinations of computational basis states (|00⟩, |01⟩, |10⟩, |11⟩)
2. Performing simultaneous readout on both qubits
3. Calculating the confusion matrix based on the measurement results

For each prepared state, we measure:
1. The readout result of the first qubit
2. The readout result of the second qubit

The measurement process involves:
1. Initializing both qubits to the ground state
2. Applying single-qubit gates to prepare the desired input state
3. Performing simultaneous readout on both qubits
4. Repeating the process multiple times to gather statistics

The outcome of this measurement will be used to:
1. Quantify the readout fidelity for two-qubit states
2. Identify and characterize crosstalk effects in the readout process
3. Provide data for readout error mitigation in two-qubit experiments

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits

Outcomes:
- 4x4 confusion matrix representing the probabilities of measuring each two-qubit state given a prepared input state
- Readout fidelity metrics for simultaneous two-qubit measurement
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
from quam_libs.lib.fit import extract_dominant_frequencies
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
from scipy.fft import fft
import xarray as xr
from quam_libs.components.gates.two_qubit_gates import SWAP_Coupler_Gate
from quam_libs.lib.fit import oscillation_decay_exp, fit_oscillation_decay_exp

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ["q0-q1"]
    num_averages: int = 200
    flux_point_joint_or_independent_or_pairwise: Literal["joint", "independent", "pairwise"] = "pairwise"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    load_data_id: Optional[int] = None
    control_amp_range : float = 0.4
    control_amp_step : float = 0.02
    idle_time_min : int = 16
    idle_time_max : int = 250
    idle_time_step : int = 4
    use_state_discrimination: bool = True
    
node = QualibrationNode(
    name="65b_SWAP_calibration_4nS", parameters=Parameters()
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

def rabi_chevron_model(ft, J, f0, a, offset,tau):
    f,t = ft
    J = J
    w = f
    w0 = f0
    g = offset+a * np.sin(2*np.pi*np.sqrt(J**2 + (w-w0)**2) * t)**2*np.exp(-tau*np.abs((w-w0))) 
    return g.ravel()

def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    da_target = ds_qp.state_target
    exp_data = da_target.values
    detuning = da_target.detuning
    time = da_target.idle_time*4*1e-9
    t,f  = np.meshgrid(time,detuning)
    initial_guess = (1e9/init_length/2,
            init_detuning,
            -1,
            1.0,
            100e-9)
    fdata = np.vstack((f.ravel(),t.ravel()))
    tdata = exp_data.ravel()
    popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
    J = popt[0]
    f0 = popt[1]
    a = popt[2]
    offset = popt[3]
    tau = popt[4]

    return J, f0, a, offset, tau

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
# Loop parameters
control_amps = np.arange(1 - node.parameters.control_amp_range, 1 + node.parameters.control_amp_range, node.parameters.control_amp_step)
idle_times = np.arange(node.parameters.idle_time_min, node.parameters.idle_time_max, node.parameters.idle_time_step) // 4

with program() as CPhase_Oscillations:
    n = declare(int)
    
    amp = declare(float)
    idle_time = declare(int)
    n_st = declare_stream()
    if node.parameters.use_state_discrimination:
        state_control = [declare(int) for _ in range(num_qubit_pairs)]
        state_target = [declare(int) for _ in range(num_qubit_pairs)]
        state = [declare(int) for _ in range(num_qubit_pairs)]
        state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_stream() for _ in range(num_qubit_pairs)]
    else:
        I_control = [declare(float) for _ in range(num_qubit_pairs)]
        Q_control = [declare(float) for _ in range(num_qubit_pairs)]
        I_target = [declare(float) for _ in range(num_qubit_pairs)]
        Q_target = [declare(float) for _ in range(num_qubit_pairs)]
        I_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
        I_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
        Q_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    
    for i, qp in enumerate(qubit_pairs):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point, qp)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)         
            with for_(*from_array(amp, control_amps)):
                with for_(*from_array(idle_time, idle_times)):
                    # reset
                    if node.parameters.reset_type == "active":
                            active_reset(qp.qubit_control)
                            active_reset(qp.qubit_target)
                            qp.align()
                    else:
                        wait(qp.qubit_control.thermalization_time * u.ns)
                        wait(qp.qubit_target.thermalization_time * u.ns)
                    align()
                    
                    # setting both qubits ot the initial state
                    qp.qubit_control.xy.play("x180")
                    
                                    
                    align()
                    qp.qubit_control.z.play("const", amplitude_scale = amp * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude  / 0.1, duration = idle_time)
                    qp.coupler.play("const", amplitude_scale = qp.gates["SWAP_Coupler"].coupler_pulse_control.amplitude / 0.1, duration = idle_time)
                    align()
                    # readout
                    if node.parameters.use_state_discrimination:
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        assign(state[i], state_control[i]*2 + state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])
                        save(state[i], state_st[i])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_control[i], Q_control[i]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_target[i], Q_target[i]))
                        save(I_control[i], I_st_control[i])
                        save(Q_control[i], Q_st_control[i])
                        save(I_target[i], I_st_target[i])
                        save(Q_target[i], Q_st_target[i])
        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st_control[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"state_control{i + 1}")
                state_st_target[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"state_target{i + 1}")
                state_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"state{i + 1}")
            else:
                I_st_control[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"I_control{i + 1}")
                Q_st_control[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"Q_control{i + 1}")
                I_st_target[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"I_target{i + 1}")
                Q_st_target[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"Q_target{i + 1}")

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
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {  "idle_time": idle_times, "amp": control_amps})
    else:
        ds, machine = load_dataset(node.parameters.load_data_id)
        
    node.results = {"ds": ds}

# %%
if not node.parameters.simulate:
    ds = ds.assign_coords(idle_time = ds.idle_time * 4)
    ds = ds.assign({"res_sum" : ds.state_control - ds.state_target})
    amp_full = np.array([control_amps * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude for qp in qubit_pairs])
    ds = ds.assign_coords({"amp_full": (["qubit", "amp"], amp_full)})    
    detunings = np.array([-(control_amps * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude)**2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in qubit_pairs])
    ds = ds.assign_coords({"detuning": (["qubit", "amp"], detunings)})
# %%
if not node.parameters.simulate:
    amplitudes = {}
    lengths = {}
    zero_paddings = {}
    fitted_ds = {}
    detunings = {}
    Js = {}
    RC_success = {}
    
    for qp in qubit_pairs:
        print(qp.name)
        ds_qp = ds.sel(qubit=qp.name)

        amp_guess = ds_qp.state_target.max("idle_time")-ds_qp.state_target.min("idle_time")
        flux_amp_idx = int(amp_guess.argmax())
        flux_amp = float(ds_qp.amp_full[flux_amp_idx])
        fit_data = fit_oscillation_decay_exp(
            ds_qp.state_control.isel(amp=flux_amp_idx), "idle_time")
        flux_time = int(1/fit_data.sel(fit_vals='f'))

        amplitudes[qp.name] =  flux_amp
        detunings[qp.name] = -flux_amp ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term
        lengths[qp.name] = flux_time
        fitted_ds[qp.name]  = ds_qp.assign({'fitted': oscillation_decay_exp(ds_qp.idle_time,
                                                                fit_data.sel(
                                                                    fit_vals="a"),
                                                                fit_data.sel(
                                                                    fit_vals="f"),
                                                                fit_data.sel(
                                                                    fit_vals="phi"),
                                                                fit_data.sel(
                                                                    fit_vals="offset"),
                                                                fit_data.sel(fit_vals="decay"))})
        try:
            t = ds.idle_time*1e-9
            f = ds.sel(qubit=qp.name).detuning
            t,f = np.meshgrid(t,f)
            J, f0, a, offset, tau = fit_rabi_chevron(ds_qp, lengths[qp.name], detunings[qp.name])
            data_fitted = rabi_chevron_model((f,t), J, f0, a, offset, tau).reshape(len(ds.amp), len(ds.idle_time))
            Js[qp.name] = J
            detunings[qp.name] = f0
            amplitudes[qp.name] = np.sqrt(-detunings[qp.name]/qp.qubit_control.freq_vs_flux_01_quad_term)
            flux_time = int(1/(2*J)*1e9)
            lengths[qp.name] = flux_time
            RC_success[qp.name] = True
        except:
            print(f"Rabi-Chevron fit for {qp.name} failed")
            RC_success[qp.name] = False
# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.to_array().sel(qubit=qubit_pair['qubit']).sel(
            variable='state_control').assign_coords(detuning_MHz = 1e-6*ds.detuning.sel(qubit = qp.name)).plot(ax = ax, x= 'idle_time', y= 'detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        # ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6*detunings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        if RC_success[qubit_pair['qubit']]:
            f_eff = np.sqrt(Js[qubit_pair['qubit']]**2 + (ds.detuning.sel(qubit=qubit_pair['qubit'])-detunings[qubit_pair['qubit']])**2)
            for n in range(10):
                ax.plot(n*0.5/f_eff*1e9,1e-6*ds.detuning.sel(qubit = qubit_pair['qubit']), color = 'red', lw = 0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qubit=qubit_pair['qubit'])
        amp_full_range = np.sqrt(-detuning_range / qp.qubit_control.freq_vs_flux_01_quad_term)
        ax2.set_ylim(amp_full_range.min(), amp_full_range.max())
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        ax.set_ylim(detuning_range.min() * 1e-6, detuning_range.max() * 1e-6)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
        
    plt.suptitle('control qubit state')
    plt.show()
    node.results["figure_control"] = grid.fig
    
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.to_array().sel(qubit =qubit_pair['qubit']).sel(
            variable='state_target').assign_coords(detuning_MHz = 1e-6*ds.detuning.sel(qubit = qp.name)).plot(ax = ax, x= 'idle_time', y= 'detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        # ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6*detunings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        if RC_success[qubit_pair['qubit']]:
            f_eff = np.sqrt(Js[qubit_pair['qubit']]**2 + (ds.detuning.sel(qubit =qubit_pair['qubit'])-detunings[qubit_pair['qubit']])**2)
            for n in range(10):
                ax.plot(n*0.5/f_eff*1e9,1e-6*ds.detuning.sel(qubit = qubit_pair['qubit']), color = 'red', lw = 0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qubit =qubit_pair['qubit'])
        amp_full_range = np.sqrt(-detuning_range / qp.qubit_control.freq_vs_flux_01_quad_term)
        ax2.set_ylim(amp_full_range.min(), amp_full_range.max())
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        ax.set_ylim(detuning_range.min() * 1e-6, detuning_range.max() * 1e-6)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
    plt.suptitle('target qubit state')
    plt.show()
    node.results["figure_target"] = grid.fig


# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:
            gate_time_ns = int(lengths[qp.name] / 2)
            gate_time_including_zeros = gate_time_ns - gate_time_ns % 4 + 4
            zero_padding = gate_time_including_zeros - gate_time_ns
            flux_pulse_amp = amplitudes[qp.name]            
            qp.gates['SWAP_Coupler'].flux_pulse_control.amplitude = flux_pulse_amp
            qp.gates['SWAP_Coupler'].flux_pulse_control.zero_padding = zero_padding
            qp.gates['SWAP_Coupler'].flux_pulse_control.length = gate_time_including_zeros
            
# %% {Save_results}
if not node.parameters.simulate:    
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    node.save()
# %%
