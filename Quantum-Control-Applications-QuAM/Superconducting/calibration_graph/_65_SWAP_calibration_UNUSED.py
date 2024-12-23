# %%
"""
Unipolar CPhase Gate Calibration

This sequence measures the time and detuning required for a unipolar CPhase gate. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

From this pattern, we extract:
- The coupling strength (J2) between the qubits.
- The optimal gate parameters (amplitude and duration) for the CPhase gate.

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning, allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.
"""

# %% {Imports}
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation
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
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubit_pairs: Optional[List[str]] = ['q0-q1']
    num_averages: int = 100
    max_time_in_ns: int = 16
    flux_point_joint_or_independent: Literal["joint", "independent, pairwise"] = "pairwise"
    reset_type: Literal['active', 'thermal'] = "active"
    simulate: bool = False
    timeout: int = 100
    method: Literal['coarse', 'fine'] = "coarse"
    amp_range_coarse : float = 0.3
    amp_step_coarse : float = 0.1
    amp_range_fine : float = 0.01
    amp_step_fine : float = 0.005
    load_data_id: Optional[int] = None 

node = QualibrationNode(
    name="65_SWAP_calibration", parameters=Parameters()
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

def baked_waveform(qp):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform_control = [qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude] * 16
    waveform_coupler = [qp.gates["SWAP_Coupler"].coupler_pulse_control.amplitude] * 16

    for i in range(1, 17):  # from first item up to pulse_duration (16)
        with baking(config, padding_method="left") as b:
            wf_control = waveform_control[:i]
            wf_coupler = waveform_coupler[:i]
            b.add_op("flux_pulse", qp.qubit_control.z.name, wf_control)
            b.add_op("flux_pulse", qp.coupler.name, wf_coupler)
            b.play("flux_pulse", qp.qubit_control.z.name)
            b.play("flux_pulse", qp.coupler.name)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)
        
    return pulse_segments

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
    time = da_target.time*1e-9
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

flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'

# define the amplitudes for the flux pulses

baked_signals = {qp.name : baked_waveform(qp) for qp in qubit_pairs}

# Loop parameters
amplitudes = np.arange(1-node.parameters.amp_range_fine, 1+node.parameters.amp_range_fine, node.parameters.amp_step_fine)
times_ns = np.arange(1, node.parameters.max_time_in_ns)

with program() as CPhase_Oscillations:
    t = declare(int)  # QUA variable for the flux pulse segment index
    idx = declare(int)
    amp = declare(fixed)    
    n = declare(int)
    n_st = declare_stream()
    state_control = [declare(int) for _ in range(num_qubit_pairs)]
    state_target = [declare(int) for _ in range(num_qubit_pairs)]
    state_st_control = [declare_stream() for _ in range(num_qubit_pairs)]
    state_st_target = [declare_stream() for _ in range(num_qubit_pairs)]
    
    for i, qp in enumerate(qubit_pairs):
        machine.set_all_fluxes(flux_point, qp)
        wait(1000)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(amp, amplitudes)):
                # first 16 nS                
                with for_(idx, 0, idx<16, idx+1):
                    # reset                    
                    if node.parameters.reset_type == "active":
                        active_reset(qp.qubit_control)
                        active_reset(qp.qubit_target)
                        qp.align()
                    else:
                        wait(1*qp.qubit_control.thermalization_time * u.ns)
                    # set both qubits to the excited state
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_control.xy.wait(5)
                    qp.align()

                    # play the flux pulse
                    with switch_(idx):
                        for j in range(16):
                            with case_(j):
                                baked_signals[qp.name][j].run(amp_array = [(qp.qubit_control.z, amp), (qp.coupler, amp)]) 
                                                                               
                    # wait for the flux pulse to end and some extra time
                    for qubit in [qp.qubit_control, qp.qubit_target]:
                        qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                    qp.align()
                    
                    # measure both qubits
                    for qubit, state, state_st in zip([qp.qubit_control, qp.qubit_target], [state_control, state_target], [state_st_control, state_st_target]):
                        readout_state(qubit, state[i]) # TODO: readout gef
                        # save data
                        save(state[i], state_st[i])  
                                                                                                                   
                # rest of the pulse
                with for_(t, 4, t < node.parameters.max_time_in_ns // 4, t + 4):
                    with for_(idx, 0, idx<16, idx+1):
                        # reset                    
                        if node.parameters.reset_type == "active":
                            active_reset_gef(qp.qubit_control)
                            active_reset(qp.qubit_target)
                            qp.align()
                        else:
                            wait(qp.qubit_control.thermalization_time * u.ns)
                        # set both qubits to the excited state
                        qp.qubit_control.xy.play("x180")
                        qp.qubit_control.xy.wait(5)
                        qp.align()

                        # play the flux pulse
                        with switch_(idx):
                            for j in range(16):
                                with case_(j):
                                    baked_signals[qp.name][j].run(amp_array = [(qp.qubit_control.z, amp), (qp.coupler, amp)])
                        qp.qubit_control.z.play('const', duration=t, amplitude_scale = qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude / qp.qubit_control.z.operations['const'].amplitude * amp)
                        qp.coupler.play('const', duration=t, amplitude_scale =  qp.gates["SWAP_Coupler"].coupler_pulse_control.amplitude / qp.coupler.operations['const'].amplitude * amp)
                        # wait for the flux pulse to end and some extra time
                        for qubit in [qp.qubit_control, qp.qubit_target]:
                            qubit.xy.wait(node.parameters.max_time_in_ns // 4 + 10)
                        qp.align()         
                                   
                        # measure both qubits
                        readout_state(qp.qubit_control, state_control[i])
                        readout_state(qp.qubit_target, state_target[i])
                        save(state_control[i], state_st_control[i])
                        save(state_target[i], state_st_target[i])

        align()
        
    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            state_st_control[i].buffer(len(amplitudes)).buffer(len(times_ns)).average().save(f"state_control{i + 1}")
            state_st_target[i].buffer(len(amplitudes)).buffer(len(times_ns)).average().save(f"state_target{i + 1}")

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
        ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"amp": amplitudes, "time": times_ns})
    else:
        ds, loaded_machine = load_dataset(node.parameters.load_data_id)
        if loaded_machine is not None:
            machine = loaded_machine
        
    node.results = {"ds": ds}

# %%
ds.state_control.plot(vmin = 0, vmax = 1)
plt.show()
ds.state_target.plot(vmin = 0, vmax = 1)
plt.show()


# %% {Data_analysis}
if not node.parameters.simulate:
    def abs_amp(qp, amp):
        return amp * pulse_amplitudes[qp.name]

    def detuning(qp, amp):
        return -(amp * pulse_amplitudes[qp.name])**2 * qp.qubit_control.freq_vs_flux_01_quad_term
    
    ds = ds.assign_coords(
        {"amp_full": (["qp", "amp"], np.array([abs_amp(qp, ds.amp) for qp in qubit_pairs]))}
    )
    ds = ds.assign_coords(
        {"detuning": (["qp", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))}
    )
# %%
if not node.parameters.simulate:
    amplitudes = {}
    lengths = {}
    zero_paddings = {}
    fitted_ds = {}
    detunings = {}
    Js = {}
    
    for qp in qubit_pairs:
        print(qp.name)
        ds_qp = ds.sel(qp=qp.name)

        amp_guess = ds_qp.state_target.max("time")-ds_qp.state_target.min("time")
        flux_amp_idx = int(amp_guess.argmax())
        flux_amp = float(ds_qp.amp_full[flux_amp_idx])
        fit_data = fit_oscillation_decay_exp(
            ds_qp.state_control.isel(amp=flux_amp_idx), "time")
        flux_time = int(1/fit_data.sel(fit_vals='f'))

        print(f"parameters for {qp.name}: amp={flux_amp}, time={flux_time}")
        amplitudes[qp.name] =  flux_amp
        detunings[qp.name] = -flux_amp ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term
        lengths[qp.name] = flux_time-flux_time%4+4
        zero_paddings[qp.name]=lengths[qp.name]-flux_time
        fitted_ds[qp.name]  = ds_qp.assign({'fitted': oscillation_decay_exp(ds_qp.time,
                                                                fit_data.sel(
                                                                    fit_vals="a"),
                                                                fit_data.sel(
                                                                    fit_vals="f"),
                                                                fit_data.sel(
                                                                    fit_vals="phi"),
                                                                fit_data.sel(
                                                                    fit_vals="offset"),
                                                                fit_data.sel(fit_vals="decay"))})
        # if node.parameters.method == "fine":
        if True:
            t = ds.time*1e-9
            f = ds.sel(qp = qp.name).detuning
            t,f = np.meshgrid(t,f)
            J, f0, a, offset, tau = fit_rabi_chevron(ds_qp, lengths[qp.name], detunings[qp.name])
            data_fitted = rabi_chevron_model((f,t), J, f0, a, offset, tau).reshape(len(ds.amp), len(ds.time))
            Js[qp.name] = J
            detunings[qp.name] = f0
            amplitudes[qp.name] = np.sqrt(-detunings[qp.name]/qp.qubit_control.freq_vs_flux_01_quad_term)
            flux_time = int(1/(2*J)*1e9)
            lengths[qp.name] = flux_time-flux_time%4+4
            zero_paddings[qp.name]=lengths[qp.name]-flux_time     
# %%
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.to_array().sel(qp=qubit_pair['qubit']).sel(
            variable='state_control').assign_coords(detuning_MHz = 1e-6*ds.detuning.sel(qp = qp.name)).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6*detunings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        f_eff = np.sqrt(Js[qubit_pair['qubit']]**2 + (ds.detuning.sel(qp=qubit_pair['qubit'])-detunings[qubit_pair['qubit']])**2)
        for n in range(10):
            ax.plot(n*0.5/f_eff*1e9,1e-6*ds.detuning.sel(qp = qubit_pair['qubit']), color = 'red', lw = 0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qp=qubit_pair['qubit'])
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
        plot = ds.to_array().sel(qp=qubit_pair['qubit']).sel(
            variable='state_target').assign_coords(detuning_MHz = 1e-6*ds.detuning.sel(qp = qp.name)).plot(ax = ax, x= 'time', y= 'detuning_MHz', add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6*detunings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']], color='k', linestyle='--', lw = 0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        f_eff = np.sqrt(Js[qubit_pair['qubit']]**2 + (ds.detuning.sel(qp=qubit_pair['qubit'])-detunings[qubit_pair['qubit']])**2)
        for n in range(10):
            ax.plot(n*0.5/f_eff*1e9,1e-6*ds.detuning.sel(qp = qubit_pair['qubit']), color = 'red', lw = 0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qp=qubit_pair['qubit'])
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

# %%

# %% {Update_state}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        with node.record_state_updates():
            for qp in qubit_pairs:
                if "Cz_unipolar" in qp.gates:
                    qp.gates['Cz_unipolar'].flux_pulse_control.amplitude = amplitudes[qp.name]
                    qp.gates['Cz_unipolar'].flux_pulse_control.length = lengths[qp.name]
                    qp.gates['Cz_unipolar'].flux_pulse_control.zero_padding = zero_paddings[qp.name]
                    qp.gates['Cz'] = f"#./Cz_unipolar"
                else:
                    qp.gates['Cz_unipolar'] = CZGate(flux_pulse_control = FluxPulse(length=lengths[qp.name], amplitude=amplitudes[qp.name], zero_padding=zero_paddings[qp.name], id = 'flux_pulse_control_' + qp.qubit_target.name))
                    qp.gates['Cz'] = f"#./Cz_unipolar"
                
                qp.J2 = Js[qp.name]
                qp.detuning = detunings[qp.name]
                
# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    node.save()
        
# %%