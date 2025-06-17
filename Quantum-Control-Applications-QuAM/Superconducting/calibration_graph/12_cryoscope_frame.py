# %%
"""
        CRYOSCOPE
"""

from qm import QuantumMachinesManager
from qm.qua import *
from qm import SimulationConfig
from scipy import signal
import matplotlib.pyplot as plt
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.plot import interrupt_on_close
from qualang_tools.multi_user import qm_session
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset
import numpy as np
from qualang_tools.units import unit
from quam_libs.components import QuAM
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, save_node
import xarray as xr
from scipy.optimize import curve_fit, minimize
from scipy.signal import deconvolve, lfilter, convolve
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from quam_libs.lib.cryoscope_tools import cryoscope_frequency, estimate_fir_coefficients, two_expdecay, expdecay, savgol


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None 
    num_averages: int = 15000
    amplitude_factor: float = 0.3
    cryoscope_len: int = 32
    num_frames: int = 17
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    only_FIR: bool = False
    load_data_id: Optional[int] = None
    
node = QualibrationNode(
    name="97b_Pi_vs_flux_time",
    parameters=Parameters()
)




# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# machine = QuAM.load()
# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
              
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_active_or_thermal
amplitude_factor = node.parameters.amplitude_factor

num_qubits = len(qubits)

# %%

####################
# Helper functions #
####################

def baked_waveform(waveform_amp, qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [waveform_amp] * 16

    for i in range(1, 17):  # from first item up to pulse_duration (16)
        with baking(config, padding_method="left") as b:
            wf = waveform[:i]
            b.add_op("flux_pulse", qubit.z.name, wf)
            b.play("flux_pulse", qubit.z.name)

        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)

    return pulse_segments


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

assert cryoscope_len % 16 == 0, 'cryoscope_len is not multiple of 16 nanoseconds'

baked_signals = {}
# Baked flux pulse segments with 1ns resolution

baked_signals = baked_waveform(qubits[0].z.operations['const'].amplitude * amplitude_factor, qubits[0]) 

cryoscope_time = np.arange(1, cryoscope_len + 5, 1)  # x-axis for plotting - must be in ns
frames = np.linspace(0, 1, node.parameters.num_frames)
# %%

with program() as cryoscope:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the flux pulse segment index
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    global_state = declare(int)
    idx = declare(int)
    idx2 = declare(int)
    frame = declare(fixed)
    qubit = qubits[0]
    i = 0
    
    # Bring the active qubits to the desired frequency point
    machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        # The first 4 nanoseconds
        with for_(idx, 0, idx<4, idx+1):
            with for_(*from_array(frame, frames)):
                if reset_type == "active":
                    for qubit in qubits:
                        active_reset(qubit)
                else:
                    wait(qubit.thermalization_time * u.ns)
                align()
                # Play first X/2
                for qubit in qubits:
                    qubit.xy.play("x90")
                align()
                # Delay between x90 and the flux pulse
                # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
                wait(50)
                align()
                # with switch_(idx):
                #     for j in range(16):
                #         with case_(j):
                #             baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_len + 160) // 4)
                    # Play second X/2
                    frame_rotation_2pi(frame, qubit.xy.name)
                    qubit.xy.play("x90")

                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])

        # The first 16-32 nanoseconds
        with for_(idx, 0, idx<16, idx+1):
            with for_(*from_array(frame, frames)):
                if reset_type == "active":
                    for qubit in qubits:
                        active_reset(qubit)
                else:
                    wait(qubit.thermalization_time * u.ns)
                align()
                # Play first X/2
                for qubit in qubits:
                    qubit.xy.play("x90")
                align()
                # Delay between x90 and the flux pulse
                # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
                wait(50)
                align()
                with switch_(idx):
                    for j in range(16):
                        with case_(j):
                            baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_len + 160) // 4)
                    # Play second X/2
                    frame_rotation_2pi(frame, qubit.xy.name)
                    qubit.xy.play("x90")
                    
                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])

        with for_(t, 4, t < cryoscope_len // 4, t + 4):

            with for_(idx, 0, idx<16, idx+1):
                with for_(*from_array(frame, frames)):
                    # Initialize the qubits
                    if reset_type == "active":
                        for qubit in qubits:
                            active_reset(qubit)
                    else:
                        wait(qubit.thermalization_time * u.ns)
                    align()
                    # Play first X/2
                    for qubit in qubits:
                        qubit.xy.play("x90")
                    align()
                    # Delay between x90 and the flux pulse
                    wait(50)
                    align()
                    with switch_(idx):
                        for j in range(16):
                            with case_(j):
                                baked_signals[j].run() 
                                qubits[0].z.play('const', duration=t, amplitude_scale =amplitude_factor)

                    # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                    # pulse arrives after the longest flux pulse
                    for qubit in qubits:
                        qubit.xy.wait((cryoscope_len + 160) // 4)
                        # Play second X/2
                        frame_rotation_2pi(frame, qubit.xy.name)
                        qubit.xy.play("x90")

                    # Measure resonator state after the sequence
                    align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])

    with stream_processing():
        # for the progress counter
        n_st.save("iteration")
        for i, qubit in enumerate(qubits):
            state_st[i].buffer(len(frames)).buffer(cryoscope_len+4).average().save(f"state{i + 1}")


# %%

simulate =  node.parameters.simulate

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=50000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cryoscope, simulation_config)
    samples = job.get_simulated_samples()
    samples.con4.plot()
    plt.show()
    samples.con5.plot()
    plt.show()

elif node.parameters.load_data_id is None:
    with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
        job = qm.execute(cryoscope)
        data_list = ["iteration"]
        results = fetching_tool(job, data_list, mode="live")

        while results.is_processing():
            fetched_data = results.fetch_all()
            n = fetched_data[0]
            progress_counter(n, n_avg, start_time=results.start_time)


# %%

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is None:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, [qubit], {"frame": frames, "time": cryoscope_time})
        plot_process = True
        node.results['ds'] = ds
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        
        
# %%
# Find phase of sine for each time step by fitting
def extract_phase(ds):
    phases = []
    for q in range(len(qubits)):
        phase_q = []
        for i in range(len(ds.state[q,:])):
            # Get data for this time step
            y_data = ds.state[q,i,:]
            x_data = ds.frame
            
            # Fit sine wave to get phase
            def sine_fit(x, phase, A, offset):
                return A * np.sin(2*np.pi*x + phase) + offset
                
            popt, _ = curve_fit(sine_fit, x_data, y_data, p0=[0, 1, 0.5], bounds=([-np.pi, 0, -np.inf], [np.pi, np.inf, np.inf]))
            # plt.plot(x_data, y_data,'.')
            # plt.plot(x_data, sine_fit(x_data, *popt))
            plt.show()
            phase_q.append(popt[0])
        phases.append(np.unwrap(phase_q))
    return ds.assign_coords(phase=(['qubit', 'time'], phases))

def extract_freqs(ds):
    freqs = ds.phase.diff('time') / ds.time.diff('time') / (2*np.pi)
    # freqs = savgol_filter(freqs, window_length=5, polyorder=2)
    return ds.assign_coords(frequencies=1e3*freqs)

def extract_flux(ds):
    fluxes = []
    for qubit in qubits:
        fluxes.append(np.sqrt(np.abs(-1e6*ds.sel(qubit = qubit.name).frequencies / qubit.freq_vs_flux_01_quad_term)))
    return ds.assign_coords(flux=(['qubit', 'time'], fluxes))

ds = extract_phase(ds)
ds = extract_freqs(ds)
ds = extract_flux(ds)

# %%
ds.state.sel(frame = 0).plot()
node.results['figure1'] = plt.gcf()
plt.show()

ds.phase.plot()
node.results['figure2'] = plt.gcf()
plt.show()

ds.frequencies.plot()
node.results['figure3'] = plt.gcf()
plt.show()

ds.flux.plot()
node.results['figure4'] = plt.gcf()
plt.show()

# %%
if not node.parameters.simulate:
        # extract the rising part of the data for analysis
    flux_cryoscope_q = ds.flux.sel(qubit = qubit.name)
    threshold = flux_cryoscope_q.max().values*0.6 # Set the threshold value
    rise_index = np.argmax(flux_cryoscope_q.values > threshold) + 1
    drop_index =  len(flux_cryoscope_q) - 0
    flux_cryoscope_tp = flux_cryoscope_q.sel(time=slice(rise_index,drop_index ))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
        time=flux_cryoscope_tp.time - rise_index + 1)


    f,axs = plt.subplots(2)
    flux_cryoscope_q.plot(ax = axs[0])
    axs[0].axvline(rise_index, color='r', lw = 0.5, ls = '--')
    axs[0].axvline(drop_index, color='r', lw = 0.5, ls = '--')
    flux_cryoscope_tp.plot(ax = axs[1])
    node.results['figure5'] = f
    plt.show()
    
    filtered_flux_cryoscope_q = savgol(ds.flux, 'time', range = 3, order = 2)
    da = flux_cryoscope_tp

    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(-20, None)).mean().values    
    
# %%
if not node.parameters.simulate and not node.parameters.only_FIR: # exponential fit is only done when only_FIR is False
    # Fit two exponents
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this

    
    try:
        p0 = [final_vals, -1+first_vals/final_vals, 50]
        fit, _  = curve_fit(expdecay, da.time[5:], da[5:],
                p0=p0)
    except:
        fit = p0
        print('single exp fit failed')
        
    if plot_process:
        da.plot(marker = '.')
        # plt.plot(filtered_flux_cryoscope_q.time, filtered_flux_cryoscope_q, label = 'filtered')
        plt.plot(da.time, expdecay(da.time, *fit), label = 'fit single exp')
        fit_text = f's={fit[0]:.6f}\na={fit[1]:.6f}\nt={fit[2]:.6f}'
        plt.text(0.02, 0.98, fit_text, transform=plt.gca().transAxes, verticalalignment='top', fontsize=8)
        plt.legend()
        plt.show()

    # Print fit parameters nicely (expdecay function)
    print("\nFit parameters (expdecay function):")
    print(f"s: {fit[0]:.6f}")
    print(f"a: {fit[1]:.6f}")
    print(f"t: {fit[2]:.6f}")

    from qualang_tools.digital_filters import exponential_decay, single_exponential_correction, bounce_and_delay_correction, calc_filter_taps
    
    exponential_filter = list(zip([fit[1]*1.0],[fit[2]]))
    feedforward_taps_1exp, feedback_tap_1exp = calc_filter_taps(exponential=exponential_filter)
    FIR_1exp = feedforward_taps_1exp
    IIR_1exp = [1,-feedback_tap_1exp[0]]
    flux_cryoscope_q[0] = 0
    filtered_response_long_1exp = lfilter(FIR_1exp,IIR_1exp, flux_cryoscope_q[1:])
    
    if plot_process:
        f,ax = plt.subplots()
        ax.plot(flux_cryoscope_q.time[1:],flux_cryoscope_q[1:],label = 'data')
        ax.plot(flux_cryoscope_q.time[1:],filtered_response_long_1exp,label = 'filtered long time 1exp')
        ax.set_ylim([final_vals*0.9,final_vals*1.05])
        ax.legend()
        plt.show()
        node.results['figure6'] = plt.gcf()
        plt.show()



# %%
if not node.parameters.simulate:
    ####  FIR filter for the response
    if node.parameters.only_FIR:
        response_long = flux_cryoscope_q[1]
    else:
        response_long = filtered_response_long_1exp
        long_FIR = FIR_1exp
        long_IIR = IIR_1exp
    
    flux_q = flux_cryoscope_q[1:].copy()
    flux_q.values = response_long
    flux_q_tp = flux_q.sel(time=slice(rise_index, 200)) # calculate the FIR only based on the first 200 nS
    flux_q_tp = flux_q_tp.assign_coords(
        time=flux_q_tp.time - rise_index)
    final_vals = flux_q_tp.sel(time=slice(-20, None)).mean().values
    step = np.ones(len(flux_q)+100)*final_vals
    fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 24)

    if node.parameters.only_FIR:
        convolved_fir = fir_est
        long_IIR = [1]
    else:
        convolved_fir = convolve(long_FIR,fir_est, mode='full')

    filtered_response_Full = lfilter(convolved_fir,long_IIR, flux_cryoscope_q[1:])

    if plot_process:
        flux_cryoscope_q.plot(label =  'data')
        plt.plot(flux_cryoscope_q.time[1:],filtered_response_long_1exp, label = 'filtered long time')
        plt.plot(flux_cryoscope_q.time[1:],filtered_response_Full, label = 'filtered full, deconvolved')
        plt.axhline(final_vals*1.001, color = 'k')
        plt.axhline(final_vals*0.999, color = 'k')
        plt.ylim([final_vals*0.95,final_vals*1.05])
        plt.legend()
        plt.show()



# %%
# if not node.parameters.simulate:
#     def find_diff(x, y, y0, plot = False):
#         filterd_y  = lfilter(x,[1,0], y)
#         diffs = np.sum(np.abs(filterd_y - y0))
#         if plot:
#             plt.plot(filterd_y)
#         return diffs

#     result = minimize(find_diff, x0=fir_est, args = (response_long[:100],np.mean(response_long[rise_index+50:drop_index])),
#                       bounds = [(-3,3)]*len(fir_est))

#     optimized_fir = result.x
#     convolved_fir = convolve(long_FIR,optimized_fir, mode='full')
#     if np.abs(np.max(convolved_fir)) > 2:
#         convolved_fir=1.99*convolved_fir/np.max(np.abs(convolved_fir))
#     filtered_response_Full = lfilter(convolved_fir,long_IIR, flux_cryoscope_q[1:])

#     if plot_process:
#         flux_cryoscope_q.plot(label =  'data')
#         plt.plot(flux_cryoscope_q.time[1:],filtered_response_long_1exp, label = 'filtered long time')
#         plt.plot(flux_cryoscope_q.time[1:],filtered_response_Full, label = 'filtered full, fitted')
#         plt.axhline(final_vals*1.001, color = 'k')
#         plt.axhline(final_vals*0.999, color = 'k')
#         plt.ylim([final_vals*0.95,final_vals*1.05])
#         plt.legend()
#         plt.show()

# %%
if not node.parameters.simulate:
    # plotting the results
    fig,ax = plt.subplots()
    ax.plot(flux_cryoscope_q.time[1:],flux_cryoscope_q[1:]/np.mean(flux_cryoscope_q[1:][-10:]),label = 'data')
    ax.plot(flux_cryoscope_q.time[1:],filtered_response_long_1exp/np.mean(filtered_response_long_1exp[-10:]),'--', label = 'slow rise correction')
    ax.plot(flux_q.time,filtered_response_Full/np.mean(filtered_response_long_1exp[-10:]),'--', label = 'expected corrected response')
    ax.axhline(1.001, color = 'k')
    ax.axhline(0.999, color = 'k')
    ax.set_ylim([0.95,1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('normalized amplitude')
    node.results['figure6'] = fig
# elif not node.parameters.simulate:
#     fig,ax = plt.subplots()
#     ax.plot(flux_cryoscope_q.time,flux_cryoscope_q/np.mean(flux_cryoscope_q[-50:]),label = 'data')
#     ax.axhline(1.001, color = 'k')
#     ax.axhline(0.999, color = 'k')
#     ax.set_ylim([0.95,1.05])
#     ax.set_xlabel('time (ns)')
#     ax.set_ylabel('normalized amplitude')    
#     ax.legend()
#     node.results['figure7'] = fig
# %%
if not node.parameters.simulate:
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['fir'] = optimized_fir.tolist()
        if not node.parameters.only_FIR:
            node.results['fit_results'][q.name]['iir'] = feedback_tap_1exp

# %%
convert_to_2GSPS = True
if not node.parameters.simulate:
    with node.record_state_updates():
        for qubit in qubits:
            if convert_to_2GSPS:
                fir_list = fir_est.tolist()
                fir_list = [element for item in fir_list for element in (item, item)]
            else:
                fir_list = fir_est.tolist() 
            qubit.z.opx_output.feedforward_filter = fir_list
            if not node.parameters.only_FIR:
                qubit.z.opx_output.exponential_filter = [*qubit.z.opx_output.exponential_filter, *exponential_filter]

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
save_node(node)
# %%