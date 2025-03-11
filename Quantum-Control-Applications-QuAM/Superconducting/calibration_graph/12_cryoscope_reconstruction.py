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

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
import xarray as xr
from scipy.optimize import curve_fit, minimize
from scipy.signal import deconvolve, lfilter, convolve
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List
from quam_libs.lib.cryoscope_tools import cryoscope_frequency, estimate_fir_coefficients, two_expdecay, expdecay, savgol


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ['qubitC2']    
    num_averages: int = 40000
    amplitude_factor: float = 0.25
    cryoscope_len: int = 64
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    reset_filters: bool = False
    analysis_mode: Literal['exponential', 'FIR', 'none'] = 'FIR'
    load_data_id: Optional[int] = 325
    
node = QualibrationNode(
    name="12_Cryoscope",
    parameters=Parameters()
)

plot_process = False



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
    
if node.parameters.reset_filters:
    for qubit in qubits:
        qubit.z.opx_output.feedforward_filter = [1.0, 0.0]
        qubit.z.opx_output.exponential_filter = [(0.0, 10.0)]
        # qubit.z.opx_output.feedback_filter = [0.0, 0.0]
                
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

cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

# %%

with program() as cryoscope:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    t = declare(int)  # QUA variable for the flux pulse segment index
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    global_state = declare(int)
    idx = declare(int)
    idx2 = declare(int)
    flag = declare(bool)
    qubit = qubits[0]
    i = 0
    
    # Bring the active qubits to the desired frequency point
    machine.set_all_fluxes(flux_point=flux_point, target=qubit)

    # Outer loop for averaging
    with for_(n, 0, n < n_avg, n + 1):
        save(n, n_st)

        # The first 16 nanoseconds
        with for_(idx, 0, idx<16, idx+1):
            # Alternate between X/2 and Y/2 pulses
            # for tomo in ['x90', 'y90']:
            with for_each_(flag, [True, False]):
                if reset_type == "active":
                    for qubit in qubits:
                        active_reset(qubit)
                else:
                    wait(qubit.thermalization_time * u.ns)
                align()
                # Play first X/2
                for qubit in qubits:
                    qubit.xy.play("x180", amplitude_scale = 0.5)
                align()
                # Delay between x90 and the flux pulse
                # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
                wait(16 // 4)
                align()
                # with switch_(idx):
                #     for j in range(16):
                #         with case_(j):
                #             baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_len + 16) // 4)
                    # Play second X/2 or Y/2
                    # if tomo == 'x90':
                    with if_(flag):
                        qubit.xy.play("x90")
                    # elif tomo == 'y90':
                    with else_():
                        qubit.xy.play("y90")
                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])


        # The first 16-32 nanoseconds
        with for_(idx, 0, idx<16, idx+1):
            # Alternate between X/2 and Y/2 pulses
            # for tomo in ['x90', 'y90']:
            with for_each_(flag, [True, False]):
                if reset_type == "active":
                    for qubit in qubits:
                        active_reset(qubit)      
                else:
                    wait(qubit.thermalization_time * u.ns)
                align()
                # Play first X/2
                for qubit in qubits:
                    qubit.xy.play("x180", amplitude_scale = 0.5)
                align()
                # Delay between x90 and the flux pulse
                # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
                wait(16 // 4)
                align()
                with switch_(idx):
                    for j in range(16):
                        with case_(j):
                            baked_signals[j].run()
                # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                # pulse arrives after the longest flux pulse
                for qubit in qubits:
                    qubit.xy.wait((cryoscope_len + 16) // 4)
                    # Play second X/2 or Y/2
                    # if tomo == 'x90':
                    with if_(flag):
                        qubit.xy.play("x90")
                    # elif tomo == 'y90':
                    with else_():
                        qubit.xy.play("y90")
                # Measure resonator state after the sequence
                align()
                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                save(state[i], state_st[i])

        with for_(t, 8, t < cryoscope_len // 4, t + 4):

            with for_(idx, 0, idx<16, idx+1):

                # Alternate between X/2 and Y/2 pulses
                # for tomo in ['x90', 'y90']:
                with for_each_(flag, [True, False]):
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
                    wait(16 // 4)
                    align()
                    with switch_(idx):
                        for j in range(16):
                            with case_(j):
                                baked_signals[j].run() 
                                qubits[0].z.play('const', duration=t-4, amplitude_scale =amplitude_factor)

                    # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
                    # pulse arrives after the longest flux pulse
                    for qubit in qubits:
                        qubit.xy.wait((cryoscope_len + 16) // 4)
                        # Play second X/2 or Y/2
                        with if_(flag):
                            qubit.xy.play("x90")
                        # elif tomo == 'y90':
                        with else_():
                            qubit.xy.play("y90")

                    # Measure resonator state after the sequence
                    align()
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])

    with stream_processing():
        # for the progress counter
        n_st.save("iteration")
        for i, qubit in enumerate(qubits):
            state_st[i].buffer(2).buffer(cryoscope_len).average().save(f"state{i + 1}")


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
        ds = fetch_results_as_xarray(job.result_handles, [qubit], {"axis": ["x","y"], "time": cryoscope_time})
        plot_process = True
        node.results['ds'] = ds
    else:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
# %%
if not node.parameters.simulate:
    if plot_process:
        ds.state.sel(qubit= qubits[0].name).plot(hue = 'axis')
        plt.show()

    sg_order = 2
    sg_range = 3
    qubit = qubits[0]
    da = ds.state.sel(qubit= qubit.name)

    flux_cryoscope_q = cryoscope_frequency(da, 
                                           quad_term=qubit.freq_vs_flux_01_quad_term,
                                           stable_time_indices=(20, cryoscope_len-20),  
                                           sg_order=sg_order,
                                           sg_range=sg_range, plot=plot_process)
    plt.show()

# %%
if not node.parameters.simulate:
        # extract the rising part of the data for analysis
    threshold = flux_cryoscope_q.max().values*0.6 # Set the threshold value
    rise_index = np.argmax(flux_cryoscope_q.values > threshold) + 0
    drop_index =  len(flux_cryoscope_q) - 1
    flux_cryoscope_tp = flux_cryoscope_q.sel(time=slice(rise_index,drop_index ))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
        time=flux_cryoscope_tp.time - rise_index + 1)


    f,axs = plt.subplots(2)
    flux_cryoscope_q.plot(ax = axs[0], marker = '.')
    axs[0].axvline(rise_index, color='r', lw = 0.5, ls = '--')
    axs[0].axvline(drop_index, color='r', lw = 0.5, ls = '--')
    flux_cryoscope_tp.plot(ax = axs[1], marker = '.')
    node.results['figure'] = f
    plt.show()
    
# %%
if not node.parameters.simulate:
    # Fit two exponents
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this
    filtered_flux_cryoscope_q = savgol(flux_cryoscope_tp, 'time', range = 11, order = 3)
    da = filtered_flux_cryoscope_q

    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(40, None)).mean().values



# %%
# pad the response with zeros before the first values
fluxes = filtered_flux_cryoscope_q.values
times = filtered_flux_cryoscope_q.time.values
negative_times = -times[::-1]
fluxes_with_zeros = np.concatenate([np.zeros(len(negative_times)), fluxes])
times_with_zeros = np.concatenate([negative_times, times])
fluxes_with_zeros /= np.max(fluxes_with_zeros)
fluxes_with_zeros -= np.mean(fluxes_with_zeros)
flux_cryoscope_with_zeros = xr.DataArray(fluxes_with_zeros, dims = 'time', coords = {'time': times_with_zeros})
flux_cryoscope_with_zeros.plot(marker = '.')

# %%
# sample the response at times insipe a power of 2
max_time = times_with_zeros[-1]
nearest_power_of_2 = int(2**(np.ceil(np.log2(max_time))))  

new_times = np.linspace(-nearest_power_of_2, nearest_power_of_2-1, 2*nearest_power_of_2)
flux_cryoscope_resampled = np.interp(new_times, flux_cryoscope_with_zeros.time, flux_cryoscope_with_zeros)
flux_cryoscope_resampled = xr.DataArray(flux_cryoscope_resampled, dims = 'time', coords = {'time': new_times})
flux_cryoscope_resampled.plot(marker = '.')

# %%
# Finde the impulse response
impulse_response = np.gradient(flux_cryoscope_resampled, flux_cryoscope_resampled.time)
impulse_response *= np.heaviside(new_times +1 , 0.5)
impulse_response = xr.DataArray(impulse_response, dims = 'time', coords = {'time': new_times})
impulse_response.plot(marker = '.')

# %% Step response reconsculred from the impulse response
import scipy.signal as signal

heaviside = np.heaviside(impulse_response.time,0.5)-0.5
# convolution = signal.fftconvolve(impulse_response,heaviside,'same')
# plt.plot(impulse_response.time,heaviside)
# plt.plot(impulse_response.time,convolution)

impulse_response_tilde = np.fft.fftshift(np.fft.fft(impulse_response))
heaviside_tilde = np.fft.fftshift(np.fft.fft(heaviside))
convolution = np.fft.ifft(np.fft.ifftshift(impulse_response_tilde*np.conj(heaviside_tilde)))[0:len(impulse_response.time)]
plt.plot(impulse_response.time,convolution)


# %%
heaviside = np.heaviside(impulse_response.time,0.5)-0.5
impulse_response_tilde = (np.fft.fft(impulse_response))
heaviside_tilde = (np.fft.fft(heaviside))
# plt.plot(impulse_response.time,impulse_response)
predistorted_tilde = heaviside_tilde/impulse_response_tilde
# predistorted_tilde[0:1] = 0
predistorted_tilde[40:84] = 0
# plt.plot(impulse_response.time,predistorted_tilde)

predistorted = -np.fft.ifft((predistorted_tilde))
predistort_out = np.fft.ifft(predistorted_tilde*impulse_response_tilde)
plt.plot(impulse_response.time,predistort_out,'.')
plt.plot(impulse_response.time,predistorted)


# %%
heaviside = np.heaviside(impulse_response.time,0.5) * np.heaviside(-(impulse_response.time-10),0.5) -0*np.heaviside(impulse_response.time-20,0.5) * np.heaviside(-(impulse_response.time-30),0.5) 
plt.plot(impulse_response.time,heaviside)
impulse_response_tilde = (np.fft.fft(impulse_response))
heaviside_tilde = (np.fft.fft(heaviside))
predistorted_tilde = heaviside_tilde/impulse_response_tilde
# predistorted_tilde[0:1] = 0
predistorted_tilde[40:84] = 0
# plt.plot(impulse_response.time,predistorted_tilde)

predistorted = -np.fft.ifft((predistorted_tilde))
predistort_out = np.fft.ifft(predistorted_tilde*impulse_response_tilde)
plt.plot(impulse_response.time,predistort_out,'.')
plt.plot(impulse_response.time,predistorted)













# %%

# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'exponential':
    # Fit two exponents
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this
    filtered_flux_cryoscope_q = savgol(flux_cryoscope_tp, 'time', range = 3, order = 2)
    da = flux_cryoscope_tp

    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(50, None)).mean().values

    try:
        p0 = [final_vals, -1+first_vals/final_vals, 4]
        fit, _  = curve_fit(expdecay, da.time[5:], da[5:],
                p0=p0,
                bounds=([0, -np.inf, 0], [np.inf, np.inf, 100]))
    except:
        fit = p0
        print('single exp fit failed')
    try:
        p0 = [fit[0], fit[1], 5, fit[1], fit[2]]
        fit2, _ = curve_fit(two_expdecay, filtered_flux_cryoscope_q.time[4:], filtered_flux_cryoscope_q[4:],
                p0 = p0)
    except:
        fit2 = p0
        print('two exp fit failed')
        
    if plot_process:
        da.plot(marker = '.')
        # plt.plot(filtered_flux_cryoscope_q.time, filtered_flux_cryoscope_q, label = 'filtered')
        plt.plot(da.time, expdecay(da.time, *fit), label = 'fit single exp')
        if fit2 is not None:
            plt.plot(da.time, two_expdecay(da.time, *fit2), label = 'fit two exp')
        plt.legend()
        plt.show()

    # Print fit2 parameters nicely (two_expdecay function)
    if fit2 is not None:
        print("Fit2 parameters (two_expdecay function):")
        print(f"s: {fit2[0]:.6f}")
        print(f"a: {fit2[1]:.6f}")
        print(f"t: {fit2[2]:.6f}")
        print(f"a2: {fit2[3]:.6f}")
        print(f"t2: {fit2[4]:.6f}")

    # Print fit parameters nicely (expdecay function)
    print("\nFit parameters (expdecay function):")
    print(f"s: {fit[0]:.6f}")
    print(f"a: {fit[1]:.6f}")
    print(f"t: {fit[2]:.6f}")

# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'exponential':

    from qualang_tools.digital_filters import exponential_decay, single_exponential_correction, bounce_and_delay_correction, calc_filter_taps

    feedforward_taps_1exp, feedback_tap_1exp = calc_filter_taps(exponential=list(zip([fit[1]*1.0],[fit[2]])))
    feedforward_taps_2exp, feedback_tap_2exp = calc_filter_taps(exponential=list(zip([fit2[1],fit2[3]],[fit2[2],fit2[4]])))

    FIR_1exp = feedforward_taps_1exp
    FIR_2exp = feedforward_taps_2exp
    IIR_1exp = [1,-feedback_tap_1exp[0]]
    IIR_2exp = convolve([1,-feedback_tap_2exp[0]],[1,-feedback_tap_2exp[1]], mode='full')
    
    filtered_response_long_1exp = lfilter(FIR_1exp,IIR_1exp, flux_cryoscope_q)
    filtered_response_long_2exp = lfilter(FIR_2exp,IIR_2exp, flux_cryoscope_q)

    if plot_process:
        f,ax = plt.subplots()
        ax.plot(flux_cryoscope_q.time,flux_cryoscope_q,label = 'data')
        ax.plot(flux_cryoscope_q.time,filtered_response_long_1exp,label = 'filtered long time 1exp')
        ax.plot(flux_cryoscope_q.time,filtered_response_long_2exp,label = 'filtered long time 2exp')
        ax.set_ylim([final_vals*0.9,final_vals*1.05])
        ax.legend()
        plt.show()

    fitting_approach = '1exp'
    if fitting_approach == '1exp':
        filtered_response_long = filtered_response_long_1exp
        long_FIR = FIR_1exp
        long_IIR = IIR_1exp
    elif fitting_approach == '2exp':
        filtered_response_long = filtered_response_long_2exp
        long_FIR = FIR_2exp
        long_IIR = IIR_2exp


# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    ####  FIR filter for the response

    
    flux_q = flux_cryoscope_q.copy()
    # flux_q.values = filtered_response_long
    flux_q_tp = flux_q.sel(time=slice(rise_index, None)) # calculate the FIR only based on the first 200 nS
    flux_q_tp = flux_q_tp.assign_coords(
        time=flux_q_tp.time - rise_index)
    final_vals = flux_q_tp.sel(time=slice(40, None)).mean().values
    step = np.ones(len(flux_q)+100)*final_vals
    fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 28)

    # FIR_new = fir_est

    # convolved_fir = convolve(long_FIR,FIR_new, mode='full')
    filtered_response_Full = lfilter(fir_est,[1,0], flux_cryoscope_q)

    if plot_process:
        flux_cryoscope_q.plot(label =  'data')
        # plt.plot(filtered_response_long, label = 'filtered long time')
        plt.plot(filtered_response_Full, label = 'filtered full, deconvolved')
        plt.axhline(final_vals*1.001, color = 'k')
        plt.axhline(final_vals*0.999, color = 'k')
        plt.ylim([final_vals*0.95,final_vals*1.05])
        plt.legend()
        plt.show()


# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    def find_diff(x, y, y0, plot = False):
        filterd_y  = lfilter(x,[1,0], y)
        diffs = np.sum(np.abs(filterd_y - y0))
        if plot:
            plt.plot(filterd_y)
        return diffs

    result = minimize(find_diff, x0=fir_est, args = (flux_cryoscope_q.values,np.mean(flux_cryoscope_q[rise_index+16:drop_index]).values),
                      bounds = [(-2,2)]*len(fir_est))

    optimized_fir = result.x

    filtered_response_optimized = lfilter(optimized_fir,[1,0], flux_cryoscope_q)

    if plot_process:
        flux_cryoscope_q.plot(label =  'data')
        plt.plot(filtered_response_Full, label = 'filtered long time')
        plt.plot(filtered_response_optimized, label = 'filtered full, fitted')
        plt.axhline(final_vals*1.001, color = 'k')
        plt.axhline(final_vals*0.999, color = 'k')
        plt.ylim([final_vals*0.95,final_vals*1.05])
        plt.legend()
        plt.show()

# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'exponential':
    # plotting the results
    fig,ax = plt.subplots()
    ax.plot(flux_cryoscope_q.time,flux_cryoscope_q/np.mean(flux_cryoscope_q[-50:]),label = 'data')
    ax.plot(flux_cryoscope_q.time,filtered_response_long/np.mean(filtered_response_long[-50:]),'--', label = 'slow rise correction')
    ax.axhline(1.001, color = 'k')
    ax.axhline(0.999, color = 'k')
    ax.set_ylim([0.95,1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('normalized amplitude')
    node.results['figure'] = fig
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    # plotting the results
    fig,ax = plt.subplots()
    ax.plot(flux_cryoscope_q.time,1e3*flux_cryoscope_q,label = 'data')
    ax.plot(flux_cryoscope_q.time,1e3*filtered_response_optimized, label = 'filtered full, fitted')
    ax.axhline(1e3*final_vals*1.001, color = 'k')
    ax.axhline(1e3*final_vals*0.999, color = 'k')
    ax.set_ylim([1e3*final_vals*0.95,1e3*final_vals*1.05])
    ax.legend()
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('flux amplitude (mV)')
    node.results['figure'] = fig    
elif not node.parameters.simulate:
    fig,ax = plt.subplots()
    ax.plot(flux_cryoscope_q.time,1e3*flux_cryoscope_q,label = 'data')
    ax.axhline(1e3*final_vals*1.001, color = 'k')
    ax.axhline(1e3*final_vals*0.999, color = 'k')
    ax.set_ylim([1e3*final_vals*0.95,1e3*final_vals*1.05])
    ax.set_xlabel('time (ns)')
    ax.set_ylabel('flux amplitude (mV)')    
    ax.legend()
    node.results['figure'] = fig
# %%
# if not node.parameters.simulate and node.parameters.reset_filters:
#     node.results['fit_results'] = {}
#     for q in qubits:
#         node.results['fit_results'][q.name] = {}
#         node.results['fit_results'][q.name]['fir'] = convolved_fir.tolist()
#         if fitting_approach == '1exp':
#             node.results['fit_results'][q.name]['iir'] = feedback_tap_1exp
#         elif fitting_approach == '2exp':
#             node.results['fit_results'][q.name]['iir']  = feedback_tap_2exp  


# opx1K  QOP 3.3
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['fir'] = optimized_fir.tolist()

if not node.parameters.simulate and node.parameters.analysis_mode == 'exponential':
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        if fitting_approach == '1exp':
            node.results['fit_results'][q.name]['exponential_filter'] = [[fit[1],fit[2]]]
        elif fitting_approach == '2exp':
            node.results['fit_results'][q.name]['exponential_filter']  = [[fit2[1],fit2[2]],[fit2[3],fit2[4]]]  



# %%

# if not node.parameters.simulate and node.parameters.reset_filters:
#     with node.record_state_updates():
#         for qubit in qubits:
#             qubit.z.opx_output.feedforward_filter = convolved_fir.tolist()
#             if fitting_approach == '1exp':
#                 qubit.z.opx_output.feedback_filter = feedback_tap_1exp
#             elif fitting_approach == '2exp':
#                 qubit.z.opx_output.feedback_filter = feedback_tap_2exp

# opx1K  QOP 3.3

if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.opx_output.feedforward_filter = node.results['fit_results'][qubit.name]['fir']

if not node.parameters.simulate and node.parameters.analysis_mode == 'exponential':
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.opx_output.exponential_filter = node.results['fit_results'][qubit.name]['exponential_filter']

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%

flux_cryoscope_tp.to_dataframe().to_csv('flux_cryoscope_tp.csv')

# %%
