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
    amplitude_factor: float = 0.3
    cryoscope_len: int = 36
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    analysis_mode: Literal[ 'FIR', 'none'] = 'none'
    use_FIR: bool = True
    load_data_id: Optional[int] = None
    
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
        
# Generate the OPX and Octave configurations
config = machine.generate_config()
# octave_config = machine.get_octave_config()
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

def Predistorion(waveform, FIR_coefficients):
    return np.convolve(waveform, FIR_coefficients, mode='full')

def generate_cryoscope_config(config, amplitude, FIR_coefficients = None):
    
    length = node.parameters.cryoscope_len
    time_step = 1
    times = np.arange(0, length, time_step)
    full_waveform = (times > 4 ) 

    for i in range(length):
        waveform = [0] * length

        for qubit in qubits:
            if node.parameters.use_FIR:
                full_waveform_q = Predistorion(full_waveform, qubit.z.extras['feedforward_filter'])
            else:
                full_waveform_q = full_waveform
            for j in range(i+1):
                waveform[j] = full_waveform_q[j] * qubit.z.operations['const'].amplitude * amplitude
            config["pulses"][f"{qubit.name}.z_pulse_{i}"] = {"operation": "control", "length" : length, "waveforms" : {"single" : f"{qubit.name}.z_waveform_{i}"}}
            config["waveforms"][f"{qubit.name}.z_waveform_{i}"] = {"type" : "arbitrary", "samples" : waveform}
            config["elements"][f"{qubit.name}.z"]["operations"][f"z_pulse_{i}"] = f"{qubit.name}.z_pulse_{i}"
    return config
    

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

assert cryoscope_len % 4 == 0, 'cryoscope_len is not multiple of 16 nanoseconds'

config = generate_cryoscope_config(config, node.parameters.amplitude_factor)
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

        with for_(idx, 0, idx<cryoscope_len, idx+1):
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
                    qubit.xy.play("x90")
                align()
                # Delay between x90 and the flux pulse
                # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
                wait(16 // 4)
                align()
                with switch_(idx):
                    for j in range(cryoscope_len):
                        with case_(j):
                            play(f"z_pulse_{j}", qubit.z.name)
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
                                           stable_time_indices=(20, cryoscope_len),  
                                           sg_order=sg_order,
                                           sg_range=sg_range, plot=plot_process)
    plt.show()

# %%
if not node.parameters.simulate:
        # extract the rising part of the data for analysis
    threshold = flux_cryoscope_q.max().values*0.6 # Set the threshold value
    rise_index = np.argmax(flux_cryoscope_q.values > threshold) + 0
    drop_index =  len(flux_cryoscope_q) - 4
    flux_cryoscope_tp = flux_cryoscope_q.sel(time=slice(rise_index,drop_index ))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
        time=flux_cryoscope_tp.time - rise_index + 1)


    f,axs = plt.subplots(2)
    flux_cryoscope_q.plot(ax = axs[0])
    axs[0].axvline(rise_index, color='r', lw = 0.5, ls = '--')
    axs[0].axvline(drop_index, color='r', lw = 0.5, ls = '--')
    flux_cryoscope_tp.plot(ax = axs[1])
    node.results['figure'] = f
    plt.show()
    
# %%
if not node.parameters.simulate:
    # Fit two exponents
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this
    filtered_flux_cryoscope_q = savgol(flux_cryoscope_tp, 'time', range = 3, order = 2)
    da = flux_cryoscope_tp

    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(-2, None)).mean().values


# %%
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    ####  FIR filter for the response

    
    flux_q = flux_cryoscope_q.copy()
    # flux_q.values = filtered_response_long
    flux_q_tp = flux_q.sel(time=slice(rise_index, None)) # calculate the FIR only based on the first 200 nS
    flux_q_tp = flux_q_tp.assign_coords(
        time=flux_q_tp.time - rise_index)
    final_vals = flux_q_tp.sel(time=slice(-2, None)).mean().values
    step = np.ones(len(flux_q)+100)*final_vals
    fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 60)

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


# opx1K  QOP 3.3
if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['fir'] = optimized_fir.tolist()

# %%


if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.extras['feedforward_filter'] = node.results['fit_results'][qubit.name]['fir']

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
# Create step function
step_len = len(optimized_fir) * 2
step = np.heaviside(np.arange(step_len) - step_len//4, 0.5)

# Convolve with FIR taps and plot
response = np.convolve(step, optimized_fir, mode='full')
plt.figure()
plt.plot(response, '.-', label='Simulated response')
plt.plot(step[:len(response)], '--', label='Input step')
plt.legend()
plt.xlabel('Time [ns]')
plt.ylabel('Amplitude [a.u.]')
plt.title('Step response with optimized FIR taps')
plt.grid(True)
plt.show()

# # %%
# optimized_fir
# # %%
# from qm import QuantumMachinesManager, SimulationConfig

# qm = qmm.open_qm(config)

# job = qm.simulate(cryoscope, SimulationConfig(5000))
# job.wait_until("Done", timeout=1000)

# job.plot_waveform_report_with_simulated_samples()
# # %%
# wfr = job.get_simulated_waveform_report()
# # %%
# %%

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as fft
from qualang_tools.units import unit
from scipy import signal, optimize
from scipy.optimize import curve_fit

measuredData = flux_cryoscope_q.values
measuredData = measuredData[1:,]  # exclude first point
# I = measuredData[:,[0]]
sso = measuredData/0.03


# %%
readout_len = 2000  # ns
const_flux_len = len(sso)  # ns

times = np.arange(-const_flux_len, const_flux_len, 1)

step_response_volt = np.append(np.zeros(const_flux_len), sso)
step_response_volt_offset = np.mean(step_response_volt)
step_response_volt -= step_response_volt_offset 
# %%
linewidth = 4
measured_color = 'blue'
plt.subplot(111)

ax = plt.gca()
ax.tick_params(direction='in', length=6, width=2, colors='black',
              grid_color='black', grid_alpha=1,bottom=True, top=True, left=True, right=True)
t = ax.yaxis.get_offset_text()
t.set_size(14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width


plt.plot(times, step_response_volt,color=measured_color,linewidth=linewidth, label=r"Measured Response")
plt.xlabel("Time [ns]",fontsize=17)
plt.ylabel("Step response [V]",fontsize=17)
plt.legend(fontsize=15, loc = 'lower right')
# plt.xlim((-200,64))

plt.tight_layout()

plt.show()
# %%
n=9
dt_fine = 1
times_fine = np.linspace(-256,255,2**n)
print(times_fine)
step_response_fine = np.interp(times_fine,times,step_response_volt)
step_response_volt = step_response_fine #should go through and eliminate this

impulse_response_fine = np.gradient(step_response_fine,times_fine)
impulse_response_fine *= np.heaviside(times_fine+dt_fine,0.5)
# %%
linewidth = 4
measured_color = 'red'
plt.subplot(111)

ax = plt.gca()
ax.tick_params(direction='in', length=6, width=2, colors='black',
              grid_color='black', grid_alpha=1,bottom=True, top=True, left=True, right=True)
t = ax.yaxis.get_offset_text()
t.set_size(14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width


plt.plot(times_fine, impulse_response_fine,color=measured_color,linewidth=linewidth, label=r"Impulse Response")
plt.xlabel("Pulse duration [ns]",fontsize=17)
plt.ylabel("Step response",fontsize=17)
plt.legend(fontsize=15, loc = 'upper right')
plt.xlim((-200,200))

plt.tight_layout()

plt.show()
# %%
heaviside = np.heaviside(times_fine,0.5)-0.5

convolution = signal.fftconvolve(impulse_response_fine,heaviside,'same')

scale = (np.mean(step_response_fine[-len(step_response_fine)//32:]))/(np.mean(convolution[-len(convolution)//32:]))

step_from_impulse_offset = 0.5
step_from_impulse = scale*convolution
# %%
linewidth = 4
measured_color = 'blue'
reconstructed_color = 'orange'
plt.subplot(111)

ax = plt.gca()
ax.tick_params(direction='in', length=6, width=2, colors='black',
              grid_color='black', grid_alpha=1,bottom=True, top=True, left=True, right=True)
t = ax.yaxis.get_offset_text()
t.set_size(14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width


plt.plot(times_fine, step_response_volt+step_response_volt_offset,color=measured_color,linewidth=linewidth, label=r"Measured Response")
plt.plot(times_fine,step_from_impulse+step_from_impulse_offset,color=reconstructed_color,linewidth=linewidth,label="Reconstructed Response")

plt.xlabel("Time [ns]",fontsize=17)
plt.ylabel("Step response [V]",fontsize=17)
plt.legend(fontsize=15, loc = 'lower right')

plt.tight_layout()
plt.xlim((-200,200))

plt.show()

# %%
impulse_response_fine_tilde = np.fft.fftshift(np.fft.fft(impulse_response_fine))
heaviside_tilde = np.fft.fftshift(np.fft.fft(heaviside))
convolution = np.fft.ifft(np.fft.ifftshift(impulse_response_fine_tilde*np.conj(heaviside_tilde)))[0:len(times_fine)]
# %%
linewidth = 4
measured_color = 'blue'
reconstructed_color = 'orange'
reconstructed_fft = 'darkorange'
plt.subplot(111)

ax = plt.gca()
ax.tick_params(direction='in', length=6, width=2, colors='black',
              grid_color='black', grid_alpha=1,bottom=True, top=True, left=True, right=True)
t = ax.yaxis.get_offset_text()
t.set_size(14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width


plt.plot(times_fine,step_from_impulse+step_from_impulse_offset,color=reconstructed_color,linewidth=linewidth,label="Reconstructed Response (Convolve)")
plt.plot(times_fine,scale*convolution+step_from_impulse_offset,'o',color=reconstructed_fft,label="Reconstructed Response (FFT)")


plt.xlabel("Time [ns]",fontsize=17)
plt.ylabel("Step response [V]",fontsize=17)
plt.legend(fontsize=15, loc = 'lower right')
plt.xlim((-200,200))

plt.tight_layout()

plt.show()

# %%
#don't mess around with any fftshifts...
# heaviside = np.append(np.zeros(len(times_fine)//2),np.ones(len(times_fine)//2+1))
heaviside = np.heaviside(times_fine,0.5)-0.5
impulse_response_fine_tilde = (np.fft.fft(impulse_response_fine))
heaviside_tilde = (np.fft.fft(heaviside))
predistorted_tilde = heaviside_tilde/impulse_response_fine_tilde/scale
predistorted = -np.fft.ifft((predistorted_tilde))
predistort_out = np.fft.ifft(predistorted_tilde*impulse_response_fine_tilde*scale)
# %%
linewidth = 4
measured_color = 'blue'
reconstructed_color = 'orange'
reconstructed_fft = 'darkorange'
predistorted_color = 'green'
predistorted_response_color = 'darkgreen'
plt.subplot(111)

ax = plt.gca()
ax.tick_params(direction='in', length=6, width=2, colors='black',
              grid_color='black', grid_alpha=1,bottom=True, top=True, left=True, right=True)
t = ax.yaxis.get_offset_text()
t.set_size(14)

plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

for axis in ['top', 'bottom', 'left', 'right']:
  ax.spines[axis].set_linewidth(2)  # change width

plt.plot(times_fine,predistorted,color=predistorted_color,linewidth=linewidth,label="Predistorted Pulse")
plt.plot(times_fine,predistort_out,'o',color=predistorted_response_color,label="Predistorted Response")

plt.xlabel("Time [ns]",fontsize=17)
plt.ylabel("Step response [V]",fontsize=17)
plt.legend(fontsize=15, loc = 'lower right')
plt.xlim((-200,200))
plt.tight_layout()

plt.show()

#To input to OPX
print([float('{:.3f}'.format(pt)) if pt > 0.1 else 0.0 for pt in (predistorted.real-np.mean((predistorted.real)[0:len(predistorted)//2-1]))][len(predistorted)//2:])
# %%


if not node.parameters.simulate and node.parameters.analysis_mode == 'FIR':
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.extras['feedforward_filter'] = np.real(predistorted).tolist()

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
# node.save()

# %%
