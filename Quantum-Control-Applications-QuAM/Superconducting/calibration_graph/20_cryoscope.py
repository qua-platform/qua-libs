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

# matplotlib.use("TKAgg")

from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal
from quam_libs.lib.cryoscope_tools import cryoscope_frequency, estimate_fir_coefficients, two_expdecay, expdecay, savgol


class Parameters(NodeParameters):
    qubits: Optional[str] = 'q4'
    num_averages: int = 1000
    amplitude_factor: float = 0.5
    cryoscope_len: int = 160
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    reset_filters: bool = True

node = QualibrationNode(
    name="12_Cryoscope",
    parameters_class=Parameters
)

node.parameters = Parameters()

###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# machine = QuAM.load()
# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.split(', ')]
    
if node.parameters.reset_filters:
    for qubit in qubits:
        qubit.z.filter_fir_taps = [1,0]
    qubit.z.filter_iir_taps = [0]

            
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
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

###################
# The QUA program #
###################
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

    # Bring the active qubits to the minimum frequency point
    if flux_point == "independent":
        machine.apply_all_flux_to_min()
        qubit.z.to_independent_idle()
    elif flux_point == "joint":
        machine.apply_all_flux_to_joint_idle()
        # qubit.z.set_dc_offset(0.01 + qubit.z.joint_offset)
    else:
        machine.apply_all_flux_to_zero()
    wait(1000)

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
                        active_reset(machine, qubit.name)
                else:
                    wait(5*machine.thermalization_time * u.ns)
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
                #     for i in range(16):
                #         with case_(i):
                #             baked_signals[i].run()

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

                multiplexed_readout(qubits, I, I_st, Q, Q_st)

                for i in range(num_qubits):
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])

        with for_(t, 4, t < cryoscope_len // 4, t + 4):

            with for_(idx2, 0, idx2<16, idx2+1):

                # Alternate between X/2 and Y/2 pulses
                # for tomo in ['x90', 'y90']:
                with for_each_(flag, [True, False]):
                    if reset_type == "active":
                        for qubit in qubits:
                            active_reset(machine, qubit.name)
                    else:
                        wait(5*machine.thermalization_time * u.ns)
                    align()
                    # Play first X/2
                    for qubit in qubits:
                        qubit.xy.play("x90")

                    align()

                    # Delay between x90 and the flux pulse
                    wait(16 // 4)

                    align()
                    with switch_(idx2):
                        for j in range(16):
                            with case_(j):
                                baked_signals[j].run() 
                                qubits[0].z.play('const', duration=t, amplitude_scale =amplitude_factor)

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
                    multiplexed_readout(qubits, I, I_st, Q, Q_st)

                    for i in range(num_qubits):
                        assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                        save(state[i], state_st[i])

    with stream_processing():
        # for the progress counter
        n_st.save("iteration")
        for i, qubit in enumerate(qubits):
            I_st[i].buffer(2).buffer(cryoscope_len).average().save(f"I{i + 1}")
            Q_st[i].buffer(2).buffer(cryoscope_len).average().save(f"Q{i + 1}")
            state_st[i].buffer(2).buffer(cryoscope_len).average().save(f"state{i + 1}")


# %%
###########################
# Run or Simulate Program #
###########################
simulate =  node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=50000)  # In clock cycles = 4ns
    job = qmm.simulate(config, cryoscope, simulation_config)
    samples = job.get_simulated_samples()
    samples.con4.plot()
    plt.show()
    samples.con5.plot()
    plt.show()
    # analog5 = job.get_simulated_samples().con1.analog['5']
    # threshold = 0.01
    # indices = np.where(np.diff(np.sign(analog5 - threshold)) != 0)[0] + 1
    # # Plot the signal
    # plt.figure(figsize=(10, 6))
    # plt.plot(analog5)
    # plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    # for idx in indices:
    #     plt.axvline(idx, color='g', linestyle='--')

    # subtracted_values = []

    # for i in range(0, len(indices), 2):
    #     if i + 1 < len(indices):
    #         subtracted_value = indices[i + 1] - indices[i]
    #         subtracted_values.append(subtracted_value)

    # # Print the subtracted values
    # for i, value in enumerate(subtracted_values):
    #     print(f"Subtracted value {i + 1}: {value}")
    # plt.show(block=False)
else:
    try:
        # Open the quantum machine
        qm = qmm.open_qm(config, close_other_machines=True)
        print("Open QMs: ", qmm.list_open_quantum_machines())
        # Send the QUA program to the OPX, which compiles and executes it
        job = qm.execute(cryoscope)
        
        # print(f"Fetching results for qubit {qubits[i].name}")
        # data_list = sum([[f"I{i + 1}", f"Q{i + 1}",f"state{i + 1}"] ], ["n"])
        # results = fetching_tool(job, data_list, mode="live")
        # while results.is_processing():
        #     fetched_data = results.fetch_all()
        #     n = fetched_data[0]
        #     progress_counter(n, n_avg, start_time=results.start_time)
        while not job.status == 'completed':
            pass
    finally:
        qm.close()
        print("Experiment QM is now closed")
        # plt.show(block=True)

# %%

# %%
if not simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, [qubit], {"axis": ["x","y"], "time": cryoscope_time})
    plot_process = True

# %%
if not simulate:
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
if not simulate:
        # extract the rising part of the data for analysis
    threshold = flux_cryoscope_q.max().values*0.6 # Set the threshold value
    rise_index = np.argmax(flux_cryoscope_q.values > threshold) + 1
    drop_index =  len(flux_cryoscope_q) - 2
    flux_cryoscope_tp = flux_cryoscope_q.sel(time=slice(rise_index,drop_index ))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(
        time=flux_cryoscope_tp.time - rise_index + 1)


    f,axs = plt.subplots(2)
    flux_cryoscope_q.plot(ax = axs[0])
    axs[0].axvline(rise_index, color='r')
    axs[0].axvline(drop_index, color='r')
    flux_cryoscope_tp.plot(ax = axs[1])
    plt.show()
# %%
if not simulate:
    # Fit two exponents
    # Filtering the data might improve the fit at the first few nS, play with range to achieve this
    filtered_flux_cryoscope_q = savgol(flux_cryoscope_tp, 'time', range = 3, order = 2)
    da = flux_cryoscope_tp

    first_vals = da.sel(time=slice(0, 1)).mean().values
    final_vals = da.sel(time=slice(50, None)).mean().values

    try:
        p0 = [final_vals, -1+first_vals/final_vals, 50]
        fit, _  = curve_fit(expdecay, da.time[5:], da[5:],
                p0=p0)
    except:
        fit = p0
        print('single exp fit failed')
    try:
        p0 = [fit[0], fit[1], 2, fit[1], fit[2]]
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

    def find_A_B(fit):
        b = fit[1]
        tau = fit[2]
        c = fit[3]
        sigma = fit[4]
        mu = (tau + sigma)/(tau - sigma)
        def solve_quadratic_equation(A, B, C):
            # Calculate the discriminant
            discriminant = (B**2) - (4*A*C)

            # Find two solutions using the quadratic formula
            solution1 = (-B + np.sqrt(discriminant)) / (2*A)
            solution2 = (-B - np.sqrt(discriminant)) / (2*A)

            return solution1, solution2
        # Coefficients
        A = mu - 1
        B = -(2 + b * (mu + 1) + c* (mu-1))
        C = 2 * c
        # Solve the quadratic equation
        solutions = solve_quadratic_equation(A, B, C)
        
        B = solutions[1].real
        A = (b + c - B) / (B + 1)
        alpha = np.exp(-1 / tau)
        beta = np.exp(-1 / sigma)
        return A, B, alpha, beta

    A, B, alpha, beta = find_A_B(fit2)
    fir1 = [1 / (1 + A), -alpha / (1 + A)]
    iir1 = [(A + alpha) / (1 + A)]

    fir2 = [1 / (1 + B), -beta / (1 + B)]
    iir2 = [(B + beta) / (1 + B)]


    IIR_for_opx = [iir2[0],iir1[0]]

    long_FIR = convolve(fir2,fir1, mode='full')/2
    long_IIR = convolve([1,-iir2[0]],[1,-iir1[0]], mode='full')/2
    filtered_response_long = lfilter(long_FIR,long_IIR, flux_cryoscope_q)

    if plot_process:
        f,ax = plt.subplots()
        ax.plot(flux_cryoscope_q.time,flux_cryoscope_q,label = 'data')
        ax.plot(flux_cryoscope_q.time,filtered_response_long,label = 'filtered long time')
        ax.set_ylim([final_vals*0.95,final_vals*1.05])
        ax.legend()
        plt.show()

# %%
if not simulate:
    ####  FIR filter for the response
    flux_q = flux_cryoscope_q.copy()
    flux_q.values = filtered_response_long
    flux_q_tp = flux_q.sel(time=slice(rise_index, drop_index))
    flux_q_tp = flux_q_tp.assign_coords(
        time=flux_q_tp.time - rise_index)
    final_vals = flux_q_tp.sel(time=slice(100, None)).mean().values
    step = np.ones(len(flux_q)+100)*final_vals
    fir_est = estimate_fir_coefficients(step, flux_q_tp.values, 28)

    FIR_new = fir_est

    filtered_response_long = lfilter(long_FIR,long_IIR, flux_cryoscope_q)

    convolved_fir = convolve(long_FIR,FIR_new, mode='full')
    filtered_response_Full = lfilter(convolved_fir,long_IIR, flux_cryoscope_q)

    if plot_process:
        flux_cryoscope_q.plot(label =  'data')
        plt.plot(filtered_response_long, label = 'filtered long time')
        plt.plot(filtered_response_Full, label = 'filtered full, deconvolved')
        plt.axhline(final_vals*1.001, color = 'k')
        plt.axhline(final_vals*0.999, color = 'k')
        plt.ylim([final_vals*0.95,final_vals*1.05])
        plt.legend()
        plt.show()


# %%
if not simulate:
    def find_diff(x, y, y0, plot = False):
        filterd_y  = lfilter(x,[1,0], y)
        diffs = np.sum(np.abs(filterd_y - y0))
        if plot:
            plt.plot(filterd_y)
        return diffs

    result = minimize(find_diff, x0=FIR_new, args = (filtered_response_long,np.mean(filtered_response_long[rise_index+50:drop_index])))

    convolved_fir = convolve(long_FIR,result.x, mode='full')
    filtered_response_Full = lfilter(convolved_fir,long_IIR, flux_cryoscope_q)

    if plot_process:
        flux_cryoscope_q.plot(label =  'data')
        plt.plot(filtered_response_long, label = 'filtered long time')
        plt.plot(filtered_response_Full, label = 'filtered full, fitted')
        plt.axhline(final_vals*1.001, color = 'k')
        plt.axhline(final_vals*0.999, color = 'k')
        plt.ylim([final_vals*0.95,final_vals*1.05])
        plt.legend()
        plt.show()

# %%
if not simulate:
    # plotting the results
    fig,ax = plt.subplots()
    ax.plot(flux_cryoscope_q.time,flux_cryoscope_q,label = 'data')
    ax.plot(flux_cryoscope_q.time,filtered_response_long,'--', label = 'slow rise correction')
    ax.plot(flux_q.time,filtered_response_Full,'--', label = 'expected corrected response')
    ax.axhline(final_vals*1.001, color = 'k')
    ax.axhline(final_vals*0.999, color = 'k')
    ax.set_ylim([final_vals*0.95,final_vals*1.05])
    ax.legend()
    node.results['figure'] = fig

# %%
if not simulate:
    node.results['fit_results'] = {}
    for q in qubits:
        node.results['fit_results'][q.name] = {}
        node.results['fit_results'][q.name]['fir'] = convolved_fir.tolist()
        node.results['fit_results'][q.name]['iir'] = IIR_for_opx


# %%

if not simulate:
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.filter_fir_taps = convolved_fir.tolist()
            qubit.z.filter_iir_taps = IIR_for_opx

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%