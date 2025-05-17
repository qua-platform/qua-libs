# %%
"""
        CRYOSCOPE
"""

from typing import List, Literal, Optional

import matplotlib
from matplotlib.pylab import normal
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm import QuantumMachinesManager, SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.multi_user import qm_session
from qualang_tools.plot import interrupt_on_close
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from quam_libs.components import QuAM
from quam_libs.lib.cryoscope_tools import cryoscope_frequency, estimate_fir_coefficients, expdecay, savgol, two_expdecay
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, save_node
from quam_libs.macros import active_reset, multiplexed_readout, node_save, qua_declaration
from scipy import signal
from scipy.optimize import curve_fit, minimize
from scipy.signal import convolve, deconvolve, lfilter


# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = ["qubitC1", "qubitC2"]
    num_averages: int = 300
    amplitude_factor: float = 1.0
    cryoscope_len: int = 240
    reset_type_active_or_thermal: Literal["active", "thermal"] = "active"
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    timeout: int = 100
    reset_filters: bool = True
    load_data_id: Optional[int] = None


node = QualibrationNode(name="12_Cryoscope", parameters=Parameters(load_data_id=int(679)))


# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load("/Users/paul/QM/qualibrate_data/031224/#679_12_Cryoscope_160836/quam_state/state.json")
# machine = QuAM.load()
# Get the relevant QuAM components
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]

# if node.parameters.reset_filters:
#     for qubit in qubits:
#         qubit.z.opx_output.feedforward_filter = [1.0, 0.0]
#         qubit.z.opx_output.feedback_filter = [0.0, 0.0]

# %%
# Generate the OPX and Octave configurations
# config = machine.generate_config()
# octave_config = machine.get_octave_config()
# Open Communication with the QOP
# qmm = machine.connect()
flux_point = node.parameters.flux_point_joint_or_independent
reset_type = node.parameters.reset_type_active_or_thermal
amplitude_factor = node.parameters.amplitude_factor

num_qubits = len(qubits)

# %%

####################
# Helper functions #
####################


# def baked_waveform(waveform_amp, qubit):
#     pulse_segments = []  # Stores the baking objects
#     # Create the different baked sequences, each one corresponding to a different truncated duration
#     waveform = [waveform_amp] * 16

#     for i in range(1, 17):  # from first item up to pulse_duration (16)
#         with baking(config, padding_method="left") as b:
#             wf = waveform[:i]
#             b.add_op("flux_pulse", qubit.z.name, wf)
#             b.play("flux_pulse", qubit.z.name)

#         # Append the baking object in the list to call it from the QUA program
#         pulse_segments.append(b)

#     return pulse_segments


# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

cryoscope_len = node.parameters.cryoscope_len  # The length of the cryoscope in nanoseconds

assert cryoscope_len % 16 == 0, "cryoscope_len is not multiple of 16 nanoseconds"

baked_signals = {}
# Baked flux pulse segments with 1ns resolution

# baked_signals = baked_waveform(qubits[0].z.operations["const"].amplitude * amplitude_factor, qubits[0])

cryoscope_time = np.arange(1, cryoscope_len + 1, 1)  # x-axis for plotting - must be in ns

# %%

# with program() as cryoscope:

#     I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
#     t = declare(int)  # QUA variable for the flux pulse segment index
#     state = [declare(int) for _ in range(num_qubits)]
#     state_st = [declare_stream() for _ in range(num_qubits)]
#     global_state = declare(int)
#     idx = declare(int)
#     idx2 = declare(int)
#     flag = declare(bool)
#     qubit = qubits[0]
#     i = 0

#     # Bring the active qubits to the minimum frequency point
#     if flux_point == "independent":
#         machine.apply_all_flux_to_min()
#         qubit.z.to_independent_idle()
#     elif flux_point == "joint":
#         machine.apply_all_flux_to_joint_idle()
#         # qubit.z.set_dc_offset(0.01 + qubit.z.joint_offset)
#     else:
#         machine.apply_all_flux_to_zero()
#     wait(1000)

#     # Outer loop for averaging
#     with for_(n, 0, n < n_avg, n + 1):
#         save(n, n_st)

#         # The first 16 nanoseconds
#         with for_(idx, 0, idx < 16, idx + 1):
#             # Alternate between X/2 and Y/2 pulses
#             # for tomo in ['x90', 'y90']:
#             with for_each_(flag, [True, False]):
#                 if reset_type == "active":
#                     for qubit in qubits:
#                         active_reset(qubit)
#                 else:
#                     wait(qubit.thermalization_time * u.ns)
#                 align()
#                 # Play first X/2
#                 for qubit in qubits:
#                     qubit.xy.play("x180", amplitude_scale=0.5)
#                 align()
#                 # Delay between x90 and the flux pulse
#                 # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
#                 wait(16 // 4)
#                 align()
#                 # with switch_(idx):
#                 #     for j in range(16):
#                 #         with case_(j):
#                 #             baked_signals[j].run()
#                 # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
#                 # pulse arrives after the longest flux pulse
#                 for qubit in qubits:
#                     qubit.xy.wait((cryoscope_len + 16) // 4)
#                     # Play second X/2 or Y/2
#                     # if tomo == 'x90':
#                     with if_(flag):
#                         qubit.xy.play("x90")
#                     # elif tomo == 'y90':
#                     with else_():
#                         qubit.xy.play("y90")
#                 # Measure resonator state after the sequence
#                 align()
#                 qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
#                 assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
#                 save(state[i], state_st[i])

#         # The first 16-32 nanoseconds
#         with for_(idx, 0, idx < 16, idx + 1):
#             # Alternate between X/2 and Y/2 pulses
#             # for tomo in ['x90', 'y90']:
#             with for_each_(flag, [True, False]):
#                 if reset_type == "active":
#                     for qubit in qubits:
#                         active_reset(qubit)
#                 else:
#                     wait(qubit.thermalization_time * u.ns)
#                 align()
#                 # Play first X/2
#                 for qubit in qubits:
#                     qubit.xy.play("x180", amplitude_scale=0.5)
#                 align()
#                 # Delay between x90 and the flux pulse
#                 # NOTE: it can be made larger than 16 nanoseconds it could be benefitial
#                 wait(16 // 4)
#                 align()
#                 with switch_(idx):
#                     for j in range(16):
#                         with case_(j):
#                             baked_signals[j].run()
#                 # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
#                 # pulse arrives after the longest flux pulse
#                 for qubit in qubits:
#                     qubit.xy.wait((cryoscope_len + 16) // 4)
#                     # Play second X/2 or Y/2
#                     # if tomo == 'x90':
#                     with if_(flag):
#                         qubit.xy.play("x90")
#                     # elif tomo == 'y90':
#                     with else_():
#                         qubit.xy.play("y90")
#                 # Measure resonator state after the sequence
#                 align()
#                 qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
#                 assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
#                 save(state[i], state_st[i])

#         with for_(t, 8, t < cryoscope_len // 4, t + 4):

#             with for_(idx, 0, idx < 16, idx + 1):

#                 # Alternate between X/2 and Y/2 pulses
#                 # for tomo in ['x90', 'y90']:
#                 with for_each_(flag, [True, False]):
#                     # Initialize the qubits
#                     if reset_type == "active":
#                         for qubit in qubits:
#                             active_reset(qubit)
#                     else:
#                         wait(qubit.thermalization_time * u.ns)
#                     align()
#                     # Play first X/2
#                     for qubit in qubits:
#                         qubit.xy.play("x90")
#                     align()
#                     # Delay between x90 and the flux pulse
#                     wait(16 // 4)
#                     align()
#                     with switch_(idx):
#                         for j in range(16):
#                             with case_(j):
#                                 baked_signals[j].run()
#                                 qubits[0].z.play("const", duration=t - 4, amplitude_scale=amplitude_factor)

#                     # Wait for the idle time set slightly above the maximum flux pulse duration to ensure that the 2nd x90
#                     # pulse arrives after the longest flux pulse
#                     for qubit in qubits:
#                         qubit.xy.wait((cryoscope_len + 16) // 4)
#                         # Play second X/2 or Y/2
#                         with if_(flag):
#                             qubit.xy.play("x90")
#                         # elif tomo == 'y90':
#                         with else_():
#                             qubit.xy.play("y90")

#                     # Measure resonator state after the sequence
#                     align()
#                     qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
#                     assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
#                     save(state[i], state_st[i])

#     with stream_processing():
#         # for the progress counter
#         n_st.save("iteration")
#         for i, qubit in enumerate(qubits):
#             state_st[i].buffer(2).buffer(cryoscope_len).average().save(f"state{i + 1}")


# %%

# simulate = node.parameters.simulate

# if node.parameters.simulate:
#     # Simulates the QUA program for the specified duration
#     simulation_config = SimulationConfig(duration=50000)  # In clock cycles = 4ns
#     job = qmm.simulate(config, cryoscope, simulation_config)
#     samples = job.get_simulated_samples()
#     samples.con4.plot()
#     plt.show()
#     samples.con5.plot()
#     plt.show()

# else:
#     with qm_session(qmm, config, timeout=node.parameters.timeout ) as qm:
#         job = qm.execute(cryoscope)
#         data_list = ["iteration"]
#         results = fetching_tool(job, data_list, mode="live")

#         while results.is_processing():
#             fetched_data = results.fetch_all()
#             n = fetched_data[0]
#             progress_counter(n, n_avg, start_time=results.start_time)


# %%

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:

    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
        plot_process = True
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, [qubit], {"axis": ["x", "y"], "time": cryoscope_time})
        plot_process = True

# %%
if not node.parameters.simulate:
    if plot_process:
        ds.state.sel(qubit=qubits[1].name).plot(hue="axis")
        plt.show()

    sg_order = 2
    sg_range = 3
    qubit = qubits[1]
    da = ds.state.sel(qubit=qubit.name)

    flux_cryoscope_q = cryoscope_frequency(
        da,
        quad_term=qubit.freq_vs_flux_01_quad_term,
        stable_time_indices=(20, cryoscope_len - 20),
        sg_order=sg_order,
        sg_range=sg_range,
        plot=plot_process,
    )
    plt.show()

# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    # extract the rising part of the data for analysis
    threshold = flux_cryoscope_q.max().values * 0.6  # Set the threshold value
    rise_index = np.argmax(flux_cryoscope_q.values > threshold) + 1
    drop_index = len(flux_cryoscope_q) - 4
    flux_cryoscope_tp = flux_cryoscope_q.sel(time=slice(rise_index, drop_index))
    flux_cryoscope_tp = flux_cryoscope_tp.assign_coords(time=flux_cryoscope_tp.time - rise_index + 1)

    f, axs = plt.subplots(2)
    flux_cryoscope_q.plot(ax=axs[0])
    axs[0].axvline(rise_index, color="r", lw=0.5, ls="--")
    axs[0].axvline(drop_index, color="r", lw=0.5, ls="--")
    flux_cryoscope_tp.plot(ax=axs[1])
    node.results["figure"] = f
    plt.tight_layout()
    plt.show()

# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    const_flux_len = len(flux_cryoscope_tp)  # ns

    times = np.arange(-const_flux_len, const_flux_len, 1)
    normalize = np.mean(flux_cryoscope_q.sel(time=slice(600, 800)))
    step_response_volt = np.append(np.zeros(const_flux_len), flux_cryoscope_tp / normalize)
    step_response_volt_offset = np.mean(step_response_volt)
    step_response_volt -= step_response_volt_offset
    if plot_process:
        plt.plot(times, step_response_volt, label=r"Measured Response")
        plt.legend()


# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    n = 800
    dt_fine = 1
    times_fine = np.linspace(-400, 400, n)
    # print(times_fine)
    step_response_fine = np.interp(times_fine, times, step_response_volt)
    step_response_volt = step_response_fine  # should go through and eliminate this

    impulse_response_fine = np.gradient(step_response_fine, times_fine)
    impulse_response_fine *= np.heaviside(times_fine + dt_fine, 0.5)

    if plot_process:
        plt.plot(times_fine, impulse_response_fine, label=r"Measured Impluse Response")


# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    heaviside = np.heaviside(times_fine, 0.5) - 0.5

    convolution = signal.fftconvolve(impulse_response_fine, heaviside, "same")

    scale = (np.mean(step_response_fine[-len(step_response_fine) // 32 :])) / (
        np.mean(convolution[-len(convolution) // 32 :])
    )

    step_from_impulse_offset = step_response_volt_offset
    step_from_impulse = scale * convolution

    if plot_process:
        plt.plot(times_fine, step_response_volt + step_response_volt_offset, label=r"Measured Response")
        plt.plot(times_fine, step_from_impulse + step_from_impulse_offset, label="Reconstructed Response")
        plt.xlim(-200, 200)


# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    impulse_response_fine_tilde = np.fft.fftshift(np.fft.fft(impulse_response_fine))
    heaviside_tilde = np.fft.fftshift(np.fft.fft(heaviside))
    convolution = np.fft.ifft(np.fft.ifftshift(impulse_response_fine_tilde * np.conj(heaviside_tilde)))[
        0 : len(times_fine)
    ]

    if plot_process:
        plt.plot(times_fine, step_from_impulse + step_from_impulse_offset, label="Reconstructed Response (Convolve)")
        plt.plot(times_fine, scale * convolution + step_from_impulse_offset, "o", label="Reconstructed Response (FFT)")
        plt.xlim(-200, 200)
        plt.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
# %%
if not node.parameters.simulate and node.parameters.reset_filters:
    heaviside = np.heaviside(times_fine,0.5)-0.5
    impulse_response_fine_tilde = (np.fft.fft(impulse_response_fine))
    heaviside_tilde = (np.fft.fft(heaviside))
    predistorted_tilde = heaviside_tilde/impulse_response_fine_tilde/scale
    predistorted = -np.fft.ifft((predistorted_tilde))
    predistort_out = np.fft.ifft(predistorted_tilde*impulse_response_fine_tilde*scale)

    if plot_process:
        plt.figure()
        plt.plot(times_fine,predistorted, '.-',label="Predistorted Pulse")
        plt.plot(times_fine,predistort_out,'o',label="Predistorted Response")
        plt.xlim(-200, 200)
        plt.legend()
        plt.show()

# %%


arb_wf = np.concatenate(((predistorted[len(predistorted) // 2 :] + 0.5), np.ones(len(flux_cryoscope_tp) - len(predistorted))))* normalize.data

smooth = savgol(arb_wf, 2, 20)

if plot_process:
    plt.figure()
    # plt.plot(arb_wf, '.--')
    plt.plot(arb_wf[4:], '.--')
    plt.plot(smooth[4:], '.--')
    plt.plot(flux_cryoscope_tp, '.--')
    plt.show()
# %%

if not node.parameters.simulate and node.parameters.reset_filters:
    with node.record_state_updates():
        for qubit in qubits:
            qubit.z.opx_output.feedforward_filter = convolved_fir.tolist()
            qubit.z.opx_output.feedback_filter = IIR_for_opx

# %%
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
save_node(node)
# %%
