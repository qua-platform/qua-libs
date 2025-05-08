# %%
"""
RAMSEY WITH VIRTUAL Z ROTATIONS
The program consists in playing a Ramsey sequence (x90 - idle_time - x90 - measurement) for different idle times.
Instead of detuning the qubit gates, the frame of the second x90 pulse is rotated (de-phased) to mimic an accumulated
phase acquired for a given detuning after the idle time.
This method has the advantage of playing resonant gates.

From the results, one can fit the Ramsey oscillations and precisely measure the qubit resonance frequency and T2*.

Prerequisites:
    - Having found the resonance frequency of the resonator coupled to the qubit under study (resonator_spectroscopy).
    - Having calibrated qubit pi pulse (x180) by running qubit, spectroscopy, rabi_chevron, power_rabi and updated the state.
    - (optional) Having calibrated the readout (readout_frequency, amplitude, duration_optimization IQ_blobs) for better SNR.

Next steps before going to the next node:
    - Update the qubits frequency (f_01) in the state.
    - Save the current state by calling machine.save("quam")
"""
from turtledemo.penrose import start

from lmfit.lineshapes import thermal_distribution
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal
from qualang_tools.multi_user import qm_session

import time


class Parameters(NodeParameters):
    qubits: Optional[str] = ["qubitC1"]
    num_repetition: int = 100_000
    ef_charge_dispersion_in_kHz : float = 200.0
    flux_point_joint_or_independent_or_arbitraryPulse_or_arbitraryDC: Literal['joint', 'independent', 'arbitraryPulse', 'arbitraryDC'] = "joint"
    thermalization_time_ns: int = 1_000_000
    simulate: bool = False
    sanity_check_with_single_ef_x90: bool = False
    use_state_discrimination: bool = True
    timeout: int = 100

node = QualibrationNode(
    name="05d_charge_parity_EF", parameters=Parameters()
)


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handle units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()
# Generate the OPX and Octave configurations
config = machine.generate_config()
octave_config = machine.get_octave_config()
# Open Communication with the QOP
qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == '':
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)
# %%
###################
# The QUA program #
###################

flux_point = node.parameters.flux_point_joint_or_independent_or_arbitraryPulse_or_arbitraryDC  # 'independent' or 'joint'
if flux_point == "arbitraryDC" or "arbitraryPulse":
    # arb_detunings = {q.name : q.xy.intermediate_frequency - q.arbitrary_intermediate_frequency for q in qubits}
    arb_detunings = {q.name: 0.0 for q in qubits}
    arb_flux_bias_offset = {q.name: q.z.arbitrary_offset for q in qubits}
else:
    arb_flux_bias_offset = {q.name: 0.0 for q in qubits}
    arb_detunings = {q.name: 0.0 for q in qubits}

n_rep = node.parameters.num_repetition
ms = np.arange(n_rep)

period_charge_parity_pi_ns = int((1.0e6/node.parameters.ef_charge_dispersion_in_kHz)/2)
if node.parameters.sanity_check_with_single_ef_x90:
    repetition_period_ns = node.parameters.thermalization_time_ns + 2*qubits[0].xy.operations.x180.length + qubits[0].xy.operations.EF_x90.length + qubits[0].resonator.operations.readout.length
else:
    repetition_period_ns = node.parameters.thermalization_time_ns + 2 * qubits[0].xy.operations.x180.length + 2 * \
                           qubits[0].xy.operations.EF_x90.length + period_charge_parity_pi_ns + qubits[
                               0].resonator.operations.readout.length
print(f'Repetition period: {repetition_period_ns*1e-3:.1f} us')

with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    m = declare(int)
    m_st = declare_stream()
    t = declare(int)  # QUA variable for the idle time
    sign = declare(int)  # QUA variable to change the sign of the detuning
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]
        
    for i, q in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_zero()
            q.z.to_independent_idle()
        elif flux_point == "joint" or "arbitraryPulse":
            machine.apply_all_flux_to_joint_idle()
        elif flux_point == "arbitraryDC":
            machine.apply_all_flux_to_joint_idle()
            q.z.set_dc_offset(q.z.arbitrary_offset+q.z.joint_offset)
            q.xy.update_frequency(q.arbitrary_intermediate_frequency)
        else:
            machine.apply_all_flux_to_zero()

        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()
        with for_(m, 0, m < n_rep, m + 1):
            save(m, m_st)
            align()
            q.xy.update_frequency(q.xy.intermediate_frequency)
            q.xy.play("x180",amplitude_scale=0)
            q.xy.play("x180")
            q.xy.update_frequency(q.xy.intermediate_frequency - q.anharmonicity)
            q.xy.play("x180",amplitude_scale=0)
            q.xy.play("EF_x90")
            
            
            if not node.parameters.sanity_check_with_single_ef_x90:
                q.xy.wait(period_charge_parity_pi_ns//4)
                q.xy.play("EF_y90")
            
            q.xy.update_frequency(q.xy.intermediate_frequency)
            q.xy.play("x180",amplitude_scale=0)
            q.xy.play('x180')
            q.xy.update_frequency(q.xy.intermediate_frequency - q.anharmonicity)
            q.xy.play("x180",amplitude_scale=0)
            q.xy.play("EF_x180")

            # Align the elements to measure after playing the qubit pulse.
            align()
            # Measure the state of the resonators

            # save data
            if node.parameters.use_state_discrimination:
                readout_state(q, state[i])
                save(state[i], state_st[i])
            else:
                q.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                save(I[i], I_st[i])
                save(Q[i], Q_st[i])

            # Wait for the qubits to decay to the ground state
            q.resonator.wait(node.parameters.thermalization_time_ns//4)

            # Reset the frame of the qubits in order not to accumulate rotations
            reset_frame(q.xy.name)

        align()

    with stream_processing():
        m_st.save("n")
        for i in range(num_qubits):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(n_rep).save(f"state{i + 1}")
            else:
                I_st[i].buffer(n_rep).save(f"I{i + 1}")
                Q_st[i].buffer(n_rep).save(f"Q{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(ramsey)

        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            n = results.fetch_all()[0]
            progress_counter(n, n_rep, start_time=results.start_time)


# %%
handles = job.result_handles
ds = fetch_results_as_xarray(handles, qubits, {"repetition": ms})

# %%
# %%

# elapsed_time = np.round(end_time - start_time,decimals=1)
time_us = np.arange(0,n_rep)*float(repetition_period_ns)*1e-3
xpl = time_us*1e-3
state = ds.state
fig1 = plt.figure()
plt.plot(xpl,state[0],'o-',ms=2)
plt.ylabel('State')
plt.xlabel('Time (ms)')
plt.show()

fig2 = plt.figure()
plt.plot(xpl[:100],state[0][:100],'o-',ms=2)
plt.ylabel('State')
plt.xlabel('Time (ms)')
plt.title('First 100 pts')
plt.show()


node.results['figure1'] = fig1
node.results['figure2'] = fig2

# node.results['elapsed_time'] = elapsed_time
node.results['state'] = state[0]
node.results['time_us'] = time_us
# node.results['idle_time_ns'] = idle_time_ns
# node.results['real_time_s'] = real_time_s



# %%
#
# if not simulate:
#     if node.parameters.use_state_discrimination:
#         fit = fit_oscillation_decay_exp(ds.state, 'time')
#     else:
#         fit = fit_oscillation_decay_exp(ds.I, 'time')
#     fit.attrs = {'long_name' : 'time', 'units' : 'usec'}
#     fitted =  oscillation_decay_exp(ds.time,
#                                                     fit.sel(
#                                                         fit_vals="a"),
#                                                     fit.sel(
#                                                         fit_vals="f"),
#                                                     fit.sel(
#                                                         fit_vals="phi"),
#                                                     fit.sel(
#                                                         fit_vals="offset"),
#                                                     fit.sel(fit_vals="decay"))
#
#     frequency = fit.sel(fit_vals = 'f')
#     frequency.attrs = {'long_name' : 'frequency', 'units' : 'MHz'}
#
#     decay = fit.sel(fit_vals = 'decay')
#     decay.attrs = {'long_name' : 'decay', 'units' : 'nSec'}
#
#     frequency = frequency.where(frequency>0,drop = True)
#
#     decay = fit.sel(fit_vals = 'decay')
#     decay.attrs = {'long_name' : 'decay', 'units' : 'nSec'}
#
#     decay_res = fit.sel(fit_vals = 'decay_decay')
#     decay_res.attrs = {'long_name' : 'decay', 'units' : 'nSec'}
#
#     tau = 1/fit.sel(fit_vals='decay')
#     tau.attrs = {'long_name' : 'T2*', 'units' : 'uSec'}
#
#     tau_error = tau * (np.sqrt(decay_res)/decay)
#     tau_error.attrs = {'long_name' : 'T2* error', 'units' : 'uSec'}
#
#
# within_detuning = (1e9*frequency < 2 * detuning).mean(dim = 'sign') == 1
# positive_shift = frequency.sel(sign = 1) > frequency.sel(sign = -1)
# freq_offset = within_detuning * (frequency* fit.sign).mean(dim = 'sign') + ~within_detuning * positive_shift * (frequency).mean(dim = 'sign') -~within_detuning * ~positive_shift * (frequency).mean(dim = 'sign')
#
# # freq_offset = (frequency* fit.sign).mean(dim = 'sign')
# decay = 1e-9*tau.mean(dim = 'sign')
# decay_error = 1e-9*tau_error.mean(dim = 'sign')
# fit_results = {q.name : {'freq_offset' : 1e9*freq_offset.loc[q.name].values, 'decay' : decay.loc[q.name].values, 'decay_error' : decay_error.loc[q.name].values} for q in qubits}
# node.results['fit_results'] = fit_results
# for q in qubits:
#     print(f"Frequency offset for qubit {q.name} : {(fit_results[q.name]['freq_offset']/1e6):.2f} MHz ")
#     print(f"T2* for qubit {q.name} : {1e6*fit_results[q.name]['decay']:.2f} us")
#
# # %%
# grid_names = [f'{q.name}_0' for q in qubits]
# grid = QubitGrid(ds, grid_names)
# for ax, qubit in grid_iter(grid):
#     if node.parameters.use_state_discrimination:
#         (ds.sel(sign = 1).loc[qubit].state).plot(ax = ax, x = 'time',
#                                              c = 'C0', marker = '.', ms = 5.0, ls = '', label = "$\Delta$ = +")
#         (ds.sel(sign = -1).loc[qubit].state).plot(ax = ax, x = 'time',
#                                              c = 'C1', marker = '.', ms = 5.0, ls = '', label = "$\Delta$ = -")
#         ax.plot(ds.time, fitted.loc[qubit].sel(sign = 1), c = 'black', ls = '-', lw=1)
#         ax.plot(ds.time, fitted.loc[qubit].sel(sign = -1), c = 'red', ls = '-', lw=1)
#         ax.set_ylabel('State')
#     else:
#         (ds.sel(sign = 1).loc[qubit].I*1e3).plot(ax = ax, x = 'time',
#                                              c = 'C0', marker = '.', ms = 5.0, ls = '', label = "$\Delta$ = +")
#         (ds.sel(sign = -1).loc[qubit].I*1e3).plot(ax = ax, x = 'time',
#                                              c = 'C1', marker = '.', ms = 5.0, ls = '', label = "$\Delta$ = -")
#         ax.set_ylabel('Trans. amp. I [mV]')
#         ax.plot(ds.time, 1e3*fitted.loc[qubit].sel(sign = 1), c = 'black', ls = '-', lw=1)
#         ax.plot(ds.time, 1e3*fitted.loc[qubit].sel(sign = -1), c = 'red', ls = '-', lw=1)
#
#     ax.set_xlabel('Idle time [nS]')
#     ax.set_title(qubit['qubit'])
#     ax.text(0.5, 0.9, f'T2* = {1e6*fit_results[q.name]["decay"]:.1f} + {1e6*fit_results[q.name]["decay_error"]:.1f} usec', transform=ax.transAxes, fontsize=10,
#     verticalalignment='top',ha='center', bbox=dict(facecolor='white', alpha=0.5))
#     # ax.legend()
# grid.fig.suptitle('Ramsey : I vs. idle time')
# plt.tight_layout()
# plt.gcf().set_figwidth(15)
# plt.show()
# node.results['figure'] = grid.fig
#
# # %%
# with node.record_state_updates():
#     for q in qubits:
#         if flux_point == "arbitraryPulse" or flux_point == "arbitraryDC":
#             q.arbitrary_intermediate_frequency = np.round(q.arbitrary_intermediate_frequency + (fit_results[q.name]['freq_offset'])) # changed - to +
#         else:
#             q.anharmonicity = q.anharmonicity - int(fit_results[q.name]['freq_offset'])

# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()




# %%
# %%
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

data_q = ds.state.sel(qubit = qubits[0].name)
time_stamp_q = xpl
f, Pxx_den = signal.welch(data_q-data_q.mean(),  1e9/np.mean(np.diff(time_stamp_q)), 
                          nperseg=2**10)
dat_fft = xr.Dataset({'Pxx_den': (['freq'], Pxx_den)}, coords={'freq': f}).Pxx_den
# %%
dat_fft.plot()

# %%
