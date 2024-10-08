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
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List

# %% {Node_parameters}
class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 100
    frequency_detuning_in_mhz: float = 4.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 2000
    wait_time_step_in_ns: int = 20
    flux_span : float = 0.02
    flux_step : float = 0.002
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100
    flux_mode_dc_or_pulsed: Literal['dc', 'pulsed'] = 'pulsed'

node = QualibrationNode(
    name="08a_Ramsey_flux_cal",
    parameters=Parameters()
)




from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, multiplexed_readout, node_save, active_reset, readout_state

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp

# %%



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

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages

# Dephasing time sweep (in clock cycles = 4ns) - minimum is 4 clock cycles
idle_times = np.arange(
    node.parameters.min_wait_time_in_ns // 4,
    node.parameters.max_wait_time_in_ns // 4,
    node.parameters.wait_time_step_in_ns // 4,
)

# Detuning converted into virtual Z-rotations to observe Ramsey oscillation and get the qubit frequency
detuning = int(1e6 * node.parameters.frequency_detuning_in_mhz)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
fluxes = np.arange(-node.parameters.flux_span / 2, node.parameters.flux_span / 2+0.001, step = node.parameters.flux_step)

# %%
with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    init_state = declare(int)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    t = declare(int)  # QUA variable for the idle time
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    flux = declare(fixed)  # QUA variable for the flux dc level

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        # Wait for the flux bias to settle
        for qb in qubits:
            wait(1000, qb.z.name)
        
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(flux, fluxes)):
                if node.parameters.flux_mode_dc_or_pulsed == 'dc':
                    # Flux sweeping for a qubit
                    if flux_point == "independent":
                        qubit.z.set_dc_offset(flux + qubit.z.independent_offset)
                    elif flux_point == "joint":
                        qubit.z.set_dc_offset(flux + qubit.z.joint_offset)
                    else:
                        raise RuntimeError(f"unknown flux_point")  
                                
                for qb in qubits:
                    wait(100, qb.z.name)
                
                align() 

                with for_(*from_array(t, idle_times)):
                    readout_state(qubit, init_state)
                    qubit.align()
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t ))
                    align()
                    # Strict_timing ensures that the sequence will be played without gaps
                    if node.parameters.flux_mode_dc_or_pulsed == 'dc':  
                        with strict_timing_():
                            qubit.xy.play("x180", amplitude_scale = 0.5)
                            qubit.xy.frame_rotation_2pi(phi)
                            qubit.xy.wait(t)
                            qubit.xy.play("x180", amplitude_scale = 0.5)
                    else:
                        # with strict_timing_():
                        qubit.xy.play("x180", amplitude_scale = 0.5)
                        qubit.align()
                        wait(20, qubit.z.name)
                        qubit.z.play("const", amplitude_scale = flux / qubit.z.operations["const"].amplitude, duration=t)
                        wait(20, qubit.z.name)
                        qubit.xy.frame_rotation_2pi(phi)
                        qubit.align()
                        qubit.xy.play("x180", amplitude_scale = 0.5)

                    # Align the elements to measure after playing the qubit pulse.
                    align()
                    # Measure the state of the resonators
                    readout_state(qubit, state[i])
                    assign(state[i], init_state ^ state[i])
                    save(state[i], state_st[i])
                    
                    # Reset the frame of the qubits in order not to accumulate rotations
                    reset_frame(qubit.xy.name)
                    qubit.align()
        
        align()

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(idle_times)).buffer(len(fluxes)).average().save(f"state{i + 1}")



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
        # Get results from QUA program
        for i in range(num_qubits):
            print(f"Fetching results for qubit {qubits[i].name}")
            data_list = ["n"] + sum([[f"state{i + 1}"] for i in range(num_qubits)], [])
            results = fetching_tool(job, data_list, mode="live")
        # Live plotting
        # fig, axes = plt.subplots(2, num_qubits, figsize=(4 * num_qubits, 8))
        # interrupt_on_close(fig, job)  # Interrupts the job when closing the figure
            while results.is_processing():
            # Fetch results
                fetched_data = results.fetch_all()
                n = fetched_data[0]

                progress_counter(n, n_avg, start_time=results.start_time)

# %%

ds = fetch_results_as_xarray(job.result_handles, qubits, {"idle_time": idle_times, "flux": fluxes})

node.results = {"ds": ds}
# %%
ds = ds.assign_coords(idle_time=4*ds.idle_time/1e3)  # convert to usec
ds.flux.attrs = {'long_name': 'flux', 'units': 'V'}
ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}


# %%
fit_data = fit_oscillation_decay_exp(ds.state, 'idle_time')
fit_data.attrs = {'long_name' : 'time', 'units' : 'usec'}
fitted =  oscillation_decay_exp(ds.state.idle_time,
                                                fit_data.sel(
                                                    fit_vals="a"),
                                                fit_data.sel(
                                                    fit_vals="f"),
                                                fit_data.sel(
                                                    fit_vals="phi"),
                                                fit_data.sel(
                                                    fit_vals="offset"),
                                                fit_data.sel(fit_vals="decay"))

frequency = fit_data.sel(fit_vals = 'f')
frequency.attrs = {'long_name' : 'frequency', 'units' : 'MHz'}

decay = fit_data.sel(fit_vals = 'decay')
decay.attrs = {'long_name' : 'decay', 'units' : 'nSec'}

tau = 1/fit_data.sel(fit_vals='decay')
tau.attrs = {'long_name' : 'T2*', 'units' : 'uSec'}

frequency = frequency.where(frequency>0,drop = True)


fitvals = frequency.polyfit(dim = 'flux', deg = 2)
flux= frequency.flux
a = {}
flux_offset = {}
freq_offset = {}
for q in qubits:
    a[q.name] = float(-1e6*fitvals.sel(qubit = q.name,degree = 2).polyfit_coefficients.values)
    flux_offset[q.name]  = float((-0.5*fitvals.sel(qubit =q.name,degree = 1).polyfit_coefficients/fitvals.sel(qubit = q.name,degree = 2).polyfit_coefficients).values)
    freq_offset[q.name]  = 1e6*float(fitvals.sel(qubit = q.name,degree = 0).polyfit_coefficients.values) - detuning


# %%
grid_names = [f'{q.name}_0' for q in qubits]
grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    ds.sel(qubit = qubit['qubit']).state.plot(ax = ax)
    ax.set_title(qubit['qubit'])
    ax.set_xlabel('Idle_time (uS)')
    ax.set_ylabel(' Flux (V)')
grid.fig.suptitle('Ramsey freq. Vs. flux')
plt.tight_layout()
plt.show()
node.results['figure_raw'] = grid.fig


grid = QubitGrid(ds, grid_names)
for ax, qubit in grid_iter(grid):
    fitted_freq = fitvals.sel(qubit = qubit['qubit'],degree = 2).polyfit_coefficients * flux**2 + fitvals.sel(qubit = qubit['qubit'],degree = 1).polyfit_coefficients * flux + fitvals.sel(qubit = qubit['qubit'],degree = 0).polyfit_coefficients
    frequency.sel(qubit = qubit['qubit']).plot( marker = '.',linewidth = 0,ax=ax)
    ax.plot(flux,fitted_freq)
    ax.set_title(qubit['qubit'])
    ax.set_xlabel(' Flux (V)')
    print(f"The quad term for {qubit['qubit']} is {a[qubit['qubit']]/1e9:.3f} GHz/V^2")
    print(f"Flux offset for {qubit['qubit']} is {flux_offset[qubit['qubit']]*1e3:.1f} mV")
    print(f"Freq offset for {qubit['qubit']} is {freq_offset[qubit['qubit']]/1e6:.3f} MHz")
    print()
grid.fig.suptitle('Ramsey freq. Vs. flux')
plt.tight_layout()
plt.show()
node.results['figure'] = grid.fig

# %%
node.results['fit_results'] = {}
for q in qubits:
    node.results['fit_results'][q.name] = {}
    node.results['fit_results'][q.name]['flux_offset'] = flux_offset[q.name]
    node.results['fit_results'][q.name]['freq_offset'] = freq_offset[q.name]
    node.results['fit_results'][q.name]['quad_term'] = a[q.name]
# %% {Update_state}
with node.record_state_updates():
    for qubit in qubits:
        qubit.xy.intermediate_frequency -= freq_offset[qubit.name]
        if flux_point == 'independent':
            qubit.z.independent_offset += flux_offset[qubit.name]
        elif flux_point == 'joint':
            qubit.z.joint_offset += flux_offset[qubit.name]
        else:
            raise RuntimeError(f"unknown flux_point")
        qubit.freq_vs_flux_01_quad_term = float(a[qubit.name])
# %% {Save_results}
node.outcomes = {q.name: "successful" for q in qubits}
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
