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
from typing import Optional, Literal


class Parameters(NodeParameters):
    qubits: Optional[str] = None
    num_averages: int = 200
    frequency_detuning_in_mhz: float = 1.0
    min_wait_time_in_ns: int = 16
    max_wait_time_in_ns: int = 10000
    wait_time_step_in_ns: int = 100
    flux_span : float = 0.05
    flux_step : float = 0.002
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="08a_Ramsey_flux_cal",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array, get_equivalent_log_array
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
if node.parameters.qubits is None:
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits.split(', ')]
num_qubits = len(qubits)
# %%
###################
# The QUA program #
###################
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
dcs = np.arange(-node.parameters.flux_span / 2, node.parameters.flux_span / 2+0.001, step = node.parameters.flux_step)
quads = {qubit.name: int(qubit.freq_vs_flux_01_quad_term * 1e-6) for qubit in qubits}
# %%
with program() as ramsey:
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    init_state = declare(int)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    t = declare(int)  # QUA variable for the idle time
    phi = declare(fixed)  # QUA variable for dephasing the second pi/2 pulse (virtual Z-rotation)
    phi2 = declare(fixed)  # QUA variable for correcting the flux effect on the phase (virtual Z-rotation)
    freq = declare(int)
    dc = declare(fixed)  # QUA variable for the flux dc level
    
    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.to_independent_idle()
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
        else:
            machine.apply_all_flux_to_zero()

        for qubit in qubits:
            wait(1000, qubit.z.name)
        
        align()

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(dc, dcs)):
                with for_(*from_array(t, idle_times)):
                    readout_state(qubit, init_state)
                    qubit.align()
                    # Rotate the frame of the second x90 gate to implement a virtual Z-rotation
                    # 4*tau because tau was in clock cycles and 1e-9 because tau is ns
                    assign(phi, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t ))
                    assign(freq, Cast.mul_int_by_fixed(quads[qubit.name],dc*dc ))
                    assign(phi2, Cast.mul_fixed_by_int(1e-3, 4 * t * freq))
                    assign(phi, phi2-phi)
                    align()
                    # Strict_timing ensures that the sequence will be played without gaps
                                      
                    qubit.xy.play("x90")
                    qubit.xy.frame_rotation_2pi(phi)
                    qubit.z.play("const", amplitude_scale = dc / 0.1, duration=t)
                    wait(10, qubit.z.name)
                    qubit.align()
                    qubit.xy.play("x90")

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
            state_st[i].buffer(len(idle_times)).buffer(len(dcs)).average().save(f"state{i + 1}")


###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, ramsey, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
else:
    # Open the quantum machine
    qm = qmm.open_qm(config,keep_dc_offsets_when_closing=False)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
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
            
    qm.close()

# %%
# %%
if not simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, qubits, {"idle_time": idle_times, "flux": dcs})

    node.results = {}
    node.results['ds'] = ds
# %%
if not simulate:
    ds = ds.assign_coords(idle_time=4*ds.idle_time/1e3)  # convert to usec
    ds.flux.attrs = {'long_name': 'flux', 'units': 'V'}
    ds.idle_time.attrs = {'long_name': 'idle time', 'units': 'usec'}

    
# %%
if not simulate:
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

    decay = fit_data.sel(fit_vals = 'decay')
    decay.attrs = {'long_name' : 'decay', 'units' : 'nSec'}

    decay_res = fit_data.sel(fit_vals = 'decay_decay')
    decay_res.attrs = {'long_name' : 'decay', 'units' : 'nSec'}
    
    tau = 1/fit_data.sel(fit_vals='decay')
    tau.attrs = {'long_name' : 'T2*', 'units' : 'uSec'}

    tau_error = tau * (np.sqrt(decay_res)/decay)
    tau_error.attrs = {'long_name' : 'T2* error', 'units' : 'uSec'}

    # fitvals = frequency.polyfit(dim = 'flux', deg = 2)
    # flux= frequency.flux
    # a = {}
    # flux_offset = {}
    # freq_offset = {}
    # for q in machine.active_qubits:
    #     a[q.name] = float(-1e6*fitvals.sel(qubit = q.name,degree = 2).polyfit_coefficients.values)
    #     flux_offset[q.name]  = float((-0.5*fitvals.sel(qubit =q.name,degree = 1).polyfit_coefficients/fitvals.sel(qubit = q.name,degree = 2).polyfit_coefficients).values)
    #     freq_offset[q.name]  = 1e6*float(fitvals.sel(qubit = q.name,degree = 0).polyfit_coefficients.values) - detuning    

# %%
if not simulate:
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

    grid_names = [f'{q.name}_0' for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        tau_data = tau.sel(qubit = qubit['qubit'])
        flux_data = tau_data.flux
        ax.errorbar(flux_data, tau_data, 
                    yerr=tau_error.sel(qubit = qubit['qubit']), 
                    fmt='o-', capsize=5)
        ax.set_title(qubit['qubit'])
        ax.set_ylabel('T1 (uS)')
        ax.set_xlabel(' Flux (V)')
        ax.set_ylim(0, 40)
    grid.fig.suptitle('T1. Vs. flux')
    plt.tight_layout()
    plt.show()
    node.results['figure'] = grid.fig

# %%
# node.results['initial_parameters'] = node.parameters.model_dump()
# node.machine = machine
# node.save()
# # %%
# %%
from importlib import reload
from quam_libs.lib import fit
reload(fit)
from quam_libs.lib.fit import oscillation_decay_exp, fit_oscillation_decay_exp

# %%
