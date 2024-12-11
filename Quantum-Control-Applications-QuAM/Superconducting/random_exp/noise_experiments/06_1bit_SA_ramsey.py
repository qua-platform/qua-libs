# %%
"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, labeled as "f_01", in the state.
    - Update the relevant flux points in the state.
    - Save the current state by calling machine.save("quam")
"""
from qualibrate import QualibrationNode, NodeParameters
from typing import Optional, Literal, List


class Parameters(NodeParameters):
    qubits: Optional[List[str]] = None
    num_averages: int = 1000
    dc_offset: float = 0.015
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False

node = QualibrationNode(
    name="99_1bit_SA_ramsey",
    parameters_class=Parameters
)

node.parameters = Parameters()


from qm.qua import *
from qm import SimulationConfig
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.plot import interrupt_on_close
from qualang_tools.loops import from_array
from qualang_tools.units import unit
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset
import xarray as xr
import xrft
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

import matplotlib
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray
from quam_libs.lib.fit import peaks_dips

# matplotlib.use("TKAgg")


###################################################
#  Load QuAM and open Communication with the QOP  #
###################################################
# Class containing tools to help handling units and conversions.
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
    qubits = [machine.qubits[q] for q in node.parameters.qubits.replace(' ', '').split(',')]
num_qubits = len(qubits)

freqs = np.arange(-5000000, -2000000, 25000, dtype=np.int32)  # Integer values from -5e6 to 0 with step 50000
idle_times = np.arange(420, 601, 100, dtype=np.int32)  # Integer values from 20 to 1000 with step 100
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
dc = node.parameters.dc_offset
n_avg = node.parameters.num_averages
###################
# The QUA program #
###################

# %% program for finding optimal freq offset and idle time
with program() as find_optimal_freq_offset_and_idle_time:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    freq = declare(int)  # QUA variable for the flux dc level
    idle_time = declare(int)

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.set_dc_offset(dc + qubit.z.independent_offset)
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
            qubit.z.set_dc_offset(dc + qubit.z.joint_offset)
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(freq, freqs)):
                with for_(*from_array(idle_time, idle_times)):
                    update_frequency(qubit.xy.name, freq + qubit.xy.intermediate_frequency)
                    active_reset(machine, qubit.name)
                    qubit.xy.play("x90")
                    wait(idle_time / 4, qubit.xy.name)
                    qubit.xy.play("x90")
                    align()
                    # Measure the state of the resonators
                    qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
                    save(state[i], state_st[i])
                    reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(idle_times)).buffer(len(freqs)).average().save(f"state{i + 1}")



# %%

###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, find_optimal_freq_offset_and_idle_time, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
else:
    # Open the quantum machine
    qm = qmm.open_qm(config,keep_dc_offsets_when_closing=False)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(find_optimal_freq_offset_and_idle_time)
    # Get results from QUA program
    for i in range(num_qubits):
        print(f"Fetching results for qubit {qubits[i].name}")
        data_list = ["n"] + sum([[f"state{i + 1}"] ], [])
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
    ds = fetch_results_as_xarray(handles, qubits, {"idle_time": idle_times, "freq": freqs})

    node.results = {}
    node.results['ds'] = ds

# %%

# %%
   
idle_time_to_run = 520 
opt_freq = {}
if not simulate:
    grid_names = [f'{q.name}_0' for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        opt_freq[qubit['qubit']] = np.abs(ds.sel(qubit = qubit['qubit']).state.sel(idle_time=idle_time_to_run)-0.45).idxmin('freq')
        ds.sel(qubit = qubit['qubit']).state.sel(idle_time=idle_time_to_run).plot(ax =ax)
        ax.axhline(0.45, color='k')
        ax.plot(opt_freq[qubit['qubit']], 0.45, 'o')
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('State')
    grid.fig.suptitle('Avg state vs. detuning')
    plt.tight_layout()
    plt.show()
    node.results['figure_raw'] = grid.fig
# %%
n_avg = 2_000_000


# %% create program for T2 spectoscopy
with program() as Ramsey_noise_spec:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]

    for i, qubit in enumerate(qubits):

        # Bring the active qubits to the minimum frequency point
        if flux_point == "independent":
            machine.apply_all_flux_to_min()
            qubit.z.set_dc_offset(dc + qubit.z.independent_offset)
        elif flux_point == "joint":
            machine.apply_all_flux_to_joint_idle()
            qubit.z.set_dc_offset(dc + qubit.z.joint_offset)
        else:
            machine.apply_all_flux_to_zero()
        wait(1000)
        update_frequency(qubit.xy.name, int(opt_freq[qubit.name]) + qubit.xy.intermediate_frequency)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            active_reset(machine, qubit.name)
            qubit.xy.play("x90", timestamp_stream=f'time_stamp{i+1}')
            wait(idle_time_to_run // 4, qubit.xy.name)
            qubit.xy.play("x90")
            align()
            # Measure the state of the resonators
            qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
            assign(state[i], Cast.to_int(I[i] > qubit.resonator.operations["readout"].threshold))
            save(state[i], state_st[i])
            reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(n_avg).save(f"state{i + 1}")


# %% 
###########################
# Run or Simulate Program #
###########################
simulate = node.parameters.simulate

if simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=10_000)  # In clock cycles = 4ns
    job = qmm.simulate(config, Ramsey_noise_spec, simulation_config)
    job.get_simulated_samples().con1.plot()
    node.results = {"figure": plt.gcf()}
else:
    # Open the quantum machine
    qm = qmm.open_qm(config,keep_dc_offsets_when_closing=False)
    # Calibrate the active qubits
    # machine.calibrate_octave_ports(qm)
    # Send the QUA program to the OPX, which compiles and executes it
    job = qm.execute(Ramsey_noise_spec)
    # Get results from QUA program
    for i in range(num_qubits):
        print(f"Fetching results for qubit {qubits[i].name}")
        data_list = ["n"] + sum([[f"state{i + 1}"] ], [])
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
if not simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, qubits, {"n": np.arange(0,n_avg,1)})

    extracted_values = np.array([v[0] for v in ds.time_stamp.values.flatten()]).reshape(ds.time_stamp.values.shape)
    ds['time_stamp'] = xr.DataArray(extracted_values, dims=ds['time_stamp'].dims, coords=ds['time_stamp'].coords)
    ds['time_stamp'] = ds['time_stamp']*4
    node.results['ds_final'] = ds

# %%
if not simulate:
    grid_names = [f'{q.name}_0' for q in qubits]
    grid = QubitGrid(ds, grid_names)
    for ax, qubit in grid_iter(grid):
        ds.sel(qubit = qubit['qubit']).state.plot.hist(bins=3,ax=ax)
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('State')
        ax.set_ylabel('Counts')
    grid.fig.suptitle('Histogram of qubit states')
    plt.tight_layout()
    plt.show()
    node.results['figure_bins'] = grid.fig

# %%
if not simulate:
    dat_fft = {}    

    for qubit in qubits:
        data_q = ds.state.sel(qubit = qubit.name)
        time_stamp_q = ds.time_stamp.sel(qubit = qubit.name).values
        
        f, Pxx_den = signal.welch(data_q-data_q.mean(),  1e9/np.mean(np.diff(time_stamp_q)), 
                          nperseg=8192*8)
        dat_fft[qubit.name] = xr.Dataset({'Pxx_den': (['freq'], Pxx_den)}, coords={'freq': f}).Pxx_den

        # dat_fft[qubit.name] = xrft.power_spectrum(data_q, real_dim='n')
        # dat_fft[qubit.name] = dat_fft[qubit.name].assign_coords(freq_n=1e9*dat_fft[qubit.name].freq_n/np.mean(np.diff(time_stamp_q)))
    
# %%    
if not simulate:
    grid_names = [f'{q.name}_0' for q in qubits]
    grid = QubitGrid(ds, grid_names, size = 5)
    for ax, qubit in grid_iter(grid):
        dat_fft[qubit['qubit']].plot(yscale='log', xscale='log', ax =ax)
        ax.grid(which='both')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('power spectrum [arb.]')
    grid.fig.suptitle('Histogram of qubit states')
    plt.tight_layout()
    node.results['figure_fft'] = grid.fig


# %%
# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
