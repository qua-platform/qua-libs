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
    qubits: Optional[List[str]] = ["qubitC2"]
    num_averages: int = 1000
    dc_offset: float = 0.01
    flux_point_joint_or_independent: Literal['joint', 'independent'] = "joint"
    simulate: bool = False
    timeout: int = 100

node = QualibrationNode(
    name="99_1bit_SA_ramsey",
    parameters_class=Parameters
)

node.parameters = Parameters()
simulate = node.parameters.simulate

# from qm.qua import *
# from qm import SimulationConfig
# from qualang_tools.results import progress_counter, fetching_tool
# from qualang_tools.plot import interrupt_on_close
# from qualang_tools.loops import from_array
# from qualang_tools.units import unit
# from quam_libs.components import QuAM
# from quam_libs.macros import qua_declaration, active_reset
import xarray as xr
import xrft
# import matplotlib.pyplot as plt
# import numpy as np
from scipy import signal

# import matplotlib
# from quam_libs.lib.plot_utils import QubitGrid, grid_iter
# from quam_libs.lib.save_utils import fetch_results_as_xarray
# from quam_libs.lib.fit import peaks_dips

from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, readout_state, active_reset
from quam_libs.lib.qua_datasets import convert_IQ_to_V
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset
from quam_libs.lib.fit import fit_oscillation_decay_exp, oscillation_decay_exp
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np


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
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

freqs_MHZ = np.arange(-5, -1, 25e-3)  # Integer values from -5e6 to 0 with step 50000

idle_time = 520  # Integer values from 20 to 1000 with step 100
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
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]    
    # freq = declare(int)  # QUA variable for the flux dc level
    phi = declare(fixed)
    t = declare(int)
    assign(t, idle_time >> 2)
    # dt = declare(fixed, 1e-9)
    freq_MHZ = declare(fixed)

    for i, qubit in enumerate(qubits):
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)            
        wait(1000)
        
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            
            with for_(*from_array(freq_MHZ, freqs_MHZ)):        
                    
                assign(phi, Cast.mul_fixed_by_int(freq_MHZ * 1e-3, idle_time))
                
                qubit.align()

                qubit.xy.play("x90")
                qubit.xy.frame_rotation_2pi(phi)
                qubit.z.wait(duration=qubit.xy.operations["x180"].length)
                
                qubit.xy.wait(t + 1)
                qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=t)
                
                qubit.xy.play("x90")
                
                qubit.align()
                
                # Measure the state of the resonators
                readout_state(qubit, state[i])
                assign(final_state[i], init_state[i] ^ state[i])
                save(final_state[i], state_st[i])
                # save(freq_GHZ, state_st[i])
                
                assign(init_state[i], state[i])
                reset_frame(qubit.xy.name)

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubits):
            state_st[i].buffer(len(freqs_MHZ)).average().save(f"state{i + 1}")



# %%

###########################
# Run or Simulate Program #
###########################
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, iq_blobs, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(find_optimal_freq_offset_and_idle_time)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg)



# %%
if not simulate:
    handles = job.result_handles
    ds = fetch_results_as_xarray(handles, qubits, {"freq": freqs_MHZ})

    node.results = {}
    node.results['ds'] = ds

# %%
   
opt_freq = {}
if not simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        opt_freq[qubit['qubit']] = np.abs(ds.sel(qubit = qubit['qubit']).state-0.5).idxmin('freq')
        ds.sel(qubit = qubit['qubit']).state.plot(ax =ax)
        ax.axhline(0.5, color='k')
        ax.plot(opt_freq[qubit['qubit']], 0.5, 'o')
        ax.set_title(qubit['qubit'])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('State')
    grid.fig.suptitle('Avg state vs. detuning')
    plt.tight_layout()
    plt.show()
    node.results['figure_raw'] = grid.fig
# %%
n_avg = 2**23 + 1

phis = {qubit.name: (opt_freq[qubit.name].values * 1e-3 * idle_time)/(2*np.pi) for qubit in qubits}
phis
# %% create program for T2 spectoscopy
with program() as Ramsey_noise_spec:

    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    init_state = [declare(int) for _ in range(num_qubits)]
    final_state = [declare(int) for _ in range(num_qubits)]
    t = declare(int)
    assign(t, idle_time >> 2)
    phi = declare(fixed)

    for i, qubit in enumerate(qubits):
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)                  
        wait(1000)
        # update_frequency(qubit.xy.name, int(opt_freq[qubit.name]) + qubit.xy.intermediate_frequency)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            assign(phi, phis[qubit.name])
            qubit.align()
            # with strict_timing_():
            # update_frequency(qubit.xy.name, int(opt_freq[qubit.name])  + qubit.xy.intermediate_frequency)
            qubit.xy.play("x90",  timestamp_stream=f'time_stamp{i+1}')
            qubit.xy.frame_rotation_2pi(phi)
            qubit.z.wait(duration=qubit.xy.operations["x180"].length)
            
            qubit.xy.wait(t + 1)
            qubit.z.play("const", amplitude_scale=dc / qubit.z.operations["const"].amplitude, duration=t)
            
            qubit.xy.play("x90")
            # update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency)
            qubit.align()            
            
            # Measure the state of the resonators
            readout_state(qubit, state[i])
            assign(final_state[i], init_state[i] ^ state[i])
            save(final_state[i], state_st[i])
            assign(init_state[i], state[i])
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

if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, Ramsey_noise_spec, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()),1,i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()
    
else:
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(Ramsey_noise_spec)
        for i in range(num_qubits):
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                n = results.fetch_all()[0]
                progress_counter(n, n_avg)


# %%
if not simulate:
    def extract_string(input_string):
        # Find the index of the first occurrence of a digit in the input string
        index = next((i for i, c in enumerate(input_string) if c.isdigit()), None)

        if index is not None:
            # Extract the substring from the start of the input string to the index
            extracted_string = input_string[:index]
            return extracted_string
        else:
            return None
        
    stream_handles = job.result_handles.keys()
    meas_vars = list(set([extract_string(handle) for handle in stream_handles if extract_string(handle) is not None]))
    meas_vars = meas_vars[::-1]
    values = np.array(
        [
    np.array(np.array( [job.result_handles.get(f"time_stamp{i + 1}").fetch_all() for i, qubit in enumerate(qubits)]).tolist()).squeeze(-1),
    np.array([job.result_handles.get(f"state{i + 1}").fetch_all() for i, qubit in enumerate(qubits)]),
    ]
        
        )

    if np.array(values).shape[-1] == 1:
        values = np.array(values).squeeze(axis=-1)

    measurement_axis = {"n": np.arange(0,n_avg)}
        
    measurement_axis["qubit"] = [qubit.name for qubit in qubits]
    measurement_axis = {key: measurement_axis[key] for key in reversed(measurement_axis.keys())}


    ds = xr.Dataset(
        {f"{meas_var}": ([key for key in measurement_axis.keys()], values[i]) for i, meas_var in enumerate(meas_vars)},
        coords=measurement_axis,
    )
    ds['time_stamp'] = ds['time_stamp']*4

    node.results['ds_final'] = ds
# %%


# %%
if not simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
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
                          nperseg=2**17)
        dat_fft[qubit.name] = xr.Dataset({'Pxx_den': (['freq'], Pxx_den)}, coords={'freq': f}).Pxx_den

        # dat_fft[qubit.name] = xrft.power_spectrum(data_q, real_dim='n')
        # dat_fft[qubit.name] = dat_fft[qubit.name].assign_coords(freq_n=1e9*dat_fft[qubit.name].freq_n/np.mean(np.diff(time_stamp_q)))
    
# %%    
if not simulate:
    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        dat_fft[qubit['qubit']].plot(yscale='log', xscale='log', ax =ax, marker='.')
        ax.grid(which='both')
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('power spectrum [arb.]')
        # ax.set_xlim(1e2, 1e3)
        
    grid.fig.suptitle('Histogram of qubit states')
    plt.tight_layout()
    node.results['figure_fft'] = grid.fig


# %%
# %%
node.results['initial_parameters'] = node.parameters.model_dump()
node.machine = machine
node.save()
# %%
