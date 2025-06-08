"""
        QUBIT SPECTROSCOPY VERSUS FLUX
This sequence involves doing a qubit spectroscopy for several flux biases in order to exhibit the qubit frequency
versus flux response.

Prerequisites:
    - Identification of the resonator's resonance frequency when coupled to the qubit in question (referred to as "resonator_spectroscopy").
    - Calibration of the IQ mixer connected to the qubit drive line (whether it's an external mixer or an Octave port).
    - Identification of the approximate qubit frequency ("qubit_spectroscopy").

Before proceeding to the next node:
    - Update the qubit frequency, in the state.
    - Update the relevant flux points in the state.
    - Update the frequency vs flux quadratic term in the state.
    - Save the current state
"""


# %% {Imports}
from datetime import datetime
from qualibrate import QualibrationNode, NodeParameters
from quam_libs.components import QuAM
from quam_libs.macros import qua_declaration, active_reset, readout_state
from quam_libs.trackable_object import tracked_updates
from quam_libs.lib.plot_utils import QubitGrid, grid_iter
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, get_node_id, save_node
from quam_libs.lib.fit import peaks_dips
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from scipy.optimize import curve_fit
import time
start = time.time()

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 10
    operation: str = "x180"
    operation_amplitude_factor: Optional[float] = 1
    duration_in_ns: Optional[int] = 400
    frequency_span_in_mhz: float = 200
    frequency_step_in_mhz: float = 0.2
    flux_amp : float = 0.08
    update_lo: bool = True
    fit_single_exponential: bool = True
    update_state: bool = False
    flux_point_joint_or_independent: Literal["joint", "independent"] = "joint"
    simulate: bool = False
    simulation_duration_ns: int = 2500
    timeout: int = 100
    load_data_id: Optional[int] = None
    multiplexed: bool = True
    reset_type_active_or_thermal: Literal['active', 'thermal'] = 'active'


node = QualibrationNode(name="97b_Pi_vs_flux_time", parameters=Parameters())
node_id = get_node_id()

# %% {Initialize_QuAM_and_QOP}
# Class containing tools to help handling units and conversions.
u = unit(coerce_to_integer=True)
# Instantiate the QuAM class from the state file
machine = QuAM.load()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)

# Modify the lo frequency to allow for maximum detuning 
tracked_qubits = []
if node.parameters.update_lo:
    for q in qubits:
        with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
            lo_band = q.xy.opx_output.band
            rf_frequency = q.xy.intermediate_frequency + q.xy.opx_output.upconverter_frequency
            lo_frequency = rf_frequency - 400e6
            if (lo_band == 3) and (lo_frequency < 6.5e9):
                lo_frequency = 6.5e9
            elif (lo_band == 2) and (lo_frequency < 4.5e9):
                lo_frequency = 4.5e9
            
            q.xy.intermediate_frequency = rf_frequency - lo_frequency
            q.xy.opx_output.upconverter_frequency = lo_frequency
            tracked_qubits.append(q)

# Generate the OPX and Octave configurations
config = machine.generate_config()

# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# %% {QUA_program}
n_avg = node.parameters.num_averages  # The number of averages
operation = node.parameters.operation  # The qubit operation to play
# Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
if node.parameters.operation_amplitude_factor:
    # pre-factor to the value defined in the config - restricted to [-2; 2)
    operation_amp = node.parameters.operation_amplitude_factor
else:
    operation_amp = 1.0
# Qubit detuning sweep with respect to their resonance frequencies
span = node.parameters.frequency_span_in_mhz * u.MHz
step = node.parameters.frequency_step_in_mhz * u.MHz
dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32)
# Flux bias sweep
times = np.arange(4, node.parameters.duration_in_ns // 4, 12, dtype=np.int32)
# times = np.logspace(np.log10(4), np.log10(node.parameters.duration_in_ns // 4), 30, dtype=np.int32)
flux_point = node.parameters.flux_point_joint_or_independent  # 'independent' or 'joint'
detuning = [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]

with program() as multi_qubit_spec_vs_flux:
    # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
    I, I_st, Q, Q_st, n, n_st = qua_declaration(num_qubits=num_qubits)
    state = [declare(int) for _ in range(num_qubits)]
    state_st = [declare_stream() for _ in range(num_qubits)]
    df = declare(int)  # QUA variable for the qubit frequency
    t_delay = declare(int)
    duration = node.parameters.duration_in_ns * u.ns
    for i, qubit in enumerate(qubits):
        # Bring the active qubits to the minimum frequency point
        machine.set_all_fluxes(flux_point=flux_point, target=qubit)

        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)

            with for_(*from_array(df, dfs)):
                # Update the qubit frequency
                
                # with for_(*from_array(t_delay, times)):
                with for_each_(t_delay, times):
                    if node.parameters.reset_type_active_or_thermal == "active":
                        active_reset(qubit)
                    else:
                        qubit.wait(qubit.thermalization_time * u.ns)                    # Flux sweeping for a qubit
                    qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detuning[i])
                    # Bring the qubit to the desired point during the saturation pulse
                    qubit.align()
                    qubit.z.play("const", amplitude_scale=node.parameters.flux_amp / qubit.z.operations["const"].amplitude, duration=t_delay+200)
                    # Apply saturation pulse to all qubits
                    # qubit.xy.wait(qubit.z.settle_time * u.ns)
                    qubit.xy.wait(t_delay)
                    qubit.xy.play(
                        operation,
                        amplitude_scale=operation_amp
                    )
                    qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                    qubit.align()
                    # QUA macro to read the state of the active resonators
                    readout_state(qubit, state[i])
                    save(state[i], state_st[i])
                    # qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                    # save data
                    # save(I[i], I_st[i])
                    # save(Q[i], Q_st[i])
                    # Wait for the qubits to decay to the ground state
                    qubit.resonator.wait(machine.depletion_time * u.ns)

        # Measure sequentially
        if not node.parameters.multiplexed:
            align()

    with stream_processing():
        n_st.save("n")
        for i, qubit in enumerate(qubits):
            # I_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"I{i + 1}")
            # Q_st[i].buffer(len(dcs)).buffer(len(dfs)).average().save(f"Q{i + 1}")
            state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")


# %% {Simulate_or_execute}
if node.parameters.simulate:
    # Simulates the QUA program for the specified duration
    simulation_config = SimulationConfig(duration=node.parameters.simulation_duration_ns * 4)  # In clock cycles = 4ns
    job = qmm.simulate(config, multi_qubit_spec_vs_flux, simulation_config)
    # Get the simulated samples and plot them for all controllers
    samples = job.get_simulated_samples()
    fig, ax = plt.subplots(nrows=len(samples.keys()), sharex=True)
    for i, con in enumerate(samples.keys()):
        plt.subplot(len(samples.keys()), 1, i+1)
        samples[con].plot()
        plt.title(con)
    plt.tight_layout()
    # Save the figure
    node.results = {"figure": plt.gcf()}
    node.machine = machine
    node.save()

elif node.parameters.load_data_id is None:
    date_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        job = qm.execute(multi_qubit_spec_vs_flux)
        results = fetching_tool(job, ["n"], mode="live")
        while results.is_processing():
            # Fetch results
            n = results.fetch_all()[0]
            # Progress bar
            progress_counter(n, n_avg, start_time=results.start_time)

# %%
######################################
# Helper functions for data analysis #
######################################

# Define the Gaussian
def gaussian(x, a, x0, sigma, offset):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2)) + offset

# Fit function for one time point
def fit_gaussian(freqs, states):
    p0 = [
        np.max(states) - np.min(states),   # amplitude
        freqs[np.argmax(states)],          # center
        (freqs[-1] - freqs[0]) / 10,        # width
        np.min(states)                     # offset
    ]
    try:
        popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
        return popt[1]  # center frequency
    except RuntimeError:
        return np.nan
    
def model_1exp(t, a0, a1, t1):
    return a0 * (1+ a1 * np.exp(-t / t1))

def model_2exp(t, a0, a1, a2, t1, t2):
    return a0 * (1 + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2))

def fit_two_exponentials(t_data: np.ndarray, y_data: np.ndarray, fit_single_exponential: bool):
    
    fit_results = {}
    try:
        popt, pcov = curve_fit(
            f=model_1exp, 
            xdata=t_data, 
            ydata=y_data, 
            p0=[np.max(y_data), np.min(y_data) / np.max(y_data) - 1, 300],
            bounds=([0, -np.inf, 0], [np.inf, np.inf, 300])  # Adding constraint for the third parameter to be below 300
            )
        fit_results['1exp'] = {"fit_successful": True, "params": popt, "covariance": pcov}
    except RuntimeError:
        print("failed to fit the first exponential")
        fit_results['1exp'] = {"fit_successful": False, "params": None, "covariance": None}
        if not fit_single_exponential:
            fit_results['2exp'] = {"fit_successful": False, "params": None, "covariance": None}
        return fit_results
    
    if not fit_single_exponential:
        # Use the first fit to initialize the second fit
        a0_0 = popt[0]
        a1_0 = a2_0 = popt[1] / 2 
        t1_0 = popt[-1]
        t2_0 = t1_0 / 10

        try:
            popt, pcov = curve_fit(
                f=model_2exp, 
                xdata=t_data, 
                ydata=y_data, 
                p0=[a0_0, a1_0, a2_0, t1_0, t2_0]
                )
            fit_results['2exp'] = {"fit_successful": True, "params": popt, "covariance": pcov}
        except RuntimeError:
            print("failed to fit the second exponential")
            fit_results['2exp'] = {"fit_successful": False, "params": None, "covariance": None}
            return fit_results
    
    return fit_results

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        machine = node.machine
        ds = xr.Dataset({"state": node.results["ds"].state})
        times = ds.time.values
        qubits = [machine.qubits[q] for q in ds.qubit.values]
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": times*4, "freq": dfs})
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term*node.parameters.flux_amp**2 for q in qubits]),
                ),
                "detuning": (
                    ["qubit", "freq"],
                    np.array([dfs + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]),
                ),
                "flux": (
                    ["qubit", "freq"],
                    np.array([np.sqrt(dfs / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp**2) for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}
    end = time.time()
    print(f"Script runtime: {end - start:.2f} seconds")

# %%  {Data_analysis}

freqs = ds['freq'].values

# Transpose to ensure ('qubit', 'time', 'freq') order
stacked = ds.transpose('qubit', 'time', 'freq')

# Now apply along 'freq' per (qubit, time)
center_freqs = xr.apply_ufunc(
    lambda states: fit_gaussian(freqs, states),
    stacked,
    input_core_dims=[['freq']],
    output_core_dims=[[]],  # no dimensions left after fitting
    vectorize=True,
    dask='parallelized',
    output_dtypes=[float]
).rename({"state": "center_frequency"})

# center_freqs now has dims ('qubit', 'time')
center_freqs = center_freqs.center_frequency + np.array([q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 * np.ones_like(times) for q in qubits])

flux_response = np.sqrt(center_freqs / xr.DataArray([q.freq_vs_flux_01_quad_term for q in qubits], coords={"qubit": center_freqs.qubit}, dims=["qubit"]))

ds['center_freqs'] = center_freqs
ds['flux_response'] = flux_response

fit_results = {}
for q in qubits:
    fit_results[q.name] = fit_two_exponentials(
        t_data=flux_response.sel(qubit=q.name).time.values, 
        y_data=flux_response.sel(qubit=q.name).values,
        fit_single_exponential=node.parameters.fit_single_exponential
    )

node.results["fit_results"] = fit_results

# %% {Plotting}
grid = QubitGrid(ds, [q.grid_location for q in qubits])

for ax, qubit in grid_iter(grid):
    im = ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
        ax=ax, add_colorbar=False, x="time", y="freq_GHz"
        
    )
    ax.set_ylabel("Freq (GHz)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
    cbar = grid.fig.colorbar(im, ax=ax)
    cbar.set_label("Qubit State")
grid.fig.suptitle(f"Qubit spectroscopy vs time after flux pulse \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_raw"] = grid.fig   


grid = QubitGrid(ds, [q.grid_location for q in qubits])

for ax, qubit in grid_iter(grid):
    (ds.loc[qubit].center_freqs / 1e9).plot(ax=ax)
    ax.set_ylabel("Freq (GHz)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
grid.fig.suptitle(f"Qubit frequency shift vs time after flux pulse \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_freqs_shift"] = grid.fig


grid = QubitGrid(ds, [q.grid_location for q in qubits])

for ax, qubit in grid_iter(grid):
    ds.loc[qubit].flux_response.plot(ax=ax)

    if not node.parameters.fit_single_exponential and fit_results[qubit["qubit"]]["2exp"]["fit_successful"]:
        popt = fit_results[qubit["qubit"]]["2exp"]["params"]
        t_data = flux_response.sel(qubit=qubit["qubit"]).time.values
        y_data = flux_response.sel(qubit=qubit["qubit"]).values
        y_fit = model_2exp(t_data, *popt)
        ax.plot(t_data, y_fit, 'r-')
        fit_text = f'a0={popt[0]:.3f}\na1={popt[1]:.3f}\na2={popt[2]:.3f}\nt1={popt[3]:.0f}ns\nt2={popt[4]:.0f}ns'
        ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, verticalalignment='top', fontsize=8)
    
    elif fit_results[qubit["qubit"]]["1exp"]["fit_successful"]:
        popt = fit_results[qubit["qubit"]]["1exp"]["params"]
        t_data = flux_response.sel(qubit=qubit["qubit"]).time.values
        y_data = flux_response.sel(qubit=qubit["qubit"]).values
        y_fit = model_1exp(t_data, *popt)
        ax.plot(t_data, y_fit, 'r-')
        fit_text = f'a0={popt[0]:.3f}\na1={popt[1]:.3f}\nt1={popt[2]:.0f}ns'
        ax.text(0.02, 0.98, fit_text, transform=ax.transAxes, verticalalignment='top', fontsize=8)

    ax.set_ylabel("Flux (V)")
    ax.set_xlabel("Time (ns)")
    ax.set_title(qubit["qubit"])
grid.fig.suptitle(f"Flux response vs time \n {date_time} #{node_id}")

plt.tight_layout()
plt.show()
node.results["figure_flux_response"] = grid.fig

# %% {Update_state}
for q in tracked_qubits:
    q.revert_changes()

if node.parameters.load_data_id is None:
    if node.parameters.update_state:
        with node.record_state_updates():
            for q in qubits:
                if not node.parameters.fit_single_exponential and fit_results[q.name]["2exp"]["fit_successful"]:
                    q.z.opx_output.exponential_filter = [
                        [fit_results[q.name]["2exp"]["params"][1], fit_results[q.name]["2exp"]["params"][3]],
                        [fit_results[q.name]["2exp"]["params"][2], fit_results[q.name]["2exp"]["params"][4]]
                        ]
                elif fit_results[q.name]["1exp"]["fit_successful"]:
                    q.z.opx_output.exponential_filter = [
                        [fit_results[q.name]["1exp"]["params"][1], fit_results[q.name]["1exp"]["params"][2]]
                        ]
                    print("updated the exponential filter")



# %% {Save_results}
node.results["ds"] = ds
node.outcomes = {q.name: "successful" for q in qubits}
node.results["initial_parameters"] = node.parameters.model_dump()
node.machine = machine
save_node(node)
# %%
