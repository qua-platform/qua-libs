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
from quam_libs.lib.qua_datasets import convert_IQ_to_V
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
import time
start = time.time()

# %% {Node_parameters}
class Parameters(NodeParameters):

    qubits: Optional[List[str]] = None
    num_averages: int = 100
    operation: str = "x180_Gaussian"
    operation_amplitude_factor: Optional[float] = 1
    duration_in_ns: Optional[int] = 500
    frequency_span_in_mhz: float = 400
    frequency_step_in_mhz: float = 0.5
    flux_amp : float = 0.05
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
# Generate the OPX and Octave configurations
config = machine.generate_config()
# Open Communication with the QOP
if node.parameters.load_data_id is None:
    qmm = machine.connect()

# Get the relevant QuAM components
if node.parameters.qubits is None or node.parameters.qubits == "":
    qubits = machine.active_qubits
else:
    qubits = [machine.qubits[q] for q in node.parameters.qubits]
num_qubits = len(qubits)


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
        plt.subplot(len(samples.keys()),1,i+1)
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

# %% {Data_fetching_and_dataset_creation}
if not node.parameters.simulate:
    if node.parameters.load_data_id is not None:
        node = node.load_from_id(node.parameters.load_data_id)
        ds = node.results["ds"]
    else:
        # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
        ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": times*4, "freq": dfs})
        # Convert IQ data into volts
        # ds = convert_IQ_to_V(ds, qubits)
        # Derive the amplitude IQ_abs = sqrt(I**2 + Q**2)
        # ds = ds.assign({"IQ_abs": np.sqrt(ds["I"] ** 2 + ds["Q"] ** 2)})
        # Add the resonator RF frequency axis of each qubit to the dataset coordinates for plotting
        ds = ds.assign_coords(
            {
                "freq_full": (
                    ["qubit", "freq"],
                    np.array([dfs + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term*node.parameters.flux_amp**2 for q in qubits]),
                )
            }
        )
        ds.freq_full.attrs["long_name"] = "Frequency"
        ds.freq_full.attrs["units"] = "GHz"
    # Add the dataset to the node
    node.results = {"ds": ds}
    end = time.time()
    print(f"Script runtime: {end - start:.2f} seconds")

    # %% {Data_analysis}

    import numpy as np
    import xarray as xr
    from scipy.optimize import curve_fit

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
    )

    # center_freqs now has dims ('qubit', 'time')
    center_freqs = center_freqs.rename({"state": "center_frequency"})
    center_freqs = center_freqs.center_frequency + [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits][0]
    flux_response = np.sqrt(center_freqs/qubits[0].freq_vs_flux_01_quad_term)
    flux_response.plot()
    

    # %%

    # Define your model function
    def model(t, a0, a1, t1):
        return a0 * (1+ a1 * np.exp(-t / t1))

    t_data = flux_response.time.values
    y_data = flux_response.isel(qubit=0).values

    # Fit the data
    popt, pcov = curve_fit(model, t_data, y_data, p0=[np.max(y_data), np.min(y_data) / np.max(y_data) - 1, 1000])  # p0 = initial guess

    # Plot
    plt.figure()
    plt.scatter(t_data, y_data, label='Data')
    plt.plot(t_data, model(t_data, *popt), 'r-', label=f'Fit: a0={popt[0]:.2f}, a1={popt[1]:.2f}, t1={popt[2]:.2f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # %% {Plotting}

    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        freq_ref = (ds.freq_full-ds.freq).sel(qubit = qubit["qubit"]).values[0]
        ds.assign_coords(freq_GHz=ds.freq_full / 1e9).loc[qubit].state.plot(
            ax=ax, add_colorbar=False, x="time", y="freq_GHz"
            
        )
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        # ax.set_xscale('log')
    grid.fig.suptitle(f"Qubit spectroscopy vs time after flux pulse \n {date_time} #{node_id}")
    
    plt.tight_layout()
    plt.show()
    node.results["figure_raw"] = grid.fig   


    grid = QubitGrid(ds, [q.grid_location for q in qubits])

    for ax, qubit in grid_iter(grid):
        added_freq = machine.qubits[qubit['qubit']].xy.RF_frequency*0 + machine.qubits[qubit['qubit']].freq_vs_flux_01_quad_term*node.parameters.flux_amp**2
        (-(center_freqs.sel(qubit = qubit["qubit"])+ added_freq)/1e9).plot()
        ax.set_ylabel("Freq (GHz)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        # ax.set_xscale('log')
        ax.grid()
    grid.fig.suptitle(f"Qubit spectroscopy vs flux \n {date_time} #{node_id}")
    
    plt.tight_layout()
    plt.show()
    node.results["figure_freqs"] = grid.fig

    grid = QubitGrid(ds, [q.grid_location for q in qubits])
    for ax, qubit in grid_iter(grid):
        (1e3*flux_response).plot(ax=ax, marker='.')
        ax.set_ylabel("Flux (mV)")
        ax.set_xlabel("Time (ns)")
        ax.set_title(qubit["qubit"])
        # ax.set_xscale('log')
        ax.grid()
    grid.fig.suptitle(f"Qubit spectroscopy vs flux \n {date_time} #{node_id}")
    
    plt.tight_layout()
    plt.show()
    node.results["figure_flux"] = grid.fig

 

    # %% {Update_state}


    # %% {Save_results}
    node.results["ds"] = ds
    node.outcomes = {q.name: "successful" for q in qubits}
    node.results["initial_parameters"] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)

# %%


    # Define your model function
    def model_1exp(t, a0, a1, t1):
        return a0 * (1+ a1 * np.exp(-t / t1))
    
    def model_2exp(t, a0, a1, a2, t1, t2):
        return a0 * (1 + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2))
    

    t_data = flux_response.time.values
    y_data = flux_response.isel(qubit=0).values

    # Fit the data
    popt, pcov = curve_fit(model_1exp, t_data, y_data, p0=[np.max(y_data), np.min(y_data) / np.max(y_data) - 1, 1000])

    a0_0 = popt[0]
    a1_0 = a2_0 = popt[1] / 2 # np.real(amplitudes_esprit)
    t1_0 = popt[-1] # lambda1_0, lambda2_0 = np.real(lambdas_esprit)
    t2_0 = 100

    initial_guess = [a0_0, a1_0, a2_0, t1_0, t2_0] # [-0.0015, -1 / 200, a2_0, lambda2_0, c_0]

    # Perform nonlinear curve fitting
    popt, pcov = curve_fit(model_2exp, t_data, y_data, p0=initial_guess)

    y_fit = model_2exp(t_data, *popt)
    # y_fit = model(t_data, -0.0015, -1 / 200, popt[2], popt[3], popt[4])

    # Plot
    plt.figure()
    plt.scatter(t_data, y_data, label='Data')
    plt.plot(t_data, y_fit, 'r-', label=f'Fit: a0={popt[0]:.3f}, a1={popt[1]:.3f}, a2={popt[2]:.3f}, t1={popt[3]:.0f}, t2={popt[4]:.0f}')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
# %%
