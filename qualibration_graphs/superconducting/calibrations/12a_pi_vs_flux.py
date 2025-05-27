# %% {Imports}
import dataclasses
from datetime import datetime
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.pi_flux import Parameters
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from scipy.optimize import curve_fit

from quam.components.pulses import DragGaussianPulse, GaussianPulse

# %% {Node_parameters}
description = """
        ....
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="12a_pi_vs_flux",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubits = ["qC1"]
    node.parameters.num_shots = 20
    node.parameters.flux_amp = 0.07
    node.parameters.update_lo = True
    node.parameters.frequency_span_in_mhz = 200
    node.parameters.frequency_step_in_mhz = 1
    node.parameters.operation_amplitude_factor = 0.8

    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    node.namespace["qubits"] = qubits = get_qubits(node)
    n_avg = node.parameters.num_shots  # The number of averages
    operation = node.parameters.operation  # The qubit operation to play

    for qubit in qubits:
        # Check if the qubit has the required operations
        if hasattr(qubit.xy.operations, operation):
            continue
        else:
            x180 = qubit.xy.operations["x180"]
            # qubit.xy.operations[operation] = GaussianPulse(
            #     length=int(x180.length), amplitude=float(x180.amplitude), sigma=int(x180.length / 4)
            # )
            qubit.xy.operations[operation] = DragGaussianPulse(
                length=int(x180.length), amplitude=float(x180.amplitude), sigma=int(x180.length / 4), alpha=0.0, anharmonicity=200e6, axis_angle=0.0
            )
    # Modify the lo frequency to allow for maximum detuning
    tracked_qubits = []
    if node.parameters.update_lo:
        for q in qubits:
            with tracked_updates(q, auto_revert=False, dont_assign_to_none=True) as q:
                q.xy.opx_output.upconverter_frequency -= 300e6
                # if q.xy.upconverter_frequency < 4.5e9:
                #     q.xy.opx_output.band = 1
                q.xy.RF_frequency -= 400e6
                tracked_qubits.append(q)

    # Adjust the pulse duration and amplitude to drive the qubit into a mixed state - can be None
    if node.parameters.operation_amplitude_factor:
        # pre-factor to the value defined in the config - restricted to [-2; 2)
        operation_amp = node.parameters.operation_amplitude_factor
    else:
        operation_amp = 1.0
    # Qubit detuning sweep with respect to their resonance frequencies
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, span // 2, step, dtype=np.int32).tolist()
    times = np.arange(4, node.parameters.duration_in_ns // 4, 12, dtype=np.int32).tolist()
    # times = np.logspace(np.log10(4), np.log10(node.parameters.duration_in_ns // 4), 30, dtype=np.int32)

    detuning = [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(dfs, attrs={"long_name": "qubit frequency", "units": "Hz"}),
        "time": xr.DataArray(times, attrs={"long_name": "Flux pulse duration", "units": "ns"}),
    }

    with program() as node.namespace["qua_program"]:
        # Macro to declare I, Q, n and their respective streams for a given number of qubit (defined in macros.py)
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state = [declare(int) for _ in range(num_qubits)]
            state_st = [declare_stream() for _ in range(num_qubits)]
        df = declare(int)  # QUA variable for the qubit frequency
        t_delay = declare(int)
        # duration = node.parameters.duration_in_ns * u.ns

        for multiplexed_qubits in qubits.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    # Update the qubit frequency
                    # with for_(*from_array(t_delay, times)):
                    with for_each_(t_delay, times):
                        # Qubit initialization
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency + detuning[i])
                            # Bring the qubit to the desired point during the saturation pulse
                            qubit.align()
                            qubit.z.play(
                                "const",
                                amplitude_scale=node.parameters.flux_amp / qubit.z.operations["const"].amplitude,
                                duration=t_delay + 200,
                            )
                            # Apply saturation pulse to all qubits
                            # qubit.xy.wait(qubit.z.settle_time * u.ns)
                            qubit.xy.wait(t_delay)
                            qubit.xy.play(operation, amplitude_scale=operation_amp)
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                        align()
                        # Qubit readout
                        for i, qubit in multiplexed_qubits.items():
                            if node.parameters.use_state_discrimination:
                                qubit.readout_state(state[i])
                                save(state[i], state_st[i])
                            else:
                                qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                                save(I[i], I_st[i])
                                save(Q[i], Q_st[i])
                        align()

        with stream_processing():
            n_st.save("n")
            for i, qubit in enumerate(qubits):
                if node.parameters.use_state_discrimination:
                    state_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"state{i + 1}")
                else:
                    I_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"I{i + 1}")
                    Q_st[i].buffer(len(times)).buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    # Store the figure, waveform report and simulated samples
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Execute the QUA program only if the quantum machine is available (this is to avoid interrupting running jobs).
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        # The job is stored in the node namespace to be reused in the fetching_data run_action
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        # Display the progress bar
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher["n"],
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset
    node.results["ds_raw"] = dataset

    dataset.I.sel(qubit="qC1").plot()


# %%
######################################
# Helper functions for data analysis #
######################################


# # Define the Gaussian
# def gaussian(x, a, x0, sigma, offset):
#     return a * np.exp(-((x - x0) ** 2) / (2 * sigma**2)) + offset


# # Fit function for one time point
# def fit_gaussian(freqs, states):
#     p0 = [
#         np.max(states) - np.min(states),  # amplitude
#         freqs[np.argmax(states)],  # center
#         (freqs[-1] - freqs[0]) / 10,  # width
#         np.min(states),  # offset
#     ]
#     try:
#         popt, _ = curve_fit(gaussian, freqs, states, p0=p0)
#         return popt[1]  # center frequency
#     except RuntimeError:
#         return np.nan


# def model_1exp(t, a0, a1, t1):
#     return a0 * (1 + a1 * np.exp(-t / t1))


# def model_2exp(t, a0, a1, a2, t1, t2):
#     return a0 * (1 + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2))


# # %% {Data_fetching_and_dataset_creation}
# if not node.parameters.simulate:
#     if node.parameters.load_data_id is not None:
#         node = node.load_from_id(node.parameters.load_data_id)
#         machine = node.machine
#         ds = xr.Dataset({"state": node.results["ds"].state})
#         times = ds.time.values
#         qubits = [machine.qubits[q] for q in ds.qubit.values]
#     else:
#         # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
#         ds = fetch_results_as_xarray(job.result_handles, qubits, {"time": times * 4, "freq": dfs})
#         ds = ds.assign_coords(
#             {
#                 "freq_full": (
#                     ["qubit", "freq"],
#                     np.array(
#                         [
#                             dfs + q.xy.RF_frequency + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2
#                             for q in qubits
#                         ]
#                     ),
#                 ),
#                 "detuning": (
#                     ["qubit", "freq"],
#                     np.array([dfs + q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 for q in qubits]),
#                 ),
#                 "flux": (
#                     ["qubit", "freq"],
#                     np.array(
#                         [np.sqrt(dfs / q.freq_vs_flux_01_quad_term + node.parameters.flux_amp**2) for q in qubits]
#                     ),
#                 ),
#             }
#         )
#         ds.freq_full.attrs["long_name"] = "Frequency"
#         ds.freq_full.attrs["units"] = "GHz"
#     # Add the dataset to the node
#     node.results = {"ds": ds}
#     end = time.time()
#     print(f"Script runtime: {end - start:.2f} seconds")

#     # %%  {Data_analysis}

#     freqs = ds["freq"].values

#     # Transpose to ensure ('qubit', 'time', 'freq') order
#     stacked = ds.transpose("qubit", "time", "freq")

#     # Now apply along 'freq' per (qubit, time)
#     center_freqs = xr.apply_ufunc(
#         lambda states: fit_gaussian(freqs, states),
#         stacked,
#         input_core_dims=[["freq"]],
#         output_core_dims=[[]],  # no dimensions left after fitting
#         vectorize=True,
#         dask="parallelized",
#         output_dtypes=[float],
#     ).rename({"state": "center_frequency"})

#     # center_freqs now has dims ('qubit', 'time')
#     center_freqs = center_freqs.center_frequency + np.array(
#         [q.freq_vs_flux_01_quad_term * node.parameters.flux_amp**2 * np.ones_like(times) for q in qubits]
#     )

#     flux_response = np.sqrt(
#         center_freqs
#         / xr.DataArray(
#             [q.freq_vs_flux_01_quad_term for q in qubits], coords={"qubit": center_freqs.qubit}, dims=["qubit"]
#         )
#     )

#     ds["center_freqs"] = center_freqs
#     ds["flux_response"] = flux_response

# # %% {Plotting}
# grid = QubitGrid(ds, [q.grid_location for q in qubits])

# for ax, qubit in grid_iter(grid):
#     # freq_ref = (ds.freq_full-ds.freq).sel(qubit=qubit["qubit"]).values[0]
#     im = (
#         ds.assign_coords(freq_GHz=ds.freq_full / 1e9)
#         .loc[qubit]
#         .state.plot(ax=ax, add_colorbar=False, x="time", y="freq_GHz")
#     )
#     ax.set_ylabel("Freq (GHz)")
#     ax.set_xlabel("Time (ns)")
#     ax.set_title(qubit["qubit"])
#     cbar = grid.fig.colorbar(im, ax=ax)
#     cbar.set_label("Qubit State")
# grid.fig.suptitle(f"Qubit spectroscopy vs time after flux pulse \n {date_time} #{node_id}")

# plt.tight_layout()
# plt.show()
# node.results["figure_raw"] = grid.fig


# grid = QubitGrid(ds, [q.grid_location for q in qubits])

# for ax, qubit in grid_iter(grid):
#     # added_freq = machine.qubits[qubit['qubit']].xy.RF_frequency*0 + machine.qubits[qubit['qubit']].freq_vs_flux_01_quad_term * node.parameters.flux_amp**2
#     # (-(center_freqs.sel(qubit=qubit["qubit"]) + added_freq)/1e9).plot()
#     # (center_freqs.sel(qubit=qubit["qubit"]) / 1e9).plot(ax=ax)
#     (ds.loc[qubit].center_freqs / 1e9).plot(ax=ax)
#     ax.set_ylabel("Freq (GHz)")
#     ax.set_xlabel("Time (ns)")
#     ax.set_title(qubit["qubit"])
# grid.fig.suptitle(f"Qubit frequency shift vs time after flux pulse \n {date_time} #{node_id}")

# plt.tight_layout()
# plt.show()
# node.results["figure_freqs_shift"] = grid.fig


# grid = QubitGrid(ds, [q.grid_location for q in qubits])

# for ax, qubit in grid_iter(grid):
#     # flux_response.sel(qubit=qubit["qubit"]).plot(ax=ax)
#     ds.loc[qubit].flux_response.plot(ax=ax)
#     ax.set_ylabel("Flux (V)")
#     ax.set_xlabel("Time (ns)")
#     ax.set_title(qubit["qubit"])
# grid.fig.suptitle(f"Flux response vs time \n {date_time} #{node_id}")

# plt.tight_layout()
# plt.show()
# node.results["figure_flux_response"] = grid.fig

# # %% {Update_state}
# for qubit in tracked_qubits:
#     qubit.revert_changes()

# # %% {Save_results}
# node.results["ds"] = ds
# node.outcomes = {q.name: "successful" for q in qubits}
# node.results["initial_parameters"] = node.parameters.model_dump()
# node.machine = machine
# save_node(node)

# # %% Fitting multi-exponential
# # TODO: Make it to work on multiple qubits


# # Define your model function
# def model_1exp(t, a0, a1, t1):
#     return a0 * (1 + a1 * np.exp(-t / t1))


# def model_2exp(t, a0, a1, a2, t1, t2):
#     return a0 * (1 + a1 * np.exp(-t / t1) + a2 * np.exp(-t / t2))


# t_data = flux_response.time.values
# y_data = flux_response.isel(qubit=0).values

# # Fit the data
# popt, pcov = curve_fit(model_1exp, t_data, y_data, p0=[np.max(y_data), np.min(y_data) / np.max(y_data) - 1, 1000])

# a0_0 = popt[0]
# a1_0 = a2_0 = popt[1] / 2
# t1_0 = popt[-1]
# t2_0 = 100

# initial_guess = [a0_0, a1_0, a2_0, t1_0, t2_0]

# # Perform nonlinear curve fitting
# popt, pcov = curve_fit(model_2exp, t_data, y_data, p0=initial_guess)

# y_fit = model_2exp(t_data, *popt)
# # y_fit = model(t_data, -0.0015, -1 / 200, popt[2], popt[3], popt[4])

# # Plot
# plt.figure()
# plt.scatter(t_data, y_data, label="Data")
# plt.plot(
#     t_data,
#     y_fit,
#     "r-",
#     label=f"Fit: a0={popt[0]:.3f}, a1={popt[1]:.3f}, a2={popt[2]:.3f}, t1={popt[3]:.0f}, t2={popt[4]:.0f}",
# )
# plt.xlabel("Time")
# plt.ylabel("Value")
# plt.legend()
# plt.show()
