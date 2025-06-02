# %% {Imports}
import warnings
from typing import List, Literal, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.chevron_cz import Parameters, fit_raw_data, plot_raw_data_with_fit, process_raw_dataset
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import NodeParameters, QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from scipy.optimize import curve_fit

# %% {Node_parameters}
description = """
Unipolar CPhase Gate Calibration
This sequence measures the time and detuning required for a unipolar CPhase gate. The process involves:

1. Preparing both qubits in their excited states.
2. Applying a flux pulse with varying amplitude and duration.
3. Measuring the resulting state populations as a function of these parameters.
4. Fitting the results to a Ramsey-Chevron pattern.

From this pattern, we extract:
- The coupling strength (J2) between the qubits.
- The optimal gate parameters (amplitude and duration) for the CPhase gate.

The Ramsey-Chevron pattern emerges due to the interplay between the qubit-qubit coupling and the flux-induced detuning,
allowing us to precisely calibrate the CPhase gate.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the flux pulse amplitude range.

Outcomes:
- Extracted J2 coupling strength.
- Optimal flux pulse amplitude and duration for the CPhase gate.
- Fitted Ramsey-Chevron pattern for visualization and verification.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="13b_chevron_cz_1ns",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = ["qD1-qD2"]
    node.parameters.reset_type = "active"
    node.parameters.num_shots = 100
    node.parameters.max_time_in_ns = 200
    node.parameters.amp_range = 0.1
    node.parameters.amp_step = 0.005
    node.parameters.use_state_discrimination = True
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
# @node.run_action(skip_if=node.parameters.load_data_id is not None)
# def create_qua_program(node: QualibrationNode[Parameters, Quam]):
#     """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

u = unit(coerce_to_integer=True)
qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
# define the amplitudes for the flux pulses
pulse_amplitudes = {}
for qp in qubit_pairs:
    detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
    pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))

node.namespace["qubits"] = qubits = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
num_qubits = len(qubits)
num_qubit_pairs = len(qubit_pairs)
n_avg = node.parameters.num_shots  # The number of averages

# Loop parameters
amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
times_cycles = np.arange(1, node.parameters.max_time_in_ns)

node.namespace["sweep_axes"] = {
    "qubit_pair": xr.DataArray([pair.id for pair in qubit_pairs], attrs={"long_name": "qubit pairs"}),
    "amplitude": xr.DataArray(amplitudes, attrs={"long_name": "amplitudes of the flux pulse"}),
    "time": xr.DataArray(times_cycles, attrs={"long_name": "pulse duration", "units": "ns"}),
}

baked_config = node.machine.generate_config()


def baked_waveform(qubit):
    pulse_segments = []  # Stores the baking objects
    # Create the different baked sequences, each one corresponding to a different truncated duration
    waveform = [0.5] * 16

    for i in range(1, 17):  # from first item up to pulse_duration (16)
        with baking(baked_config, padding_method="right") as b:
            wf = waveform[:i]
            b.add_op(f"flux_pulse{i}", qubit.z.name, wf)
            b.play(f"flux_pulse{i}", qubit.z.name)
        # Append the baking object in the list to call it from the QUA program
        pulse_segments.append(b)

    return pulse_segments


baked_signals = {qubits.qubit_control.name: baked_waveform(qubits.qubit_control) for qubits in qubit_pairs}

node.namespace["baked_config"] = baked_config

with program() as node.namespace["qua_program"]:
    t = declare(int)  # QUA variable for the flux pulse segment indexz
    amp = declare(fixed)
    n = declare(int)
    n_st = declare_stream()
    t_left_ns = declare(int)  # QUA variable for the flux pulse segment index
    t_cycles = declare(int)  # QUA variable for the flux pulse segment index
    I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
    if node.parameters.use_state_discrimination:
        state = [declare(int) for _ in range(num_qubits)]
        state_st = [declare_stream() for _ in range(num_qubits)]

    for qubit in node.machine.active_qubits:
        node.machine.initialize_qpu(target=qubit)
        align()

    for ii, qp in enumerate(qubit_pairs):
        with for_(n, 0, n < n_avg, n + 1):
            save(n, n_st)
            with for_(*from_array(amp, amplitudes)):
                # rest of the pulse
                with for_(*from_array(t, times_cycles)):
                    # Qubit initialization
                    qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                    qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                    align()
                    # set both qubits to the excited state
                    qp.qubit_control.xy.play("x180")
                    qp.qubit_target.xy.play("x180")

                    align()

                    # play the flux pulse
                    with if_(t <= 16):
                        with switch_(t):
                            for j in range(1, 17):
                                with case_(j):
                                    baked_signals[qp.qubit_control.name][j - 1].run(amp_array=[(qp.qubit_control.z.name, pulse_amplitudes[qp.name] / 0.5 * amp)])

                    with else_():
                        assign(t_cycles, t >> 2)  # Right shift by 2 is a quick way to divide by 4
                        assign(t_left_ns, t - (t_cycles << 2))  # left shift by 2 is a quick way to multiply by 4
                        with switch_(t_left_ns):
                            with case_(0):
                                qp.qubit_control.z.play(
                                    "const",
                                    duration=t_cycles,
                                    amplitude_scale=pulse_amplitudes[qp.name]
                                    / qp.qubit_control.z.operations["const"].amplitude
                                    * amp,
                                )
                            for j in range(1, 4):
                                with case_(j):
                                    qp.qubit_control.z.play(
                                        "const",
                                        duration=t_cycles,
                                        amplitude_scale=pulse_amplitudes[qp.name]
                                        / qp.qubit_control.z.operations["const"].amplitude
                                        * amp,
                                    )
                                    baked_signals[qp.qubit_control.name][j - 1].run(amp_array=[(qp.qubit_control.z.name, pulse_amplitudes[qp.name] / 0.5 * amp)])
                    align()

                    if node.parameters.use_state_discrimination:
                        qp.qubit_control.readout_state(state[ii])
                        qp.qubit_target.readout_state(state[ii + 1])
                        save(state[ii], state_st[ii])
                        save(state[ii + 1], state_st[ii + 1])
                    else:
                        qp.qubit_control.resonator.measure("readout", qua_vars=(I[ii], Q[ii]))
                        qp.qubit_target.resonator.measure("readout", qua_vars=(I[ii + 1], Q[ii + 1]))
                        save(I[ii], I_st[ii])
                        save(Q[ii], Q_st[ii])
                        save(I[ii + 1], I_st[ii + 1])
                        save(Q[ii + 1], Q_st[ii + 1])

    with stream_processing():
        n_st.save("n")
        for i in range(num_qubit_pairs):
            if node.parameters.use_state_discrimination:
                state_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_control{i}")
                state_st[i + 1].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"state_target{i+1}")
            else:
                I_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_control{i}")
                Q_st[i].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_control{i}")
                I_st[i + 1].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"I_target{i+1}")
                Q_st[i + 1].buffer(len(times_cycles)).buffer(len(amplitudes)).average().save(f"Q_target{i+1}")


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
    config = node.namespace["baked_config"]
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


ds = node.results["ds_raw"]

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

ds.state_target.plot(ax=ax)

node.results["figure_raw"] = fig

node.save()

# %%

# # %% {Data_fetching_and_dataset_creation}
# if not node.parameters.simulate:
#     if node.parameters.load_data_id is None:
#         # Fetch the data from the OPX and convert it into a xarray with corresponding axes (from most inner to outer loop)
#         ds = fetch_results_as_xarray(job.result_handles, qubit_pairs, {"time": 4 * times_cycles, "amp": amplitudes})
#     else:
#         ds, loaded_machine = load_dataset(node.parameters.load_data_id)
#         if loaded_machine is not None:
#             machine = loaded_machine

#     node.results = {"ds": ds}

# # %% {Data_analysis}
# if not node.parameters.simulate:

#     def abs_amp(qp, amp):
#         return amp * pulse_amplitudes[qp.name]

#     def detuning(qp, amp):
#         return -((amp * pulse_amplitudes[qp.name]) ** 2) * qp.qubit_control.freq_vs_flux_01_quad_term

#     ds = ds.assign_coords({"amp_full": (["qubit", "amp"], np.array([abs_amp(qp, ds.amp.data) for qp in qubit_pairs]))})
#     ds = ds.assign_coords({"detuning": (["qubit", "amp"], np.array([detuning(qp, ds.amp) for qp in qubit_pairs]))})

# # %%
# if not node.parameters.simulate:
#     amplitudes = {}
#     lengths = {}
#     zero_paddings = {}
#     fitted_ds = {}
#     detunings = {}
#     Js = {}

#     for qp in qubit_pairs:
#         print(qp.name)
#         ds_qp = ds.sel(qubit=qp.name)

#         amp_guess = ds_qp.state_target.max("time") - ds_qp.state_target.min("time")
#         flux_amp_idx = int(amp_guess.argmax())
#         flux_amp = float(ds_qp.amp_full[flux_amp_idx])
#         fit_data = fit_oscillation_decay_exp(ds_qp.state_target.isel(amp=flux_amp_idx), "time")
#         flux_time = int(1 / fit_data.sel(fit_vals="f"))

#         print(f"parameters for {qp.name}: amp={flux_amp}, time={flux_time}")
#         amplitudes[qp.name] = flux_amp
#         detunings[qp.name] = -(flux_amp**2) * qp.qubit_control.freq_vs_flux_01_quad_term
#         lengths[qp.name] = flux_time - flux_time % 4 + 4
#         zero_paddings[qp.name] = lengths[qp.name] - flux_time
#         fitted_ds[qp.name] = ds_qp.assign(
#             {
#                 "fitted": oscillation_decay_exp(
#                     ds_qp.time,
#                     fit_data.sel(fit_vals="a"),
#                     fit_data.sel(fit_vals="f"),
#                     fit_data.sel(fit_vals="phi"),
#                     fit_data.sel(fit_vals="offset"),
#                     fit_data.sel(fit_vals="decay"),
#                 )
#             }
#         )
#         if True:
#             t = ds.time * 1e-9
#             f = ds.sel(qubit=qp.name).detuning
#             t, f = np.meshgrid(t, f)
#             J, f0, a, offset, tau = fit_rabi_chevron(ds_qp, lengths[qp.name], detunings[qp.name])
#             data_fitted = rabi_chevron_model((f, t), J, f0, a, offset, tau).reshape(len(ds.amp), len(ds.time))
#             Js[qp.name] = J
#             detunings[qp.name] = f0
#             amplitudes[qp.name] = np.sqrt(-detunings[qp.name] / qp.qubit_control.freq_vs_flux_01_quad_term)
#             flux_time = int(1 / (2 * J) * 1e9)
#             lengths[qp.name] = flux_time - flux_time % 4 + 4
#             zero_paddings[qp.name] = lengths[qp.name] - flux_time
# # %%
# if not node.parameters.simulate:
#     grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
#     grid = QubitPairGrid(grid_names, qubit_pair_names)
#     for ax, qubit_pair in grid_iter(grid):
#         plot = (
#             ds.assign_coords(detuning_MHz=1e-6 * ds.detuning)
#             .state_control.sel(qubit=qubit_pair["qubit"])
#             .plot(ax=ax, x="time", y="detuning_MHz", add_colorbar=False)
#         )
#         plt.colorbar(plot, ax=ax, orientation="horizontal", pad=0.2, aspect=30, label="Amplitude")
#         ax.plot(
#             [lengths[qubit_pair["qubit"]] - zero_paddings[qubit_pair["qubit"]]],
#             [1e-6 * detunings[qubit_pair["qubit"]]],
#             marker=".",
#             color="red",
#         )
#         ax.axhline(y=1e-6 * detunings[qubit_pair["qubit"]], color="k", linestyle="--", lw=0.5)
#         ax.axvline(
#             x=lengths[qubit_pair["qubit"]] - zero_paddings[qubit_pair["qubit"]], color="k", linestyle="--", lw=0.5
#         )
#         ax.set_title(qubit_pair["qubit"])
#         ax.set_ylabel("Detuning [MHz]")
#         ax.set_xlabel("time [nS]")
#         f_eff = np.sqrt(
#             4 * Js[qubit_pair["qubit"]] ** 2
#             + (ds.detuning.sel(qubit=qubit_pair["qubit"]) - detunings[qubit_pair["qubit"]]) ** 2
#         )
#         for n in range(10):
#             ax.plot(n / f_eff * 1e9, 1e-6 * ds.detuning.sel(qubit=qubit_pair["qubit"]), color="red", lw=0.3)

#         quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term
#         print(f"qubit_pair: {qubit_pair['qubit']}, quad: {quad}")

#         def detuning_to_flux(det, quad=quad):
#             return 1e3 * np.sqrt(-1e6 * det / quad)

#         def flux_to_detuning(flux, quad=quad):
#             return -1e-6 * (flux / 1e3) ** 2 * quad

#         ax2 = ax.secondary_yaxis("right", functions=(detuning_to_flux, flux_to_detuning))
#         ax2.set_ylabel("Flux amplitude [V]")
#         ax.set_ylabel("Detuning [MHz]")

#     plt.suptitle("control qubit state")
#     plt.show()
#     node.results["figure_control"] = grid.fig

#     grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
#     grid = QubitPairGrid(grid_names, qubit_pair_names)
#     for ax, qubit_pair in grid_iter(grid):
#         plot = (
#             ds.assign_coords(detuning_MHz=1e-6 * ds.detuning)
#             .state_target.sel(qubit=qubit_pair["qubit"])
#             .plot(ax=ax, x="time", y="detuning_MHz", add_colorbar=False)
#         )
#         plt.colorbar(plot, ax=ax, orientation="horizontal", pad=0.2, aspect=30, label="Amplitude")
#         ax.plot(
#             [lengths[qubit_pair["qubit"]] - zero_paddings[qubit_pair["qubit"]]],
#             [1e-6 * detunings[qubit_pair["qubit"]]],
#             marker=".",
#             color="red",
#         )
#         ax.axhline(y=1e-6 * detunings[qubit_pair["qubit"]], color="k", linestyle="--", lw=0.5)
#         ax.axvline(
#             x=lengths[qubit_pair["qubit"]] - zero_paddings[qubit_pair["qubit"]], color="k", linestyle="--", lw=0.5
#         )
#         ax.set_title(qubit_pair["qubit"])
#         ax.set_ylabel("Detuning [MHz]")
#         ax.set_xlabel("time [nS]")
#         f_eff = np.sqrt(
#             4 * Js[qubit_pair["qubit"]] ** 2
#             + (ds.detuning.sel(qubit=qubit_pair["qubit"]) - detunings[qubit_pair["qubit"]]) ** 2
#         )
#         for n in range(10):
#             ax.plot(n / f_eff * 1e9, 1e-6 * ds.detuning.sel(qubit=qubit_pair["qubit"]), color="red", lw=0.3)
#         quad = machine.qubit_pairs[qubit_pair["qubit"]].qubit_control.freq_vs_flux_01_quad_term

#         def detuning_to_flux(det, quad=quad):
#             return 1e3 * np.sqrt(-1e6 * det / quad)

#         def flux_to_detuning(flux, quad=quad):
#             return -1e-6 * (flux / 1e3) ** 2 * quad

#         ax2 = ax.secondary_yaxis("right", functions=(detuning_to_flux, flux_to_detuning))
#         ax2.set_ylabel("Flux amplitude [V]")
#         ax.set_ylabel("Detuning [MHz]")

#     plt.suptitle("target qubit state")
#     plt.show()
#     node.results["figure_target"] = grid.fig

# # %%

# # %% {Update_state}
# if not node.parameters.simulate:
#     if node.parameters.load_data_id is None:
#         with node.record_state_updates():
#             for qp in qubit_pairs:
#                 qp.gates["Cz_unipolar"] = CZGate(
#                     flux_pulse_control=FluxPulse(
#                         length=lengths[qp.name],
#                         amplitude=amplitudes[qp.name],
#                         zero_padding=zero_paddings[qp.name],
#                         id="flux_pulse_control_" + qp.qubit_target.name,
#                     )
#                 )
#                 qp.gates["Cz"] = f"#./Cz_unipolar"

#                 qp.J2 = Js[qp.name]
#                 qp.detuning = detunings[qp.name]

# # %% {Save_results}
# if not node.parameters.simulate:
#     node.outcomes = {qp.name: "successful" for qp in qubit_pairs}
#     node.results["initial_parameters"] = node.parameters.model_dump()
#     node.machine = machine
#     save_node(node)


# %%
# def rabi_chevron_model(ft, J, f0, a, offset, tau):
#     f, t = ft
#     J = J
#     w = f
#     w0 = f0
#     g = offset + a * np.sin(2 * np.pi * np.sqrt(4 * J**2 + (w - w0) ** 2) * t) ** 2 * np.exp(-tau * np.abs((w - w0)))
#     return g.ravel()


# def fit_rabi_chevron(ds_qp, init_length, init_detuning):
#     da_target = ds_qp.state_target
#     exp_data = da_target.values
#     detuning = da_target.detuning
#     time = da_target.time * 1e-9
#     t, f = np.meshgrid(time, detuning)
#     initial_guess = (1e9 / init_length / 2, init_detuning, -1, 1.0, 100e-9)
#     fdata = np.vstack((f.ravel(), t.ravel()))
#     tdata = exp_data.ravel()
#     popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
#     J = popt[0]
#     f0 = popt[1]
#     a = popt[2]
#     offset = popt[3]
#     tau = popt[4]

#     return J, f0, a, offset, tau

#     return J, f0, a, offset, tau

#     return J, f0, a, offset, tau
#     return J, f0, a, offset, tau
#     return J, f0, a, offset, tau
