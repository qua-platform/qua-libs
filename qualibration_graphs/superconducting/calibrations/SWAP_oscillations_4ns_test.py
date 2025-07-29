# %% {Imports}
from qualibration_libs.data import XarrayDataFetcher
from quam_libs.components import QuAM
from quam_libs.macros import active_reset, readout_state, readout_state_gef, active_reset_gef
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from quam_libs.lib.save_utils import fetch_results_as_xarray, load_dataset, save_node
from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit
from qm import SimulationConfig
from qm.qua import *
from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import warnings
from qualang_tools.bakery import baking
from quam_libs.lib.fit import extract_dominant_frequencies
from quam_libs.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from scipy.optimize import curve_fit
from quam_libs.components.gates.two_qubit_gates import CZGate
from quam_libs.lib.pulses import FluxPulse
from scipy.fft import fft
import xarray as xr
from quam_libs.components.gates.two_qubit_gates import SWAP_Coupler_Gate
from quam_libs.lib.fit import oscillation_decay_exp, fit_oscillation_decay_exp

### modifications
from quam_config import Quam
from calibration_utils.swap_oscillations_test import (
    Parameters
)
from qualibration_libs.parameters import get_qubits
from qualibrate import QualibrationNode, NodeParameters
from qualibration_libs.runtime import simulate_and_plot
# %% {Initialisation}
description = """
        SWAP OSCILLATIONS

Prerequisites:

State update:
    
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="SWAP_oscillations_4ns",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    node.parameters.min_wait_time_in_ns  = 16
    node.parameters.max_wait_time_in_ns  = 250
    node.parameters.wait_time_num_points  = 100
    node.use_state_discrimination = True
    node.parameters.qubits = ["q1", "q2"]
    node.parameters.qubit_pairs = ["q1-2"] #TODO: MAKE SURE IT WORKS! and add a function in the untils. Look at what Paul did in cz chevron


# Instantiate the QUAM class from the state file
node.machine = Quam.load(r"C:\Git\qua-libs\qualibration_graphs\superconducting\quam_state")


# %%

####################
# Helper functions #
####################

def rabi_chevron_model(ft, J, f0, a, offset, tau):
    f, t = ft
    J = J
    w = f
    w0 = f0
    g = offset + a * np.sin(2 * np.pi * np.sqrt(J ** 2 + (w - w0) ** 2) * t) ** 2 * np.exp(-tau * np.abs((w - w0)))
    return g.ravel()


def fit_rabi_chevron(ds_qp, init_length, init_detuning):
    da_target = ds_qp.state_target
    exp_data = da_target.values
    detuning = da_target.detuning
    time = da_target.idle_time * 4 * 1e-9
    t, f = np.meshgrid(time, detuning)
    initial_guess = (1e9 / init_length / 2,
                     init_detuning,
                     -1,
                     1.0,
                     100e-9)
    fdata = np.vstack((f.ravel(), t.ravel()))
    tdata = exp_data.ravel()
    popt, pcov = curve_fit(rabi_chevron_model, fdata, tdata, p0=initial_guess)
    J = popt[0]
    f0 = popt[1]
    a = popt[2]
    offset = popt[3]
    tau = popt[4]

    return J, f0, a, offset, tau



# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubits from the node and organize them by batches
    node.namespace["qubits"] = qubits = get_qubits(node)
    node.namespace["qubit_pairs"] = qubit_pairs = node.parameters.qubit_pairs #TODO: CHANGE! Look at what Paul did in cz chevron

    num_qubit_pairs = len(qubit_pairs)
    num_qubits = len(qubits)
    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    # The amplitude sweep for the coupler and qubit flux pulses
    control_amps = np.arange(1 - node.parameters.control_amp_range, 1 + node.parameters.control_amp_range,
                             node.parameters.control_amp_step)
    # The idle time sweep for the coupler and qubit flux pulses
    idle_times = np.arange(node.parameters.min_wait_time_in_ns, node.parameters.max_wait_time_in_ns,
                           node.parameters.wait_time_num_points) // 4

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "amplitude": xr.DataArray(control_amps, attrs={"long_name": "SWAP_amplitude", "units": "V"}),
        "duration": xr.DataArray(control_amps, attrs={"long_name": "SWAP_duration", "units": "ns"}),
    }
    with program() as node.namespace["qua_program"]:
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()

        amp = declare(float)
        idle_time = declare(int)
        n_st = declare_stream()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_st = [declare_stream() for _ in range(num_qubit_pairs)]



        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()
        wait(1000) #TODO:DO I NEED THIS?
        for multiplexed_qubit_pairs in qubit_pairs.batch():
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(amp, control_amps)):
                    with for_(*from_array(idle_time, idle_times)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            # reset
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)

                            align()

                            # setting qubits ot the initial state
                            qp.qubit_control.xy.play("x180")

                            align()
                            qp.qubit_control.z.play("const", amplitude_scale=amp * qp.gates[
                                "SWAP_Coupler"].flux_pulse_control.amplitude / 0.1, duration=idle_time)
                            qp.coupler.play("const",
                                            amplitude_scale=qp.gates["SWAP_Coupler"].coupler_pulse_control.amplitude / 0.1,
                                            duration=idle_time)
                            align()
                            # readout
                            if node.parameters.use_state_discrimination:
                                qp.qubit_control.readout_state(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                save(state_c[ii], state_c_st[ii])
                                save(state_t[ii], state_t_st[ii])

                                assign(state[ii], state_c[ii] * 2 + state_t[ii])
                                save(state[ii], state_st[ii])
                            else:
                                qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_c[ii], I_c_st[ii])
                                save(Q_c[ii], Q_c_st[ii])
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])
                align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(
                        f"state_control{i + 1}")
                    state_t_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(
                        f"state_target{i + 1}")
                    state_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"state{i + 1}")
                else:
                    I_c_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(len(idle_times)).buffer(len(control_amps)).average().save(f"Q_target{i + 1}")

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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    qubit_pairs = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]
    # define the amplitudes for the flux pulses
    pulse_amplitudes = {}
    for qp in qubit_pairs:
        detuning = qp.qubit_control.xy.RF_frequency - qp.qubit_target.xy.RF_frequency - qp.qubit_target.anharmonicity
        pulse_amplitudes[qp.name] = float(np.sqrt(-detuning / qp.qubit_control.freq_vs_flux_01_quad_term))
    node.namespace["pulse_amplitudes"] = pulse_amplitudes
    node.namespace["qubits"] = [qp.qubit_control for qp in qubit_pairs] + [qp.qubit_target for qp in qubit_pairs]
    node.namespace["qubit_pairs"] = [node.machine.qubit_pairs[pair] for pair in node.parameters.qubit_pairs]



# %%
if not node.parameters.simulate:
    ds = ds.assign_coords(idle_time=ds.idle_time * 4)
    ds = ds.assign({"res_sum": ds.state_control - ds.state_target})
    amp_full = np.array([control_amps * qp.gates["SWAP_Coupler"].flux_pulse_control.amplitude for qp in qubit_pairs])
    ds = ds.assign_coords({"amp_full": (["qubit", "amp"], amp_full)})
    detunings = np.array([-(control_amps * qp.gates[
        "SWAP_Coupler"].flux_pulse_control.amplitude) ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term for qp in
                          qubit_pairs])
    ds = ds.assign_coords({"detuning": (["qubit", "amp"], detunings)})
# %%
if not node.parameters.simulate:
    amplitudes = {}
    lengths = {}
    zero_paddings = {}
    fitted_ds = {}
    detunings = {}
    Js = {}
    RC_success = {}

    for qp in qubit_pairs:
        print(qp.name)
        ds_qp = ds.sel(qubit=qp.name)

        amp_guess = ds_qp.state_target.max("idle_time") - ds_qp.state_target.min("idle_time")
        flux_amp_idx = int(amp_guess.argmax())
        flux_amp = float(ds_qp.amp_full[flux_amp_idx])
        fit_data = fit_oscillation_decay_exp(
            ds_qp.state_control.isel(amp=flux_amp_idx), "idle_time")
        flux_time = int(1 / fit_data.sel(fit_vals='f'))

        amplitudes[qp.name] = flux_amp
        detunings[qp.name] = -flux_amp ** 2 * qp.qubit_control.freq_vs_flux_01_quad_term
        lengths[qp.name] = flux_time
        fitted_ds[qp.name] = ds_qp.assign({'fitted': oscillation_decay_exp(ds_qp.idle_time,
                                                                           fit_data.sel(
                                                                               fit_vals="a"),
                                                                           fit_data.sel(
                                                                               fit_vals="f"),
                                                                           fit_data.sel(
                                                                               fit_vals="phi"),
                                                                           fit_data.sel(
                                                                               fit_vals="offset"),
                                                                           fit_data.sel(fit_vals="decay"))})
        try:
            t = ds.idle_time * 1e-9
            f = ds.sel(qubit=qp.name).detuning
            t, f = np.meshgrid(t, f)
            J, f0, a, offset, tau = fit_rabi_chevron(ds_qp, lengths[qp.name], detunings[qp.name])
            data_fitted = rabi_chevron_model((f, t), J, f0, a, offset, tau).reshape(len(ds.amp), len(ds.idle_time))
            Js[qp.name] = J
            detunings[qp.name] = f0
            amplitudes[qp.name] = np.sqrt(-detunings[qp.name] / qp.qubit_control.freq_vs_flux_01_quad_term)
            flux_time = int(1 / (2 * J) * 1e9)
            lengths[qp.name] = flux_time
            RC_success[qp.name] = True
        except:
            print(f"Rabi-Chevron fit for {qp.name} failed")
            RC_success[qp.name] = False
# %% {Plotting}
if not node.parameters.simulate:
    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.to_array().sel(qubit=qubit_pair['qubit']).sel(
            variable='state_control').assign_coords(detuning_MHz=1e-6 * ds.detuning.sel(qubit=qp.name)).plot(ax=ax,
                                                                                                             x='idle_time',
                                                                                                             y='detuning_MHz',
                                                                                                             add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        # ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6 * detunings[qubit_pair['qubit']], color='k', linestyle='--', lw=0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']], color='k', linestyle='--', lw=0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        if RC_success[qubit_pair['qubit']]:
            f_eff = np.sqrt(Js[qubit_pair['qubit']] ** 2 + (
                        ds.detuning.sel(qubit=qubit_pair['qubit']) - detunings[qubit_pair['qubit']]) ** 2)
            for n in range(10):
                ax.plot(n * 0.5 / f_eff * 1e9, 1e-6 * ds.detuning.sel(qubit=qubit_pair['qubit']), color='red', lw=0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qubit=qubit_pair['qubit'])
        amp_full_range = np.sqrt(-detuning_range / qp.qubit_control.freq_vs_flux_01_quad_term)
        ax2.set_ylim(amp_full_range.min(), amp_full_range.max())
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        ax.set_ylim(detuning_range.min() * 1e-6, detuning_range.max() * 1e-6)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()

    plt.suptitle('control qubit state')
    plt.show()
    node.results["figure_control"] = grid.fig

    grid_names, qubit_pair_names = grid_pair_names(qubit_pairs)
    grid = QubitPairGrid(grid_names, qubit_pair_names)
    for ax, qubit_pair in grid_iter(grid):
        plot = ds.to_array().sel(qubit=qubit_pair['qubit']).sel(
            variable='state_target').assign_coords(detuning_MHz=1e-6 * ds.detuning.sel(qubit=qp.name)).plot(ax=ax,
                                                                                                            x='idle_time',
                                                                                                            y='detuning_MHz',
                                                                                                            add_colorbar=False)
        plt.colorbar(plot, ax=ax, orientation='horizontal', pad=0.2, aspect=30, label='Amplitude')
        # ax.plot([lengths[qubit_pair['qubit']]-zero_paddings[qubit_pair['qubit']]],[1e-6*detunings[qubit_pair['qubit']]],marker= '.', color = 'red')
        ax.axhline(y=1e-6 * detunings[qubit_pair['qubit']], color='k', linestyle='--', lw=0.5)
        ax.axvline(x=lengths[qubit_pair['qubit']], color='k', linestyle='--', lw=0.5)
        ax.set_title(qubit_pair["qubit"])
        ax.set_ylabel('Detuning [MHz]')
        ax.set_xlabel('time [nS]')
        if RC_success[qubit_pair['qubit']]:
            f_eff = np.sqrt(Js[qubit_pair['qubit']] ** 2 + (
                        ds.detuning.sel(qubit=qubit_pair['qubit']) - detunings[qubit_pair['qubit']]) ** 2)
            for n in range(10):
                ax.plot(n * 0.5 / f_eff * 1e9, 1e-6 * ds.detuning.sel(qubit=qubit_pair['qubit']), color='red', lw=0.3)

        ax2 = ax.twinx()
        detuning_range = ds.detuning.sel(qubit=qubit_pair['qubit'])
        amp_full_range = np.sqrt(-detuning_range / qp.qubit_control.freq_vs_flux_01_quad_term)
        ax2.set_ylim(amp_full_range.min(), amp_full_range.max())
        ax2.set_ylabel('Flux amplitude [V]')
        ax.set_ylabel('Detuning [MHz]')
        ax.set_ylim(detuning_range.min() * 1e-6, detuning_range.max() * 1e-6)
        ax2.yaxis.set_label_position('right')
        ax2.yaxis.tick_right()
    plt.suptitle('target qubit state')
    plt.show()
    node.results["figure_target"] = grid.fig

# %% {Update_state}
if not node.parameters.simulate:
    with node.record_state_updates():
        for qp in qubit_pairs:
            gate_time_ns = int(lengths[qp.name] / 2)
            gate_time_including_zeros = gate_time_ns - gate_time_ns % 4 + 4
            zero_padding = gate_time_including_zeros - gate_time_ns
            flux_pulse_amp = amplitudes[qp.name]
            qp.gates['SWAP_Coupler'].flux_pulse_control.amplitude = flux_pulse_amp
            qp.gates['SWAP_Coupler'].flux_pulse_control.zero_padding = zero_padding
            qp.gates['SWAP_Coupler'].flux_pulse_control.length = gate_time_including_zeros

# %% {Save_results}
if not node.parameters.simulate:
    node.outcomes = {q.name: "successful" for q in qubit_pairs}
    node.results['initial_parameters'] = node.parameters.model_dump()
    node.machine = machine
    save_node(node)
# %%