# %% {Imports}
from dataclasses import asdict

from typing import Literal, Optional, List
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from qm import SimulationConfig
from qm.qua import *
from qualibrate import QualibrationNode, NodeParameters
from qualibration_libs.parameters import get_qubit_pairs

from quam_config.my_quam import Quam
from qualibration_libs.legacy.macros import active_reset, readout_state, readout_state_gef, active_reset_gef, active_reset_simple
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names
from qualibration_libs.legacy.lib.save_utils import fetch_results_as_xarray, load_dataset
from qualibration_libs.legacy.lib.fit import fit_oscillation, oscillation, fix_oscillation_phi_2pi
from qualibration_libs.legacy.lib.plot_utils import QubitPairGrid, grid_iter, grid_pair_names

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


from qualang_tools.results import progress_counter, fetching_tool
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from calibration_utils.coupler_leakage_phase_optimizatin import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_conditional_phase_data,
    plot_leakage_data
)
# %% {Initialisation}
description = """ auto geenrate by chatgpt, need to check again!!!!!

"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18c_coupler_leakage_phase_optimization",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)

@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    node.parameters.num_averages = 1

# Instantiate the QUAM class from the state file
node.machine = Quam.load()

# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    u = unit(coerce_to_integer=True)

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    n_avg = node.parameters.num_averages  # The number of averages
    num_qubit_pairs = len(node.parameters.qubit_pairs)
    flux_point = node.parameters.flux_point_joint_or_independent_or_pairwise  # 'independent' or 'joint' or 'pairwise'
    # Loop parameters
    fluxes_coupler = (
        np.arange(
            node.parameters.coupler_flux_min,
            node.parameters.coupler_flux_max + 0.0001,
            node.parameters.coupler_flux_step
        )
        + qubit_pairs[0].macros["Cz"].coupler_flux_pulse.amplitude
    )

    fluxes_qubit = np.arange(
        node.parameters.qubit_flux_min,
        node.parameters.qubit_flux_max + 0.0001,
        node.parameters.qubit_flux_step
    )

    fluxes_qp = {}
    for qp in qubit_pairs:
        # estimate the flux shift to get the control qubit to the target qubit frequency
        fluxes_qp[qp.name] = fluxes_qubit + qp.detuning
        pulse_duration = qp.macros["Cz"].coupler_flux_pulse.length - qp.macros["Cz"].coupler_flux_pulse.zero_padding
        assert pulse_duration % 4 == 0, f"Expected pulse_duration to be a multiple of 4, got {pulse_duration}"
    node.namespace["fluxes_qp"] = fluxes_qp

    reset_coupler_bias = False
    frames = np.arange(0, 1, 1 / node.parameters.num_frames)
    node.namespace["sweep_axes"] = {
    "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
    "coupler_flux": xr.DataArray(fluxes_coupler, attrs={"long_name": "coupler flux", "units": "V"}),
    "qubit_flux": xr.DataArray(fluxes_qubit, attrs={"long_name": "qubit flux", "units": "ns"}),
    "frames":xr.DataArray(frames,attrs={"long_name":"frames","unit":"arb."}),
    "control_ax":xr.DataArray([0,1],attrs={"long_name":"control_ax"})
    }


    with program() as node.namespace["qua_program"]:
        n = declare(int)
        flux_coupler = declare(float)
        flux_qubit = declare(float)
        comp_flux_qubit = declare(float)
        n_st = declare_stream()
        qua_pulse_duration = declare(int, value = int(pulse_duration/4))
        frame = declare(fixed)
        control_initial = declare(int)

        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            leakage_control = [declare(fixed) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
            leakage_control_st = [declare_stream() for _ in range(num_qubit_pairs)]

        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()
        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for ii, qp in multiplexed_qubit_pairs.items():
                print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))
                # Bring the active qubits to the minimum frequency point
                node.machine.set_all_fluxes(flux_point, qp)
                wait(1000)

                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(flux_qubit, fluxes_qp[qp.name])):
                            with for_(*from_array(frame, frames)):
                                with for_(*from_array(control_initial, [0, 1])):        
                                    # reset
                                    if node.parameters.use_state_discrimination:
                                        assign(leakage_control[ii], 0)

                                    # Qubit initialization
                                    qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                                    qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                                    align()

                                    if "coupler_qubit_crosstalk" in qp.extras:
                                        assign(comp_flux_qubit, qp.detuning  +  qp.extras["coupler_qubit_crosstalk"] * flux_coupler )
                                    else:
                                        assign(comp_flux_qubit, qp.detuning)
                                    qp.align()
                                    
                                    # setting both qubits ot the initial state
                                    with if_(control_initial == 1, unsafe = True):
                                        qp.qubit_control.xy.play("x180")
                                    qp.qubit_target.xy.play("x90")
                                    qp.align()

                                    qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = qua_pulse_duration)
                                    qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = qua_pulse_duration)

                                    qp.align()
                                    frame_rotation_2pi(frame, qp.qubit_target.xy.name)
                                    qp.qubit_target.xy.play("x90")
                                    qp.align()
                                    # readout
                                    if node.parameters.use_state_discrimination:
                                        qp.qubit_control.readout_state_gef(state_c[ii])
                                        qp.qubit_target.readout_state(state_t[ii])
                                        assign(leakage_control[ii], Cast.to_fixed( state_c[ii] == 2))
                                        save(state_c[ii], state_c_st[ii])
                                        save(state_t[ii], state_t_st[ii])
                                        save(leakage_control[ii], leakage_control_st[ii])

                                    else:
                                        qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                        qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                        save(I_c[ii], I_c_st[ii])
                                        save(Q_c[ii], Q_c_st[ii])
                                        save(I_t[ii], I_t_st[ii])
                                        save(Q_t[ii], Q_t_st[ii])

                align(*([qp.qubit_control.xy.name for qp in qubit_pairs] +
                        [qp.qubit_control.z.name for qp in qubit_pairs] +
                        [qp.qubit_control.resonator.name for qp in qubit_pairs]))

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                    state_t_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                    leakage_control_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"state_control_f{i + 1}")

                else:
                    I_c_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(2).buffer(len(frames)).buffer(len(fluxes_qubit)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

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
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict()}

# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset."""
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
                node.parameters.num_averages,
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
    # Get the active qubits from the loaded node parameters
    node.results['ds_raw'] = node.results.pop("ds")
    
# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted results in the "fit_results" dictionary."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result["success"] else "failed")
        for qubit_pair_name, fit_result in node.results["fit_results"].items()
    }

# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_leakage = plot_leakage_data(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"],node)
    fig_condition_phase = plot_conditional_phase_data(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"],node)
    plt.show()
    node.results["figures"] = {"leakage": fig_leakage,"condition_phase":fig_condition_phase}
    

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    with (node.record_state_updates()):
        for qp in node.namespace["qubit_pairs"]:
            flux_coupler = node.results["fit_results"][qp.name]["coupler_flux_Cz"]
            qp.macros["Cz"].coupler_flux_pulse.amplitude = flux_coupler
            if "coupler_qubit_crosstalk" in qp.extras:
                qp.macros["Cz"].flux_pulse_control.amplitude = node.results["fit_results"][qp.name]["qubit_flux_Cz"] + qp.extras["coupler_qubit_crosstalk"] * flux_coupler
            else:
                qp.macros["Cz"].flux_pulse_control.amplitude = node.results["fit_results"][qp.name]["qubit_flux_Cz"]

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()

# %%
