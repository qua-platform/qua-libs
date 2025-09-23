# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.results import progress_counter
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config.my_quam import Quam

from qualibrate import QualibrationNode
from quam_config import Quam
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs

from calibration_utils.coupler_interaction_strength import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_target_data,
    plot_control_data,
    plot_domain_frequency
)


# %% {Initialisation}
description = """ auto geenrate by chatgpt, need to check again!!!!!
        COUPLER INTERACTION-STRENGTH CALIBRATION

        
This routine characterizes the effective ZZ/XX interaction activated by a tunable coupler
when both qubits are flux-biased near their entangling configuration. The goal is to find
the coupler flux amplitude that produces the desired oscillation rate for a fast, high-fidelity
CZ (or iSWAP) gate.

The program performs a 2D sweep over:
    - Coupler flux amplitudes around the decouple operating point.
    - Interaction dwell times, sampled in 4-ns clock cycle increments.

Each sequence prepares the qubits (|11> for CZ, |10> for iSWAP), applies synchronized flux pulses
to the control qubit and coupler, optionally compensates qubit detuning for cross-talk, and then
measures both qubits using state discrimination (or IQ acquisition). The resulting oscillations
in the joint populations reveal the instantaneous interaction frequency.

The acquired data is converted to an xarray dataset with the sweep axes, and a dominant-frequency
analysis is performed to extract the interaction rate as a function of coupler bias. From this,
the calibration identifies:
    - The optimal coupler flux amplitude that achieves the target gate time.
    - The coupler flux offset that minimizes residual interaction.
    - Optional qubit flux compensation derived from coupler cross-talk coefficients.

Detected optima are plotted for quick validation, and the node optionally writes the calibrated
coupler-pulse amplitude back into the machine model (e.g., `macros["Cz"].coupler_flux_pulse.amplitude`).
This calibration should be run:
    - After zero-point alignment of the coupler.
    - Whenever the desired gate speed changes.
    - Following any hardware drifts that affect coupling strength.

Prerequisites include calibrated readout, reliable active reset, and a CZ/iSWAP macro template.
The resulting interaction map is fundamental for tuning entangling gates and benchmarking two-qubit
performance.
"""
# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="18b_coupler_interaction_strength",  # Name should be unique
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
    fluxes_coupler = np.arange(node.parameters.coupler_flux_min, node.parameters.coupler_flux_max, node.parameters.coupler_flux_step)
    idle_times = np.arange(node.parameters.idle_time_min, node.parameters.idle_time_max, node.parameters.idle_time_step) // 4
    
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "flux_coupler": xr.DataArray(fluxes_coupler, attrs={"long_name": "flux coupler", "units": "V"}),
        "idle_time": xr.DataArray(idle_times, attrs={"long_name": "idle time", "units": "ns"}),
    }


    with program() as node.namespace["qua_program"]:
        n = declare(int)
        flux_coupler = declare(float)
        comp_flux_qubit = declare(float)
        idle_time = declare(int)
        n_st = declare_stream()
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        else:
            I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
            I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        for qubit in node.machine.active_qubits:
            node.machine.initialize_qpu(target=qubit)
            align()

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for ii,qp in multiplexed_qubit_pairs.items():
                print("qubit control: %s, qubit target: %s" % (qp.qubit_control.name, qp.qubit_target.name))
                # Bring the active qubits to the minimum frequency point
                node.machine.set_all_fluxes(flux_point, qp)
                wait(1000)
                
                with for_(n, 0, n < n_avg, n + 1):
                    save(n, n_st)
                    with for_(*from_array(flux_coupler, fluxes_coupler)):
                        with for_(*from_array(idle_time, idle_times)):
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
                            qp.qubit_control.xy.play("x180")
                            if node.parameters.cz_or_iswap == "cz":
                                qp.qubit_target.xy.play("x180")
                            qp.qubit_control.z.wait(qp.qubit_control.xy.operations["x180"].length >> 2)
                            qp.coupler.wait(qp.qubit_control.xy.operations["x180"].length >> 2)

                            # Play the flux pulse on the qubit control and coupler
                            qp.qubit_control.z.play("const", amplitude_scale = comp_flux_qubit / qp.qubit_control.z.operations["const"].amplitude, duration = idle_time)
                            qp.coupler.play("const", amplitude_scale = flux_coupler / qp.coupler.operations["const"].amplitude, duration = idle_time)

                            qp.align()
                            # readout
                            if node.parameters.use_state_discrimination:
                                qp.qubit_control.readout_state_gef(state_c[ii])
                                qp.qubit_target.readout_state(state_t[ii])
                                save(state_c[ii], state_c_st[ii])
                                save(state_t[ii], state_t_st[ii])
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
                    state_c_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_control{i + 1}")
                    state_t_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"state_target{i + 1}")
                else:
                    I_c_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(len(idle_times)).buffer(len(fluxes_coupler)).average().save(f"Q_target{i + 1}")

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
    fig_control = plot_control_data(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"],node)
    fig_target = plot_target_data(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"],node)
    fig_domain_frequency = plot_domain_frequency(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["fit_results"],node)
    plt.show()
    node.results["figures"] = {
                                "fig_control": fig_control,
                               "fig_target":fig_target,
                               "fig_domain_frequency":fig_domain_frequency}

# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            coupler_flux_pulse_amp = float(node.results['fit_results'][qp.name]['coupler_flux_pulse'])
            if "coupler_qubit_crosstalk" in qp.extras:
                qubit_flux_pulse_amp = qp.detuning + qp.extras["coupler_qubit_crosstalk"] * coupler_flux_pulse_amp
            else:
                qubit_flux_pulse_amp = qp.detuning

            # qp.macros["Cz"].flux_pulse_control.amplitude = qubit_flux_pulse_amp
            if node.parameters.cz_or_iswap == "cz":
                qp.macros["Cz"].coupler_flux_pulse.amplitude = coupler_flux_pulse_amp

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
