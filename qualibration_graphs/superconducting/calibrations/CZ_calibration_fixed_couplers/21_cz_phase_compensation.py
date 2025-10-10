# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_phase_compensation import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs, get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        CZ PHASE COMPENSATION
This program calibrates and compensates the residual single-qubit (local Z) phase shifts induced by a CZ gate.
For each selected qubit pair, the sequence prepares either the control or the target qubit in a Ramsey-like
fragment (x90 – CZ – x90) while sweeping an additional virtual frame rotation (fractional phase in [0,1) → 2π) that
is added on top of the currently stored phase compensation term (phase_shift_control / phase_shift_target).
The resulting oscillations versus frame rotation are sinusoidal; fitting their phase gives the residual phase error.

From the fitted phases, updated compensation values are computed and written back so that future executions of the
CZ macro implicitly cancel these local phases, leaving only the intended conditional phase.

Prerequisites:
    - Having calibrated single-qubit gates and readout.
    - The CZ macro (e.g. "cz_unipolar") must be calibrated and produce a conditional phase near π.
    - (Optional) Prior coarse conditional phase calibration (e.g. separate CZ phase characterization) improves fit robustness.

State update:
    - qp.macros[operation].phase_shift_control
    - qp.macros[operation].phase_shift_target
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="21_cz_phase_compensation",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under quam_experiment/experiments/node_name
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubit_pairs = ["q1-q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    frames = np.arange(0, 1, 1 / node.parameters.num_frames)
    cz_operation = node.parameters.operation

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        frame = declare(fixed)
        n = declare(int)
        n_st = declare_stream()
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]
        extra_phase_c = declare(fixed)
        extra_phase_t = declare(fixed)

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the QPU in terms of flux points (flux tunable transmons and/or tunable couplers)
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()
            # Loop over the number of averages
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Loop over the virtual frame rotations for phase tomography
                with for_(*from_array(frame, frames)):
                    for ii, qp in multiplexed_qubit_pairs.items():
                        # Add the current compensation from the active state
                        assign(extra_phase_c, qp.macros[cz_operation].phase_shift_control)
                        assign(extra_phase_t, qp.macros[cz_operation].phase_shift_target)
                        # Loop over the state preparations for control and target qubits
                        for qubit, state_q, state_st in [
                            (qp.qubit_control, state_c[ii], state_c_st[ii]),
                            (qp.qubit_target, state_t[ii], state_t_st[ii]),
                        ]:
                            # Reset both qubits and align
                            qp.qubit_control.reset(
                                reset_type=node.parameters.reset_type, simulate=node.parameters.simulate
                            )
                            qp.qubit_target.reset(
                                reset_type=node.parameters.reset_type, simulate=node.parameters.simulate
                            )
                            qp.align()
                            # Prepare the qubit in a superposition state with an x90 pulse
                            qubit.xy.play("x90")
                            qubit.align()
                            # Apply the CZ gate with the added frame rotation for phase tomography of the corresponding qubit
                            if qubit is qp.qubit_control:
                                qp.macros[cz_operation].apply(phase_shift_control=frame + extra_phase_c)
                            elif qubit is qp.qubit_target:
                                qp.macros[cz_operation].apply(phase_shift_target=frame + extra_phase_t)
                            # Final x90 pulse to complete the Ramsey sequence
                            qubit.xy.play("x90")
                            qp.align()

                            # Measure the corresponding qubit and save the results in the appropriate stream
                            if node.parameters.use_state_discrimination:
                                # measure both qubits
                                qubit.readout_state(state_q)
                                save(state_q, state_st)
                            else:
                                if qubit is qp.qubit_control:
                                    qp.qubit_control.resonator.measure("readout", qua_vars=(I_c[ii], Q_c[ii]))
                                    save(I_c[ii], I_c_st[ii])
                                    save(Q_c[ii], Q_c_st[ii])
                                elif qubit is qp.qubit_target:
                                    qp.qubit_target.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                    save(I_t[ii], I_t_st[ii])
                                    save(Q_t[ii], Q_t_st[ii])

        with stream_processing():
            n_st.save("n")
            for ii in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[ii].buffer(len(frames)).average().save(f"state_control{ii + 1}")
                    state_t_st[ii].buffer(len(frames)).average().save(f"state_target{ii + 1}")
                else:
                    I_c_st[ii].buffer(len(frames)).average().save(f"I_control{ii + 1}")
                    Q_c_st[ii].buffer(len(frames)).average().save(f"Q_control{ii + 1}")
                    I_t_st[ii].buffer(len(frames)).average().save(f"I_target{ii + 1}")
                    Q_t_st[ii].buffer(len(frames)).average().save(f"Q_target{ii + 1}")


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
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """
    Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset
    called "ds_raw".
    """
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


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    # Load the specified dataset
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    # Get the active qubits from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """
    Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the fitted
    results in the "fit_results" dictionary.
    """
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    # Fit the raw data and extract the relevant fitted parameters
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result.get("success") else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in specific figures whose shape is given by qubit.grid_location."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    # Store the generated figures
    node.results["figures"] = {
        "raw_and_fit": fig_raw_fit,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    cz_operation = node.parameters.operation
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            old_phase_control = qp.macros[cz_operation].phase_shift_control
            old_phase_target = qp.macros[cz_operation].phase_shift_target
            qp.macros[cz_operation].phase_shift_control = (
                old_phase_control - float(node.results["ds_fit"].sel(qubit_pair=qp.name).fitted_control_phase.values)
            ) % 1
            qp.macros[cz_operation].phase_shift_target = (
                old_phase_target - float(node.results["ds_fit"].sel(qubit_pair=qp.name).fitted_target_phase.values)
            ) % 1


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
