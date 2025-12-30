# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
from quam.core import operation
import xarray as xr
from calibration_utils.cz_conditional_phase import (
    FitResults,
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm import SimulationConfig
from qm.qua import *
from qualang_tools.bakery import baking
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
CALIBRATION OF THE CONTROLLED-PHASE (CPHASE) OF THE CZ GATE

This sequence calibrates the CPhase of the CZ gate by scanning the pulse amplitude and measuring the
resulting phase of the target qubit. The calibration compares two scenarios:

1. Control qubit in the ground state
2. Control qubit in the excited state

For each amplitude, we measure:
1. The phase difference of the target qubit between the two scenarios
2. The amount of leakage to the |f> state when the control qubit is in the excited state

The calibration process involves:
1. Applying a CZ gate with varying amplitudes
2. Measuring the phase of the target qubit for both control qubit states
3. Calculating the phase difference
4. Measuring the population in the |f> state to quantify leakage

The optimal CZ gate amplitude is determined by finding the point where:
1. The phase difference is closest to π (0.5 in normalized units)
2. The leakage to the |f> state is minimized

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout for both qubits
- Initial estimate of the CZ gate amplitude

State update:
- The optimal CZ gate amplitude: qubit_pair.gates["Cz"].flux_pulse_control.amplitude
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="20_cz_conditional_phase",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
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
    n_avg = node.parameters.num_averages
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    frames = np.arange(0, 1, 1 / node.parameters.num_frame_rotations)

    # Select the CZ operation type
    operation = node.parameters.operation

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
        "control_axis": xr.DataArray([0, 1], attrs={"long_name": "control qubit state"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I_c, I_c_st, Q_c, Q_c_st, n, n_st = node.machine.declare_qua_variables()
        I_t, I_t_st, Q_t, Q_t_st, _, _ = node.machine.declare_qua_variables()
        amp = declare(fixed)
        frame = declare(fixed)
        control_initial = declare(int)
        if node.parameters.use_state_discrimination:
            state_c = [declare(int) for _ in range(num_qubit_pairs)]
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
            state_c_st = [declare_stream() for _ in range(num_qubit_pairs)]
            state_t_st = [declare_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the qubits
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            # Loop for averaging
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Loop over the amplitude scale
                with for_(*from_array(amp, amplitudes)):
                    # Loop over the frame rotations
                    with for_(*from_array(frame, frames)):
                        # Loop over the initial state of the control qubit
                        with for_(*from_array(control_initial, [0, 1])):
                            for ii, qp in multiplexed_qubit_pairs.items():
                                # Reset the qubits
                                qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                                qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                                qp.align()
                                # Reset the frames of both qubits
                                reset_frame(qp.qubit_target.xy.name)
                                reset_frame(qp.qubit_control.xy.name)
                                # setting both qubits to the initial state
                                qp.qubit_control.xy.play("x180", condition=control_initial == 1)
                                qp.qubit_target.xy.play("x90")
                                qp.align()
                                # play the CZ gate
                                qp.macros[operation].apply(amplitude_scale_control=amp)
                                # rotate the frame
                                qp.qubit_target.xy.frame_rotation_2pi(frame)
                                # Tomographic rotation on the target qubit
                                qp.qubit_target.xy.play("x90")
                                qp.align()

                                if node.parameters.use_state_discrimination:
                                    # measure both qubits
                                    qp.qubit_control.readout_state(state_c[ii])
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
                    state_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(
                        f"state_control{i + 1}"
                    )
                    state_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(
                        f"state_target{i + 1}"
                    )
                else:
                    I_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"I_control{i + 1}")
                    Q_c_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"Q_control{i + 1}")
                    I_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).average().save(f"Q_target{i + 1}")


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
    # Get the active qubit pairs from the loaded node parameters
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analysis the raw data and store the fitted data in another xarray dataset and the fitted results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)

    # Store fit results in the format expected by the rest of the node
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    # Log the relevant information extracted from the data analysis
    log_fitted_results(fit_results, log_callable=node.log)

    # Set outcomes based on fit success
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result.success else "failed")
        for qubit_pair_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    qubit_pairs = node.namespace["qubit_pairs"]

    # Plot phase calibration data
    fig_phase = plot_raw_data_with_fit(
        node.results["ds_fit"],
        qubit_pairs,
    )
    plt.show()

    node.results["phase_figure"] = fig_phase


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit pair data analysis was successful."""

    operation = node.parameters.operation
    with node.record_state_updates():
        fit_results = node.results["fit_results"]
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            qp.macros[operation].flux_pulse_control.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()


# %%
