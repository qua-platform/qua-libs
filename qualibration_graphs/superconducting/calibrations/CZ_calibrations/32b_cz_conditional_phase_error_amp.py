# %% {Imports}
from dataclasses import asdict
from multiprocessing.process import parent_process

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_conditional_phase_error_amp import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    get_moving_qubit,
    get_stationary_qubit,
    verify_moving_qubit,
    plot_moving_qubit_populations,
    plot_raw_data_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.core import tracked_updates
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
CALIBRATION OF THE CONTROLLED-PHASE (CPHASE) OF THE CZ GATE with error amplification

Calibrates the CPhase of the CZ gate by scanning the flux-pulse amplitude on the moving qubit
and measuring the conditional phase acquired by the other qubit. The CZ gate is applied
repeatedly (error amplification) to improve sensitivity to small phase errors.

The calibration compares two scenarios:
1. Moving qubit in the ground state
2. Moving qubit in the excited state

For each amplitude and repetition count, we measure:
1. The phase difference of the other qubit between the two scenarios
2. The |g⟩, |e⟩, and |f⟩ populations of the moving qubit (leakage monitor)

The optimal CZ gate amplitude is the point where:
1. The phase difference (after error amplification) is closest to π (0.5 in normalised units)
2. The leakage to the |f⟩ state is minimised

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair.
- Calibrated readout for both qubits.
- Initial estimate of the CZ gate amplitude (from node 32a or manual entry).

State update:
- ``qubit_pair.macros[operation].flux_pulse_qubit.amplitude`` → optimal CZ amplitude.
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="32b_cz_conditional_phase_error_amp",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/cz_conditional_phase/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubit_pairs = ["q1-q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Verify that qp.moving_qubit in state matches the recalculation from qubit frequencies and
    # anharmonicity. Logs a warning and corrects it in-memory if they disagree; state is persisted
    # at the end of the node.
    for qp in qubit_pairs:
        verify_moving_qubit(qp, log_callable=node.log)

    # Precompute moving/stationary qubit objects once per pair so QUA program loops use dict lookups.
    moving_qubit_map = {qp.name: get_moving_qubit(qp) for qp in qubit_pairs}
    stationary_qubit_map = {qp.name: get_stationary_qubit(qp) for qp in qubit_pairs}

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_averages
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    frames = np.arange(0, 1, 1 / node.parameters.num_frame_rotations)

    operation = node.parameters.operation
    num_operations = node.parameters.number_of_operations
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "number_of_operations": xr.DataArray(
            np.arange(1, num_operations + 1),
            attrs={"long_name": "number of operations"},
        ),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
        "control_axis": xr.DataArray([0, 1], attrs={"long_name": "control qubit state"}),
    }

    tracked_qp_list = []
    for qp in qubit_pairs:
        with tracked_updates(qp, auto_revert=False, dont_assign_to_none=False) as tracked_qp:
            # Mark the CZ gate amplitude as tracked for updates if the calibration is successful
            tracked_qp.macros[operation].phase_shift_control = 0.0
            tracked_qp.macros[operation].phase_shift_target = 0.0
            tracked_qp_list.append(tracked_qp)

    node.namespace["tracked_qubit_pairs"] = tracked_qp_list
    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        amp = declare(fixed)  # amplitude scaling factor for the CZ gate
        frame = declare(fixed)  # frame rotation of the other qubit (even number of operations)
        frame_odd = declare(fixed)  # frame rotation of the other qubit (odd number of operations)
        moving_initial = declare(int)  # initial state of the moving qubit
        n_op = declare(int)  # number of CZ operations
        count = declare(int)  # loop counter
        I_m, I_m_st, Q_m, Q_m_st, n, n_st = node.machine.declare_qua_variables()
        I_s, I_s_st, Q_s, Q_s_st, _, _ = node.machine.declare_qua_variables()
        if node.parameters.use_state_discrimination:
            state_mq = [declare(int) for _ in range(num_qubit_pairs)]
            state_sq = [declare(int) for _ in range(num_qubit_pairs)]
            state_mg_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_me_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_mf_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
            state_sq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the qubits
            for qp in multiplexed_qubit_pairs.values():
                mq = moving_qubit_map[qp.name]
                sq = stationary_qubit_map[qp.name]
                node.machine.initialize_qpu(target=mq)
                node.machine.initialize_qpu(target=sq)
            # Loop for averaging
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Loop over the number of CZ operations for error amplification
                with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                    # Loop over amplitude scale
                    with for_(*from_array(amp, amplitudes)):
                        # Loop over the frame rotations
                        with for_(*from_array(frame, frames)):
                            # Loop over the initial state of the control qubit
                            with for_(*from_array(moving_initial, [0, 1])):
                                for ii, qp in multiplexed_qubit_pairs.items():
                                    mq = moving_qubit_map[qp.name]
                                    sq = stationary_qubit_map[qp.name]
                                    # Reset the qubits
                                    mq.reset(node.parameters.reset_type, node.parameters.simulate)
                                    sq.reset(node.parameters.reset_type, node.parameters.simulate)
                                    qp.align()
                                    # Reset the frames of both qubits
                                    reset_frame(sq.xy.name)
                                    reset_frame(mq.xy.name)
                                    # setting both qubits to the initial state
                                    mq.xy.play("x180", condition=moving_initial == 1)
                                    sq.xy.play("x90")
                                    qp.align()
                                    # Loop over the number of CZ operations
                                    with for_(count, 0, count < n_op, count + 1):
                                        # play the CZ gate
                                        qp.macros[operation].apply(amplitude_scale_qubit=amp)
                                    # rotate the frame by 𝜋/2 for odd number of operations
                                    with if_(((n_op & 1) == 0) & (moving_initial == 1)):
                                        assign(frame_odd, frame - 0.5)
                                        sq.xy.frame_rotation_2pi(frame_odd)
                                    with else_():
                                        sq.xy.frame_rotation_2pi(frame)
                                    # return the other qubit before measurement
                                    sq.xy.play("x90")
                                    qp.align()

                                    if node.parameters.use_state_discrimination:
                                        # measure both qubits
                                        mq.readout_state_gef(state_mq[ii])
                                        sq.readout_state(state_sq[ii])
                                        # save each state outcome to its corresponding stream
                                        with switch_(state_mq[ii]):
                                            with case_(0):
                                                wait(4)
                                                save(1, state_mg_st[ii])
                                                save(0, state_me_st[ii])
                                                save(0, state_mf_st[ii])
                                            with case_(1):
                                                wait(4)
                                                save(0, state_mg_st[ii])
                                                save(1, state_me_st[ii])
                                                save(0, state_mf_st[ii])
                                            with default_():
                                                wait(4)
                                                save(0, state_mg_st[ii])
                                                save(0, state_me_st[ii])
                                                save(1, state_mf_st[ii])
                                        save(state_sq[ii], state_sq_st[ii])
                                    else:
                                        mq.resonator.measure("readout", qua_vars=(I_m[ii], Q_m[ii]))
                                        sq.resonator.measure("readout", qua_vars=(I_s[ii], Q_s[ii]))
                                        save(I_m[ii], I_m_st[ii])
                                        save(Q_m[ii], Q_m_st[ii])
                                        save(I_s[ii], I_s_st[ii])
                                        save(Q_s[ii], Q_s_st[ii])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_mg_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"g_state_moving{i + 1}")
                    state_me_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"e_state_moving{i + 1}")
                    state_mf_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"f_state_moving{i + 1}")
                    state_sq_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"state_stationary{i + 1}")
                else:
                    I_m_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"I_moving{i + 1}")
                    Q_m_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"Q_moving{i + 1}")
                    I_s_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"I_stationary{i + 1}")
                    Q_s_st[i].buffer(2).buffer(len(frames)).buffer(len(amplitudes)).buffer(
                        num_operations
                    ).average().save(f"Q_stationary{i + 1}")


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
                data_fetcher.get("n", 0),
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
    """Analyse raw data, fit, log results, set outcomes and store structured fit results."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qubit_pair_name: ("successful" if fit_result.success else "failed")
        for qubit_pair_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    qubit_pairs = node.namespace["qubit_pairs"]

    ds_fit = node.results["ds_fit"]

    fig_phase = plot_raw_data_with_fit(ds_fit, qubit_pairs)
    plt.show()

    node.results["phase_figure"] = fig_phase

    if "g_state_moving" in ds_fit.data_vars:
        fig_populations = plot_moving_qubit_populations(ds_fit, qubit_pairs)
        plt.show()
        node.results["populations_figure"] = fig_populations
    else:
        node.log("No populations data found in the fit dataset.")


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit pair data analysis was successful."""

    operation = node.parameters.operation
    with node.record_state_updates():
        fit_results = node.results["fit_results"]
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit flagged unsuccessful.")
                continue
            qp.macros[operation].flux_pulse_qubit.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):

    for qp in node.namespace.get("tracked_qubit_pairs", []):
        qp.revert_changes()

    node.save()
