# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr
from calibration_utils.JAZZ_ZZ import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_decay_rate_data,
    plot_effective_coupling,
    plot_fit_data,
    plot_raw_data,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Initialisation}
description = """
JAZZ_ZZ CALIBRATION FOR COUPLER DECOUPLE OFFSET

This calibration measures residual ZZ interaction in a qubit pair by sweeping the coupler pulse
amplitude and interaction time in a JAZZ-style echo sequence.

For each qubit pair, the sequence:
1. Resets the control and target qubits
2. Prepares the measured qubit in a superposition
3. Applies two coupler pulses separated by echo pi-pulses on both qubits
4. Applies a virtual phase rotation corresponding to the artificial detuning
5. Measures the selected qubit using either state discrimination or IQ readout

The experiment is repeated for a range of coupler amplitudes and pulse durations.
For each amplitude, the measured oscillation versus interaction time is fitted to a damped cosine model
to extract:
1. The effective interaction strength J_eff
2. The decay rate gamma
3. The decay time constant tau = 1 / gamma

The main calibration target is the coupler amplitude for which the extracted J_eff is closest to the
requested artificial detuning. This amplitude is stored as the optimal_amplitude.
For diagnostics, the analysis also records the amplitude that gives the maximum decay time constant.

Prerequisites:
- Calibrated single-qubit gates for both qubits in the pair

State update:
- qubit_pair.coupler.decouple_offset += optimal_amplitude
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="19_zz_off_jazz",  # Name should be unique
    description=description,  # Describe what the node is doing, which is also reflected in the QUAlibrate GUI
    parameters=Parameters(),  # Node parameters defined under calibration_utils/JAZZ_ZZ/parameters.py
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    node.parameters.qubit_pairs = None
    node.parameters.measure_qubit = "control"
    node.parameters.amp_step = 0.01
    node.parameters.num_shots = 10

    # node.parameters.qubit_pairs = ["qC4-C5"]
    # node.parameters.time_min_in_ns = 16
    # node.parameters.time_max_in_ns = 500
    # node.parameters.time_step_in_ns = 4
    # node.parameters.artificial_detuning_in_mhz = 10
    node.parameters.amp_min = -0.1
    node.parameters.amp_max = 0.1
    # node.parameters.amp_step = 0.001
    # node.parameters.num_shots = 200
    # node.parameters.use_state_discrimination = True
    # node.parameters.reset_type = "active"
    # node.parameters.measure_qubit = "target"
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):  # pylint: disable=too-many-statements
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""

    # Class containing tools to help handle units and conversions.
    u = unit(coerce_to_integer=True)
    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)
    measured_qubits = []
    for qp in qubit_pairs:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    amplitudes = np.arange(node.parameters.amp_min, node.parameters.amp_max, node.parameters.amp_step)
    durations = np.arange(node.parameters.time_min_in_ns, node.parameters.time_max_in_ns, node.parameters.time_step_in_ns) // 4
    detuning = node.parameters.artificial_detuning_in_mhz * u.MHz

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "coupler amplitude", "units": "V"}),
        "time": xr.DataArray(durations * 4, attrs={"long_name": "interaction time", "units": "ns"}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        I_t, I_t_st, Q_t, Q_t_st, n, n_st = node.machine.declare_qua_variables()
        virtual_detuning_phase = declare(fixed)
        amp = declare(fixed)
        t = declare(int)
        if node.parameters.use_state_discrimination:
            state_t = [declare(int) for _ in range(num_qubit_pairs)]
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
                    with for_(*from_array(t, durations)):
                        assign(virtual_detuning_phase, Cast.mul_fixed_by_int(detuning * 1e-9, 4 * t))
                        for ii, qp in multiplexed_qubit_pairs.items():
                            # Reset the qubits
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()
                            # Reset the frames of both qubits
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)
                            qp.align()
                            # setting the protagonist and antagonist qubits
                            if node.parameters.measure_qubit == "control":
                                protagonist_qubit = qp.qubit_control
                                antagonist_qubit = qp.qubit_target
                            else:
                                protagonist_qubit = qp.qubit_target
                                antagonist_qubit = qp.qubit_control
                            qp.align()
                            # play the x90 gate on the protagonist qubit
                            protagonist_qubit.xy.play("x90")
                            qp.coupler.wait(protagonist_qubit.xy.operations["x90"].length//4)
                            antagonist_qubit.wait(protagonist_qubit.xy.operations["x90"].length//4)
                            # play the coupler pulse
                            qp.coupler.play(
                                "const", amplitude_scale=amp / qp.coupler.operations["const"].amplitude, duration=t
                            )
                            protagonist_qubit.xy.wait(t)
                            antagonist_qubit.xy.wait(t)
                            # Echo pulse
                            protagonist_qubit.xy.play("x180")
                            antagonist_qubit.xy.play("x180")
                            qp.coupler.wait(protagonist_qubit.xy.operations["x180"].length//4)
                            # rotate the frame
                            protagonist_qubit.xy.frame_rotation_2pi(virtual_detuning_phase)
                            # play the coupler pulse
                            qp.coupler.play(
                                "const", amplitude_scale=amp / qp.coupler.operations["const"].amplitude, duration=t
                            )
                            protagonist_qubit.xy.wait(t)
                            antagonist_qubit.xy.wait(t)
                            # Tomographic rotation on the protagonist qubit
                            protagonist_qubit.xy.play("x90")
                            qp.align()

                            if node.parameters.use_state_discrimination:
                                # measure both qubits
                                protagonist_qubit.readout_state(state_t[ii])
                                save(state_t[ii], state_t_st[ii])
                            else:
                                protagonist_qubit.resonator.measure("readout", qua_vars=(I_t[ii], Q_t[ii]))
                                save(I_t[ii], I_t_st[ii])
                                save(Q_t[ii], Q_t_st[ii])

            align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_t_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"state_target{i + 1}")
                else:
                    I_t_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"I_target{i + 1}")
                    Q_t_st[i].buffer(len(durations)).buffer(len(amplitudes)).average().save(f"Q_target{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    # Connect to the QOP
    qmm = node.machine.connect()
    # Get the config from the machine
    config = node.machine.generate_config()
    # Simulate the QUA program, generate the waveform report and plot the simulated samples
    _, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
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
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    measured_qubit_names = [q.name for q in node.namespace["measured_qubits"]]
    dataset = dataset.assign_coords(measured_qubit_name=("qubit_pair", measured_qubit_names))
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
    measured_qubits = []
    for qp in node.namespace["qubit_pairs"]:
        if node.parameters.measure_qubit == "control":
            measured_qubits.append(qp.qubit_control)
        else:
            measured_qubits.append(qp.qubit_target)
    node.namespace["measured_qubits"] = measured_qubits
    if "measured_qubit_name" not in node.results["ds_raw"].coords:
        measured_qubit_names = [q.name for q in measured_qubits]
        node.results["ds_raw"] = node.results["ds_raw"].assign_coords(
            measured_qubit_name=("qubit_pair", measured_qubit_names)
        )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset and the fitted results."""
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
    """Plot J_eff, decay time, raw oscillations, and fitted oscillations."""
    qubit_pairs = node.namespace["qubit_pairs"]
    ds_fit = node.results["ds_fit"]

    fig_jeff = plot_effective_coupling(ds_fit, qubit_pairs)
    fig_decay = plot_decay_rate_data(ds_fit, qubit_pairs)
    fig_raw = plot_raw_data(node.results["ds_raw"], qubit_pairs)
    fig_fit = plot_fit_data(ds_fit, qubit_pairs)

    node.results["figures"] = {
        "jeff_vs_amp": fig_jeff,
        "decay_time": fig_decay,
        "raw_data": fig_raw,
        **({"fit_data": fig_fit} if fig_fit is not None else {}),
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit pair data analysis was successful."""

    with node.record_state_updates():
        fit_results = node.results["fit_results"]
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue
            node.machine.qubit_pairs[qp.name].coupler.decouple_offset += fit_results[qp.name]["optimal_amplitude"]
            # node.machine.qubit_pairs[qp.name].coupler.decouple_offset += fit_results[qp.name][
            #     "max_decay_time_amplitude"
            # ]
            # pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results to the node storage."""
    node.save()


# %%
