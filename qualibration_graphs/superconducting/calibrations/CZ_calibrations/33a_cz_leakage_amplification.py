"""CZ leakage amplification calibration node."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

from calibration_utils.cz_leakage_amp import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    QubitRoles,
    verify_moving_qubit,
    plot_raw_data_with_fit,
    process_raw_dataset,
)

# %% {Initialisation}
description = """
CALIBRATION OF THE CZ GATE COUPLER AMPLITUDE VIA LEAKAGE with error amplification

This sequence calibrates the CZ gate by scanning the coupler pulse amplitude and measuring
the probability that both qubits remain in |1> (P(11)) after repeated CZ gates.
Both qubits start in |1>; the CZ is applied a variable number of times for error amplification.

For each amplitude we measure:
- P(11) = fraction of shots where both qubits are in |1> (``state``)
- Per-qubit GEF readout on the high- and low-frequency qubits (``state_high_q``, ``state_low_q``)

**Error amplification:**
The CZ gate is applied repeatedly (number_of_operations) for each measurement. This amplifies
leakage and coherent errors, making the optimal amplitude easier to identify.

The optimal coupler amplitude is the amplitude at which the mean of P(11) over number_of_operations
is highest (best preservation of |11>).

Prerequisites:
- Tunable-coupler architecture: ``macros[operation]`` must define ``coupler_flux_pulse`` (fixed-coupler pairs are not supported).
- Calibrated single-qubit gates for both qubits in the pair
- Calibrated readout (with state discrimination for |g>,|e>,|f> if used)
- Initial estimate of the CZ coupler amplitude (typically from node 30)

State update:
- The optimal CZ coupler amplitude: qubit_pair.macros[operation].coupler_flux_pulse.amplitude
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="33a_cz_leakage_amplification",  # Name should be unique
    description=description,
    parameters=Parameters(),  # Node parameters: calibration_utils/cz_leakage_amp/parameters.py
    machine=Quam.load(),  # Instantiate the QUAM class from the state file
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set custom parameters for debugging purposes only."""
    # node.parameters.qubit_pairs = ["q1-q2"]
    node.parameters.qubit_pairs = ["coupler_q4_q5"]
    node.parameters.operation = "cz_unipolar"
    node.parameters.use_state_discrimination = True
    node.parameters.reset_type = "active"
    node.parameters.amp_range = 0.05
    node.parameters.amp_step = 0.0005
    node.parameters.num_shots = 100
    node.parameters.number_of_operations = 40
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):  # pylint: disable=too-many-statements
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if not node.parameters.use_state_discrimination:
        raise ValueError("33a_cz_leakage_amplification requires use_state_discrimination=True for P(11) analysis.")

    # Get the active qubit pairs from the node and organize them by batches
    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    operation = node.parameters.operation

    # Verify qp.moving_qubit against recalculation and precompute roles for QUA program loops.
    # Logs a warning and corrects qp.moving_qubit in-memory if they disagree; state is persisted
    # at the end of the node.
    qubit_roles_map = {}
    for qp in qubit_pairs:
        verify_moving_qubit(qp, operation=operation, log_callable=node.log)
        qubit_roles_map[qp.name] = QubitRoles.resolve(qp)
    node.namespace["qubit_roles_map"] = qubit_roles_map

    # Extract the sweep parameters and axes from the node parameters
    n_avg = node.parameters.num_shots
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)

    num_operations = node.parameters.number_of_operations
    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "number_of_operations": xr.DataArray(
            np.arange(1, num_operations + 1),
            attrs={"long_name": "number of operations"},
        ),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
    }

    # The QUA program stored in the node namespace to be transfer to the simulation and execution run_actions
    with program() as node.namespace["qua_program"]:
        amp = declare(fixed)  # amplitude scaling factor for the CZ gate coupler pulse
        n = declare(int)
        n_op = declare(int)  # number of CZ operations
        count = declare(int)  # loop counter
        n_st = declare_output_stream()
        state_high_q = [declare(int) for _ in range(num_qubit_pairs)]
        state_low_q = [declare(int) for _ in range(num_qubit_pairs)]
        state_high_q_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_low_q_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            # Initialize the qubits
            for qp in multiplexed_qubit_pairs.values():
                qubit_role = qubit_roles_map[qp.name]
                high_q, low_q = qubit_role.high, qubit_role.low
                node.machine.initialize_qpu(target=high_q)
                node.machine.initialize_qpu(target=low_q)
            # Loop for averaging
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                # Loop over the number of CZ operations for error amplification
                with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                    # Loop over amplitude scale
                    with for_(*from_array(amp, amplitudes)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qubit_role = qubit_roles_map[qp.name]
                            high_q, low_q = qubit_role.high, qubit_role.low
                            # Reset the qubits
                            high_q.reset(node.parameters.reset_type, node.parameters.simulate)
                            low_q.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()
                            # Reset the frames of both qubits
                            reset_frame(low_q.xy.name)
                            reset_frame(high_q.xy.name)
                            # setting both qubits to the initial state
                            high_q.xy.play("x180")
                            low_q.xy.play("x180")
                            qp.align()
                            # Loop over the number of CZ operations
                            with for_(count, 0, count < n_op, count + 1):
                                # play the CZ gate
                                qp.macros[operation].apply(amplitude_scale_coupler=amp)
                            qp.align()

                            # measure both qubits
                            high_q.readout_state_gef(state_high_q[ii])
                            low_q.readout_state_gef(state_low_q[ii])

                            with if_((state_high_q[ii] == 1) & (state_low_q[ii] == 1)):
                                wait(4)
                                save(1, state_st[ii])
                            with else_():
                                wait(4)
                                save(0, state_st[ii])

                            save(state_high_q[ii], state_high_q_st[ii])
                            save(state_low_q[ii], state_low_q_st[ii])
                        align()
        with stream_processing():
            n_st.save("n")
            for i in range(num_qubit_pairs):
                state_high_q_st[i].buffer(len(amplitudes)).buffer(num_operations).average().save(f"state_high_q{i + 1}")
                state_low_q_st[i].buffer(len(amplitudes)).buffer(num_operations).average().save(f"state_low_q{i + 1}")
                state_st[i].buffer(len(amplitudes)).buffer(num_operations).average().save(f"state{i + 1}")


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
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        # Display the execution report to expose possible runtime errors
        node.log(job.execution_report())
    # Register the raw dataset and role data needed for reproducible re-analysis
    node.results["ds_raw"] = dataset
    qubit_roles_map = node.namespace["qubit_roles_map"]
    node.results["qubit_roles"] = {
        name: {field: getattr(role, field).name for field in role._fields} for name, role in qubit_roles_map.items()
    }


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
    node.namespace["qubit_roles_map"] = {
        name: QubitRoles(**{field: node.machine.qubits[qname] for field, qname in roles.items()})
        for name, roles in node.results["qubit_roles"].items()
    }


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

    figures = plot_raw_data_with_fit(node.results["ds_fit"], qubit_pairs)
    for fig in figures.values():
        plt.show()
    node.results["figures"] = {
        "leakage_raw": figures["raw"],
        "leakage_mean": figures["mean"],
    }


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
            qp.macros[operation].coupler_flux_pulse.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
