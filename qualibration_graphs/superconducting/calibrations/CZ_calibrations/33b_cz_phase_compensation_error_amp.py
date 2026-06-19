# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_phase_compensation_error_amp import (
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
from qualibration_libs.parameters import get_qubit_pairs
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
CZ PHASE COMPENSATION WITH ERROR AMPLIFICATION

This node calibrates residual single-qubit (local Z) phase shifts induced by the CZ macro.
Compared to the standard phase-compensation node, it repeats the CZ operation a variable
number of times to amplify phase errors before tomography.

For each selected qubit pair:
1. Prepare either control or target qubit in a Ramsey-like sequence.
2. Apply a train of CZ macros (1..N repetitions).
3. Sweep a final virtual frame rotation and fit the Ramsey oscillation phase per repetition count.
4. Average the signal over repetitions and fit a sinc model to locate the optimal compensation frame.

State update:
    - qp.macros[operation].phase_shift_control
    - qp.macros[operation].phase_shift_target
"""

node = QualibrationNode[Parameters, Quam](
    name="33b_cz_phase_compensation_error_amp",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debug parameter overrides when running directly from IDE."""
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):  # pylint: disable=too-many-statements
    """Create sweep axes and generate the QUA program."""
    unit(coerce_to_integer=True)

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    n_avg = node.parameters.num_shots
    frames = np.linspace(-node.parameters.frame_range / 2, node.parameters.frame_range / 2, node.parameters.num_frames)
    num_operations = node.parameters.number_of_operations
    cz_operation = node.parameters.operation

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "number_of_operations": xr.DataArray(
            np.arange(1, num_operations + 1),
            attrs={"long_name": "number of CZ operations"},
        ),
        "frame": xr.DataArray(frames, attrs={"long_name": "frame rotation", "units": "2π"}),
    }

    with program() as node.namespace["qua_program"]:
        frame = declare(fixed)
        n = declare(int)
        n_op = declare(int)
        count = declare(int)
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
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(n_op, 1, n_op <= num_operations, n_op + 1):
                    with for_(*from_array(frame, frames)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            assign(extra_phase_c, qp.macros[cz_operation].phase_shift_control)
                            assign(extra_phase_t, qp.macros[cz_operation].phase_shift_target)
                            for qubit, state_q, state_st in [
                                (qp.qubit_control, state_c[ii], state_c_st[ii]),
                                (qp.qubit_target, state_t[ii], state_t_st[ii]),
                            ]:
                                qp.qubit_control.reset(
                                    reset_type=node.parameters.reset_type, simulate=node.parameters.simulate
                                )
                                qp.qubit_target.reset(
                                    reset_type=node.parameters.reset_type, simulate=node.parameters.simulate
                                )
                                qp.align()

                                qubit.xy.play("x90")
                                qubit.align()

                                with for_(count, 0, count < n_op, count + 1):
                                    if qubit is qp.qubit_control:
                                        qp.macros[cz_operation].apply(phase_shift_control=extra_phase_c + frame)
                                    elif qubit is qp.qubit_target:
                                        qp.macros[cz_operation].apply(phase_shift_target=extra_phase_t + frame)

                                qubit.xy.play("x90")
                                qp.align()

                                if node.parameters.use_state_discrimination:
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
                                align()

        align()
        with stream_processing():
            n_st.save("n")
            for ii in range(num_qubit_pairs):
                if node.parameters.use_state_discrimination:
                    state_c_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"state_control{ii + 1}")
                    state_t_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"state_target{ii + 1}")
                else:
                    I_c_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"I_control{ii + 1}")
                    Q_c_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"Q_control{ii + 1}")
                    I_t_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"I_target{ii + 1}")
                    Q_t_st[ii].buffer(len(frames)).buffer(num_operations).average().save(f"Q_target{ii + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process raw data, run fits, and set outcomes."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qp_name: ("successful" if fit_result.success else "failed") for qp_name, fit_result in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot phase-vs-operations fit for each qubit pair."""
    fig_raw_fit = plot_raw_data_with_fit(node.results["ds_raw"], node.namespace["qubit_pairs"], node.results["ds_fit"])
    plt.show()
    node.results["figures"] = {"phase_vs_operations": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update local-Z compensation with the fitted phase from the sinc fit."""
    cz_operation = node.parameters.operation
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                continue

            old_phase_control = qp.macros[cz_operation].phase_shift_control
            old_phase_target = qp.macros[cz_operation].phase_shift_target
            fitted_control_phase = float(node.results["ds_fit"].sel(qubit_pair=qp.name).fitted_control_phase.values)
            fitted_target_phase = float(node.results["ds_fit"].sel(qubit_pair=qp.name).fitted_target_phase.values)

            qp.macros[cz_operation].phase_shift_control = (old_phase_control + fitted_control_phase) % 1
            qp.macros[cz_operation].phase_shift_target = (old_phase_target + fitted_target_phase) % 1


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save node results and state updates."""
    node.save()


# %%
