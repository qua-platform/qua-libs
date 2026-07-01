# pylint: disable=R0801
# pylint: disable=duplicate-code

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_jazz_n import (
    Parameters,
    QubitRoles,
    coerce_to_4k_plus_1,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
    process_raw_dataset,
    verify_moving_qubit,
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
        JAZZ-N CZ AMPLITUDE CALIBRATION
This node calibrates the CZ-pulse amplitude using the JAZZ-N protocol
(arXiv:2402.18926v3, Appendix I.1, Fig. 13(a)). The pulse sequence is

    x90(stationary) -- CZ -- [X_pi(moving) & X_pi(stationary) -- CZ] x N -- x90(stationary) -- measure(stationary)

where N = 4k + 1 (k = 0, 1, 2, ...). With the X_pi refocusing pulses on both
qubits, the stationary |1> population evolves as

    P_|1>(stationary) = (1 - cos((2k+1) * theta_CZ)) / 2,

independently of any virtual-Z (single-qubit) phase shifts inside the CZ
macro. The optimal CZ amplitude is the value where theta_CZ = pi, i.e. where
P_|1> is maximal. As N grows the peak around theta_CZ = pi becomes sharper,
giving finer amplitude resolution.

The Z-pulse is supplied by the full CZGate macro selected via the
``operation`` parameter, so any qubit/coupler flux pulse shape that the
macro provides (cz_unipolar, cz_flattop, cz_bipolar, cz_flattop_erf, cz_SNZ)
can be calibrated.

JAZZ-N is a precision fine-tuning upgrade over 32b. The X_pi refocusing
pulses echo out ordinary single-qubit phase (residual detuning, AC-Stark
shifts) accumulated over the sequence, so the extracted phase is purely
theta_CZ = theta_11 - theta_10 - theta_01 + theta_00, immune to control's
frequency calibration. This also means control's |0> and |1> conditions
don't need to be prepared and measured as two separate calibration runs --
the echo sweeps control through both states within a single sequence and
cancels the state-independent part automatically, making the experiment
faster than the naive conditioned-tomography approach.

Prerequisites:
    - Calibrated single-qubit gates (x90, x180) for both qubits in the pair.
    - Calibrated, state-discriminating readout for the stationary qubit.
    - An initial estimate of the CZ amplitude (e.g. from 32a_cz_conditional_phase
      or 32b_cz_conditional_phase_error_amp).

State update:
    - qubit_pair.macros[operation].flux_pulse_qubit.amplitude (fitted optimal CZ amplitude).
"""

node = QualibrationNode[Parameters, Quam](
    name="32c_JAZZ_N",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow local debug parameter overrides when running directly from IDE."""
    # node.parameters.qubit_pairs = ["q1-q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):  # pylint: disable=too-many-statements
    """Create the sweep axes and generate the QUA program for the JAZZ-N sequence."""
    unit(coerce_to_integer=True)

    if not node.parameters.use_state_discrimination:
        raise RuntimeError(
            "JAZZ-N reads the stationary qubit |1> population and therefore requires "
            "use_state_discrimination = True."
        )

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    qubit_roles_map = {}
    for qp in qubit_pairs:
        verify_moving_qubit(qp, operation=node.parameters.operation, log_callable=node.log)
        qubit_roles_map[qp.name] = QubitRoles.resolve(qp)
    node.namespace["qubit_roles_map"] = qubit_roles_map

    # Coerce N_min / N_max to the nearest 4k + 1 (>= 1) and warn if changed.
    n_min_req = int(node.parameters.N_min)
    n_max_req = int(node.parameters.N_max)
    n_min = coerce_to_4k_plus_1(n_min_req)
    n_max = coerce_to_4k_plus_1(n_max_req)
    if n_min > n_max:
        n_min, n_max = n_max, n_min
    if n_min != n_min_req:
        node.log(f"N_min {n_min_req} coerced to nearest 4k+1 value: {n_min}.")
    if n_max != n_max_req:
        node.log(f"N_max {n_max_req} coerced to nearest 4k+1 value: {n_max}.")

    n_avg = node.parameters.num_shots
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    n_values = np.arange(n_min, n_max + 1, 4, dtype=int)
    operation = node.parameters.operation

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "N": xr.DataArray(n_values, attrs={"long_name": "echo count N = 4k+1"}),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
    }

    with program() as node.namespace["qua_program"]:
        amp = declare(fixed)
        n = declare(int)
        n_op = declare(int)
        count = declare(int)
        n_st = declare_output_stream()
        state_sq = [declare(int) for _ in range(num_qubit_pairs)]
        state_sq_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                qubit_role = qubit_roles_map[qp.name]
                mq, sq = qubit_role.moving, qubit_role.stationary
                node.machine.initialize_qpu(target=mq)
                node.machine.initialize_qpu(target=sq)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(n_op, int(n_min), n_op <= int(n_max), n_op + 4):
                    with for_(*from_array(amp, amplitudes)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qubit_role = qubit_roles_map[qp.name]
                            mq, sq = qubit_role.moving, qubit_role.stationary
                            mq.reset(node.parameters.reset_type, node.parameters.simulate)
                            sq.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()
                            reset_frame(sq.xy.name)
                            reset_frame(mq.xy.name)

                            # Initial pi/2 on stationary; moving stays in |0>.
                            sq.xy.play("x90")
                            qp.align()

                            # First CZ (the "Z" preceding the (pi-Z)^N pattern).
                            qp.macros[operation].apply(amplitude_scale_qubit=amp)

                            # N echoes interleaved with N more CZs: [X_pi X_pi, CZ] x N.
                            with for_(count, 0, count < n_op, count + 1):
                                qp.align()
                                mq.xy.play("x180")
                                sq.xy.play("x180")
                                qp.align()
                                qp.macros[operation].apply(amplitude_scale_qubit=amp)

                            qp.align()
                            sq.xy.play("x90")
                            qp.align()

                            sq.readout_state(state_sq[ii])
                            save(state_sq[ii], state_sq_st[ii])

            align()

        with stream_processing():
            n_st.save("n")
            for ii in range(num_qubit_pairs):
                state_sq_st[ii].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"state_stationary{ii + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report.to_dict(), "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data."""
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
    qubit_roles_map = node.namespace["qubit_roles_map"]
    node.results["qubit_roles"] = {
        name: {field: getattr(role, field).name for field in role._fields} for name, role in qubit_roles_map.items()
    }


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubit_pairs"] = get_qubit_pairs(node)
    if "qubit_roles" in node.results:
        node.namespace["qubit_roles_map"] = {
            name: QubitRoles(**{field: node.machine.qubits[qname] for field, qname in roles.items()})
            for name, roles in node.results["qubit_roles"].items()
        }
    else:
        node.namespace["qubit_roles_map"] = {
            qp.name: QubitRoles.resolve(qp) for qp in node.namespace["qubit_pairs"]
        }


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process the raw data, run the coarse-to-fine fit and set node outcomes."""
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
    """Plot the raw and fitted data in a specific figure whose shape is given by qubit pair grid locations."""
    figures = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["qubit_pairs"])
    for fig in figures.values():
        plt.show()
    node.results["figures"] = {
        "jazz_n_map": figures["map"],
        "jazz_n_avg": figures["avg"],
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the CZ flux-pulse amplitude for every successfully fitted qubit pair."""
    operation = node.parameters.operation
    fit_results = node.results["fit_results"]
    with node.record_state_updates():
        for qp in node.namespace["qubit_pairs"]:
            if node.outcomes[qp.name] == "failed":
                node.log(f"Skipping state update for {qp.name}: fit failed.")
                continue
            qp.macros[operation].flux_pulse_qubit.amplitude = fit_results[qp.name]["optimal_amplitude"]


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save the calibration results."""
    node.save()


# %%
