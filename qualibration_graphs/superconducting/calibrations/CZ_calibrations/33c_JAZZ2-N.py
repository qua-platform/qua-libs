# pylint: disable=R0801
# pylint: disable=duplicate-code

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.cz_jazz2_n import (
    Parameters,
    coerce_to_even,
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


# %% {Initialisation}
description = """
        JAZZ2-N CZ AMPLITUDE CALIBRATION
This node calibrates the CZ-pulse amplitude using the JAZZ2-N protocol
(arXiv:2402.18926v3, Appendix I.1, Fig. 13(b)). The pulse sequence is

    x90(control) & x90(target)               (X_{pi/2} X_{pi/2})
    CZ                                       (initial Z)
    [X_pi(control) & X_pi(target) -- CZ] x (2N + 1)
    x90(control) & x90(target)               (X_{pi/2} X_{pi/2})
    measure(control), measure(target) -> p00 = (1 - state_c) * (1 - state_t)

where N = 2k (k = 0, 1, 2, ...). With the X_pi refocusing pulses on both
qubits, the joint ground-state probability evolves as

    P_|00>(amp, N) = (1 - cos((N + 1) * theta_CZ(amp))) / 2,

independently of any virtual-Z (single-qubit) phase shifts inside the CZ
macro. The optimal CZ amplitude is the value where theta_CZ = pi, i.e. where
P_|00> is maximal. Compared to JAZZ-N, the principal-peak fringe is denser in
amplitude for a given total pulse count, so this node is a sharper follow-up
amplitude calibration; the same measurement can also be used downstream as
the reward signal for Z-pulse shape optimisation.

The Z-pulse is supplied by the full CZGate macro selected via the
``operation`` parameter, so any qubit/coupler flux pulse shape that the
macro provides (cz_unipolar, cz_flattop, cz_bipolar, cz_flattop_erf, cz_SNZ)
can be calibrated.

Prerequisites:
    - Calibrated single-qubit gates (x90, x180) for both qubits in the pair.
    - Calibrated, state-discriminating readout for BOTH qubits.
    - An initial estimate of the CZ amplitude (e.g. from 33b_JAZZ-N).

State update:
    - qubit_pair.macros[operation].flux_pulse_qubit.amplitude (fitted optimal CZ amplitude).
"""

node = QualibrationNode[Parameters, Quam](
    name="33c_JAZZ2_N",
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
    """Create the sweep axes and generate the QUA program for the JAZZ2-N sequence."""
    unit(coerce_to_integer=True)

    if not node.parameters.use_state_discrimination:
        raise RuntimeError(
            "JAZZ2-N reads the joint P_|00> of the qubit pair and therefore requires "
            "use_state_discrimination = True."
        )

    node.namespace["qubit_pairs"] = qubit_pairs = get_qubit_pairs(node)
    num_qubit_pairs = len(qubit_pairs)

    # Coerce N_min / N_max to the nearest even integer (>= 0) and warn if changed.
    n_min_req = int(node.parameters.N_min)
    n_max_req = int(node.parameters.N_max)
    n_min = coerce_to_even(n_min_req)
    n_max = coerce_to_even(n_max_req)
    if n_min > n_max:
        n_min, n_max = n_max, n_min
    if n_min != n_min_req:
        node.log(f"N_min {n_min_req} coerced to nearest even value: {n_min}.")
    if n_max != n_max_req:
        node.log(f"N_max {n_max_req} coerced to nearest even value: {n_max}.")

    n_avg = node.parameters.num_shots
    amplitudes = np.arange(1 - node.parameters.amp_range, 1 + node.parameters.amp_range, node.parameters.amp_step)
    n_values = np.arange(n_min, n_max + 1, 2, dtype=int)
    operation = node.parameters.operation

    node.namespace["sweep_axes"] = {
        "qubit_pair": xr.DataArray(qubit_pairs.get_names()),
        "N": xr.DataArray(n_values, attrs={"long_name": "repetition count N = 2k"}),
        "amp": xr.DataArray(amplitudes, attrs={"long_name": "amplitude scale", "units": "a.u."}),
    }

    with program() as node.namespace["qua_program"]:
        amp = declare(fixed)
        n = declare(int)
        n_op = declare(int)
        count = declare(int)
        n_st = declare_output_stream()
        state_c = [declare(int) for _ in range(num_qubit_pairs)]
        state_t = [declare(int) for _ in range(num_qubit_pairs)]
        p00 = [declare(int) for _ in range(num_qubit_pairs)]
        p00_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_c_st = [declare_output_stream() for _ in range(num_qubit_pairs)]
        state_t_st = [declare_output_stream() for _ in range(num_qubit_pairs)]

        for multiplexed_qubit_pairs in qubit_pairs.batch():
            for qp in multiplexed_qubit_pairs.values():
                node.machine.initialize_qpu(target=qp.qubit_control)
                node.machine.initialize_qpu(target=qp.qubit_target)

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(n_op, n_min, n_op <= n_max, n_op + 2):
                    with for_(*from_array(amp, amplitudes)):
                        for ii, qp in multiplexed_qubit_pairs.items():
                            qp.qubit_control.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.qubit_target.reset(node.parameters.reset_type, node.parameters.simulate)
                            qp.align()
                            reset_frame(qp.qubit_target.xy.name)
                            reset_frame(qp.qubit_control.xy.name)

                            # Boundary X_{pi/2} X_{pi/2} (both qubits).
                            qp.qubit_control.xy.play("x90")
                            qp.qubit_target.xy.play("x90")
                            qp.align()

                            # First CZ (the "Z" preceding the (pi-Z)^(2N+1) pattern).
                            qp.macros[operation].apply(amplitude_scale_qubit=amp)

                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x180")
                            qp.align()

                            # First CZ (the "Z" preceding the (pi-Z)^(2N+1) pattern).
                            qp.macros[operation].apply(amplitude_scale_qubit=amp)

                            # (X_pi X_pi, CZ) x (2N + 1).
                            with for_(count, 1, count <= n_op, count + 1):
                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x180")
                                qp.align()
                                qp.macros[operation].apply(amplitude_scale_qubit=amp)
                                qp.qubit_control.xy.frame_rotation_2pi(0.5)
                                qp.qubit_target.xy.frame_rotation_2pi(0.5)
                                qp.qubit_control.xy.play("x180")
                                qp.qubit_target.xy.play("x180")
                                qp.align()
                                qp.macros[operation].apply(amplitude_scale_qubit=amp)
                                qp.qubit_control.xy.frame_rotation_2pi(-0.5)
                                qp.qubit_target.xy.frame_rotation_2pi(-0.5)

                            qp.align()
                            # Boundary X_{pi/2} X_{pi/2} (both qubits).
                            qp.qubit_control.xy.play("x90")
                            qp.qubit_control.xy.play("x180")
                            qp.qubit_target.xy.play("x90")
                            qp.qubit_target.xy.play("x180")
                            qp.align()

                            qp.qubit_control.readout_state(state_c[ii])
                            qp.qubit_target.readout_state(state_t[ii])
                            assign(p00[ii], (1 - state_c[ii]) * (1 - state_t[ii]))
                            save(p00[ii], p00_st[ii])
                            save(state_c[ii], state_c_st[ii])
                            save(state_t[ii], state_t_st[ii])

            align()

        with stream_processing():
            n_st.save("n")
            for ii in range(num_qubit_pairs):
                p00_st[ii].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"p{ii + 1}")
                state_c_st[ii].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"state_c{ii + 1}")
                state_t_st[ii].buffer(len(amplitudes)).buffer(len(n_values)).average().save(f"state_t{ii + 1}")


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
    """Plot the JAZZ2-N P_|00> map and the fitted optimal amplitude for each qubit pair."""
    fig = plot_raw_data_with_fit(node.results["ds_fit"], node.namespace["qubit_pairs"])
    plt.show()
    node.results["figures"] = {"jazz2_n_amplitude": fig}


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
