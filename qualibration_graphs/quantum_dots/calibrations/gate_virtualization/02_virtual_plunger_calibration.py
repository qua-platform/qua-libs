# %% {Imports}
import warnings

import numpy as np
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_virtualization.virtual_plunger_parameters import (
    VirtualPlungerParameters,
)
from calibration_utils.gate_virtualization.plotting import (
    plot_virtual_plunger_diagnostic,
)
from calibration_utils.gate_virtualization.analysis import (
    process_raw_dataset,
)
from calibration_utils.gate_virtualization.virtual_plunger_analysis import (
    extract_virtual_plunger_coefficients,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


description = """
        VIRTUAL PLUNGER CALIBRATION — 2D SCAN
Performs 2D scans of a plunger gate versus other device gates (plungers or
barriers) using compensated sensor gates. For plunger-plunger scans the charge
transition line slopes are extracted to determine the virtual plunger gate
transformation that decouples the quantum dots, allowing independent chemical
potential control.

This node assumes that the sensor gate compensation has already been applied
(node 01_sensor_gate_compensation).

The scan can be performed using OPX outputs, QDAC outputs, or a combination
of the two.

Prerequisites:
    - Calibrated sensor gate compensation (node 01).
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Calibrated resonators coupled to SensorDot components.
    - Registered QuantumDot and SensorDot elements in QUAM.
    - Configured VirtualGateSet with sensor compensation in the matrix.
    - (If using QDAC) Configured QdacSpec on each VoltageGate and VirtualDCSet.
"""


node = QualibrationNode[VirtualPlungerParameters, Quam](
    name="02_virtual_plunger_calibration",
    description=description,
    parameters=VirtualPlungerParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # node.parameters.plunger_device_mapping = {
    #     "virtual_dot_1": ["virtual_dot_2", "barrier_12"],
    # }
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Create 2D scan QUA programs for each plunger-device pair."""
    from calibration_utils.gate_virtualization.scan_utils import create_2d_scan_program

    node.namespace["sensors"] = sensors = get_sensors(node)
    p = node.parameters

    mapping = p.plunger_device_mapping
    if mapping is None:
        raise ValueError(
            "plunger_device_mapping must be provided. " "Automatic generation from the machine is not yet implemented."
        )

    programs = {}
    sweep_axes_all = {}
    for plunger_gate, device_gates in mapping.items():
        for device_gate in device_gates:
            pair_key = f"{plunger_gate}_vs_{device_gate}"
            qua_prog, sweep_axes = create_2d_scan_program(
                node,
                sensors,
                x_axis_name=plunger_gate,
                y_axis_name=device_gate,
                x_span=p.plunger_gate_span,
                x_points=p.plunger_gate_points,
                y_span=p.device_gate_span,
                y_points=p.device_gate_points,
                x_from_qdac=p.plunger_gate_from_qdac,
                y_from_qdac=p.device_gate_from_qdac,
            )
            programs[pair_key] = qua_prog
            sweep_axes_all[pair_key] = sweep_axes

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Simulate the first QUA program for sanity-checking."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    first_key = next(iter(node.namespace["programs"]))
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["programs"][first_key], node.parameters)
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Execute all plunger pair scans sequentially and store raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    datasets = {}
    for pair_key, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes_all"][pair_key])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    node.parameters.num_shots,
                    start_time=data_fetcher.t_start,
                )
            print(f"[{pair_key}] {job.execution_report()}")
        datasets[pair_key] = dataset
    node.results["ds_raw_all"] = datasets


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def analyse_data(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Analyse each 2D scan to extract virtual plunger gate coefficients."""
    plunger_set = set(node.parameters.plunger_gates or [])
    fit_results = {}
    for pair_key, ds_raw in node.results["ds_raw_all"].items():
        ds = process_raw_dataset(ds_raw, node)
        plunger_gate_name, device_gate_name = pair_key.split("_vs_", maxsplit=1)
        is_asymmetric = bool(plunger_set and not (plunger_gate_name in plunger_set and device_gate_name in plunger_set))
        fit_results[pair_key] = extract_virtual_plunger_coefficients(
            ds,
            plunger_gate_name=plunger_gate_name,
            device_gate_name=device_gate_name,
            asymmetric=is_asymmetric,
        )
    node.results["fit_results"] = fit_results


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def plot_data(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Plot each 2D scan with edge map, segments, and fitted T matrix."""
    fit_results = node.results.get("fit_results", {})
    figures = {
        pair_key: plot_virtual_plunger_diagnostic(
            process_raw_dataset(ds_raw, node),
            fit_results.get(pair_key),
            pair_key,
        )
        for pair_key, ds_raw in node.results["ds_raw_all"].items()
    }
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def update_state(
    node: QualibrationNode[VirtualPlungerParameters, Quam],
):
    """Update the compensation matrix with measured pair-wise cross-talk.

    The stored matrix *A* maps virtual → physical gate voltages (Volk et al.,
    Eq. 1, npj Quantum Information 5, 29, 2019).  Diagonal entries are 1;
    off-diagonal ``A[i, j]`` is the cross-talk from virtual gate *j* to
    physical gate *i*.

    For each calibrated pair ``plunger_vs_device``, the 2×2 matrix *M* has::

        M = [[   1,        α_xy ],
             [ α_yx,       1    ]]

    **Plunger–plunger pairs** (symmetric): both off-diagonal entries are
    meaningful and both are written:

        A[plunger, device] += α_xy   (device's effect on plunger)
        A[device, plunger] += α_yx   (plunger's effect on device)

    **Asymmetric pairs** (plunger vs barrier/sensor/...): only ``α_xy`` is
    meaningful — it captures the non-plunger gate's cross-talk onto the
    plunger dot.  The reciprocal ``α_yx`` from a charge stability scan does
    not measure the plunger's effect on that gate (sensors are calibrated
    in node 01, barriers in node 03).  So only one entry is written:

        A[plunger, non_plunger] += α_xy

    The ``plunger_gates`` parameter lists the plunger gate names.  A pair is
    symmetric only when *both* gates are in this list.  When ``plunger_gates``
    is None, all pairs are treated as symmetric.
    """
    if "fit_results" not in node.results:
        raise RuntimeError(
            "update_state called but 'fit_results' not found in node.results. " "Run analyse_data before update_state."
        )

    plunger_set = set(node.parameters.plunger_gates or [])

    for pair_key, fit_res in node.results["fit_results"].items():
        plunger_gate, device_gate = pair_key.split("_vs_", maxsplit=1)
        is_asymmetric = bool(plunger_set and not (plunger_gate in plunger_set and device_gate in plunger_set))

        if fit_res is None:
            if is_asymmetric:
                warnings.warn(f"Skipping asymmetric pair '{pair_key}': analysis returned None.")
                continue
            raise RuntimeError(f"fit_results['{pair_key}'] is None — analysis failed for this pair.")
        fit_params = fit_res.get("fit_params", {})
        if not fit_params.get("success", False):
            reason = fit_params.get("reason", "unknown")
            if is_asymmetric:
                warnings.warn(f"Skipping asymmetric pair '{pair_key}': {reason}")
                continue
            raise RuntimeError(f"Analysis for pair '{pair_key}' was not successful: {reason}")
        M = fit_res["T_matrix"]
        if M is None:
            raise RuntimeError(f"T_matrix is None for pair '{pair_key}' despite success=True.")

        vgs = None
        for candidate in node.machine.virtual_gate_sets.values():
            source_gates = candidate.layers[0].source_gates
            if plunger_gate in source_gates and device_gate in source_gates:
                vgs = candidate
                break
        if vgs is None:
            raise ValueError(
                f"Could not find a VirtualGateSet containing both " f"'{plunger_gate}' and '{device_gate}'."
            )

        source_gates = vgs.layers[0].source_gates
        plunger_row = source_gates.index(plunger_gate)
        device_row = source_gates.index(device_gate)

        delta = np.asarray(M, dtype=float)
        if delta.shape != (2, 2):
            raise ValueError(f"Expected a 2x2 T_matrix for '{pair_key}', got shape {delta.shape}.")

        layer = vgs.layers[0]
        target = "both" if vgs.id in node.machine.virtual_dc_sets else "opx"

        # Always write: device gate's effect on the plunger gate.
        alpha_xy = float(delta[0, 1])
        plunger_physical = layer.target_gates[plunger_row]
        plunger_ch = vgs.channels[plunger_physical]
        current_xy = float(layer.matrix[plunger_row][device_row])

        node.machine.update_cross_compensation_submatrix(
            virtual_names=[device_gate],
            channels=[plunger_ch],
            matrix=[[current_xy + alpha_xy]],
            target=target,
        )

        # Only write the reciprocal if both gates are plungers.
        both_plungers = not plunger_set or (  # no list → treat all as symmetric
            plunger_gate in plunger_set and device_gate in plunger_set
        )
        if both_plungers:
            alpha_yx = float(delta[1, 0])
            device_physical = layer.target_gates[device_row]
            device_ch = vgs.channels[device_physical]
            current_yx = float(layer.matrix[device_row][plunger_row])

            node.machine.update_cross_compensation_submatrix(
                virtual_names=[plunger_gate],
                channels=[device_ch],
                matrix=[[current_yx + alpha_yx]],
                target=target,
            )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
