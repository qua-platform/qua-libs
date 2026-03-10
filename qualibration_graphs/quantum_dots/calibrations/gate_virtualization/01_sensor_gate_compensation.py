# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from quam_config import Quam
from calibration_utils.gate_virtualization import (
    SensorCompensationParameters,
    get_voltage_arrays,
    create_2d_scan_program,
    read_qdac_voltage,
    plot_sensor_compensation_diagnostic,
)
from calibration_utils.gate_virtualization.analysis import (
    process_raw_dataset,
)
from calibration_utils.gate_virtualization.sensor_compensation_analysis import (
    extract_sensor_compensation_coefficients,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


description = """
            SENSOR GATE COMPENSATION — 2D SCAN
Performs a 2D voltage scan of a sensor gate versus a device gate (plunger or
barrier) to measure the cross-talk between them. The extracted coefficient is
used to update the off-diagonal entry in the virtual gate compensation matrix
so that sensor dot operating points remain stable when device gates are swept.

The scan can be performed using OPX outputs, QDAC outputs, or a combination
of the two.

Prerequisites:
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Calibrated resonators coupled to SensorDot components.
    - Registered QuantumDot and SensorDot elements in QUAM.
    - Configured VirtualGateSet with initial (identity) compensation matrix.
    - (If using QDAC) Configured QdacSpec on each VoltageGate and VirtualDCSet.
"""


node = QualibrationNode[SensorCompensationParameters, Quam](
    name="01_sensor_gate_compensation",
    description=description,
    parameters=SensorCompensationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # node.parameters.x_axis_name = "virtual_sensor_1"
    # node.parameters.y_axis_name = "virtual_dot_1"
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.run_in_video_mode)
def create_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Create 2D scan QUA programs for each sensor-device pair."""
    node.namespace["sensors"] = sensors = get_sensors(node)

    mapping = node.parameters.sensor_device_mapping
    if mapping is None:
        raise ValueError(
            "sensor_device_mapping must be provided. " "Automatic generation from the machine is not yet implemented."
        )

    # Build a program for each (sensor_gate, device_gate) pair
    p = node.parameters
    programs = {}
    sweep_axes_all = {}
    centers_all = {}
    for sensor_gate, device_gates in mapping.items():
        for device_gate in device_gates:
            pair_key = f"{sensor_gate}_vs_{device_gate}"

            # Read pre-sweep QDAC voltages so we can return to them after the scan.
            x_center = None
            if p.sensor_gate_from_qdac:
                x_obj = node.machine.get_component(sensor_gate)
                x_center = read_qdac_voltage(node, x_obj)
            y_center = None
            if p.device_gate_from_qdac:
                y_obj = node.machine.get_component(device_gate)
                y_center = read_qdac_voltage(node, y_obj)

            qua_prog, sweep_axes = create_2d_scan_program(
                node,
                sensors,
                x_axis_name=sensor_gate,
                y_axis_name=device_gate,
                x_span=p.sensor_gate_span,
                x_points=p.sensor_gate_points,
                y_span=p.device_gate_span,
                y_points=p.device_gate_points,
                x_from_qdac=p.sensor_gate_from_qdac,
                y_from_qdac=p.device_gate_from_qdac,
                x_center=x_center,
                y_center=y_center,
            )
            programs[pair_key] = qua_prog
            sweep_axes_all[pair_key] = sweep_axes
            centers_all[pair_key] = (sensor_gate, device_gate, x_center, y_center)

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all
    node.namespace["centers_all"] = centers_all


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
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
def execute_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Execute all sensor-device pair scans sequentially and store raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    p = node.parameters
    datasets = {}
    for pair_key, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=p.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes_all"][pair_key])
            for dataset in data_fetcher:
                progress_counter(
                    data_fetcher.get("n", 0),
                    p.num_shots,
                    start_time=data_fetcher.t_start,
                )
            print(f"[{pair_key}] {job.execution_report()}")
        datasets[pair_key] = dataset

    node.results["ds_raw_all"] = datasets


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def analyse_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Analyse each 2D scan to extract sensor-device cross-talk coefficients."""
    fit_results = {}
    for pair_key, ds_raw in node.results["ds_raw_all"].items():
        ds = process_raw_dataset(ds_raw, node)
        sensor_gate_name, device_gate_name = pair_key.split("_vs_", maxsplit=1)
        fit_results[pair_key] = extract_sensor_compensation_coefficients(
            ds,
            sensor_gate_name=sensor_gate_name,
            device_gate_name=device_gate_name,
        )
    node.results["fit_results"] = fit_results


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def plot_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Plot each 2D scan with Lorentzian fit overlay and residual panel."""
    fit_results = node.results.get("fit_results", {})
    figures = {
        pair_key: plot_sensor_compensation_diagnostic(
            process_raw_dataset(ds_raw),
            fit_results.get(pair_key),
            pair_key,
        )
        for pair_key, ds_raw in node.results["ds_raw_all"].items()
    }
    node.results["figures"] = figures


# %% {Update_state}
@node.run_action(skip_if=node.parameters.run_in_video_mode or node.parameters.simulate)
def update_state(
    node: QualibrationNode[SensorCompensationParameters, Quam],
):
    """Update the compensation matrix with the measured sensor-device cross-talk.

    Because the scan is performed through the virtual gate voltage sequence,
    the measured alpha is the *residual* cross-talk after any existing
    compensation.  The update is therefore additive:

        M[sensor_row][device_col] += alpha_measured

    so that repeated calibration runs converge towards zero residual cross-talk.
    The diagonal is never modified (sensor self-coupling stays 1).
    """
    if "fit_results" not in node.results:
        return

    for pair_key, fit_res in node.results["fit_results"].items():
        sensor_gate, device_gate = pair_key.split("_vs_", maxsplit=1)
        alpha_measured = fit_res["coefficient"]

        # Find the VirtualGateSet that owns both gates.
        vgs = None
        for candidate in node.machine.virtual_gate_sets.values():
            source_gates = candidate.layers[0].source_gates
            if sensor_gate in source_gates and device_gate in source_gates:
                vgs = candidate
                break
        if vgs is None:
            raise ValueError(
                f"Could not find a VirtualGateSet containing both " f"'{sensor_gate}' and '{device_gate}'."
            )

        # Read current entry and compute the updated value (add-residual).
        source_gates = vgs.layers[0].source_gates
        sensor_row = source_gates.index(sensor_gate)
        device_col = source_gates.index(device_gate)
        current = vgs.layers[0].matrix[sensor_row][device_col]
        new_alpha = current + alpha_measured

        # Map virtual name → physical name → channel object.
        # vgs.channels is keyed by physical names (target_gates), not virtual names (source_gates).
        sensor_physical = vgs.layers[0].target_gates[sensor_row]
        sensor_ch = vgs.channels[sensor_physical]

        # Update OPX matrix; also update the DC set if one exists.
        target = "both" if vgs.id in node.machine.virtual_dc_sets else "opx"
        node.machine.update_cross_compensation_submatrix(
            virtual_names=[device_gate],
            channels=[sensor_ch],
            matrix=[[new_alpha]],
            target=target,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
