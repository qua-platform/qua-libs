# %% {Imports}
from qm.qua import *

from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.run_in_video_mode
)
def create_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Create 2D scan QUA programs for each plunger-device pair."""
    from calibration_utils.gate_virtualization.scan_utils import create_2d_scan_program

    node.namespace["sensors"] = sensors = get_sensors(node)

    mapping = node.parameters.plunger_device_mapping
    if mapping is None:
        raise ValueError(
            "plunger_device_mapping must be provided. "
            "Automatic generation from the machine is not yet implemented."
        )

    programs = {}
    sweep_axes_all = {}
    for plunger_gate, device_gates in mapping.items():
        for device_gate in device_gates:
            pair_key = f"{plunger_gate}_vs_{device_gate}"
            node.parameters.x_axis_name = plunger_gate
            node.parameters.y_axis_name = device_gate
            qua_prog, sweep_axes = create_2d_scan_program(node, sensors)
            programs[pair_key] = qua_prog
            sweep_axes_all[pair_key] = sweep_axes

    node.namespace["programs"] = programs
    node.namespace["sweep_axes_all"] = sweep_axes_all


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Simulate the first QUA program for sanity-checking."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    first_key = next(iter(node.namespace["programs"]))
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["programs"][first_key], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.run_in_video_mode
)
def execute_qua_program(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Execute all plunger pair scans sequentially and store raw data."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    datasets = {}
    for pair_key, qua_prog in node.namespace["programs"].items():
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            job = qm.execute(qua_prog)
            data_fetcher = XarrayDataFetcher(
                job, node.namespace["sweep_axes_all"][pair_key]
            )
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
    fit_results = {}
    for pair_key, ds_raw in node.results["ds_raw_all"].items():
        ds = process_raw_dataset(ds_raw, node)
        fit_results[pair_key] = extract_virtual_plunger_coefficients(
            ds,
            plunger_x_name="x_volts",
            plunger_y_name="y_volts",
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
    """Write the plunger cross-talk correction into the compensation layer.

    The analysis returns T (physical→virtual, close to identity).
    The QUAM compensation layer stores a matrix C whose inverse the
    hardware applies when resolving virtual→physical voltages:

        v_physical = C⁻¹ @ v_virtual

    Setting C = T gives C⁻¹ = M (virtual→physical), which decouples
    the two dots.  The 2×2 plunger block is written as::

        C[plunger_x, plunger_x] = T[0,0]   C[plunger_x, plunger_y] = T[0,1]
        C[plunger_y, plunger_x] = T[1,0]   C[plunger_y, plunger_y] = T[1,1]
    """
    if "fit_results" not in node.results:
        return

    for pair_key, fit_res in node.results["fit_results"].items():
        if fit_res is None:
            continue
        if not fit_res.get("fit_params", {}).get("success", False):
            continue
        plunger_gate, device_gate = pair_key.split("_vs_")
        T = fit_res["T_matrix"]
        if T is None:
            continue

        # Find the VirtualGateSet that owns both virtual gates.
        vgs = None
        for candidate in node.machine.virtual_gate_sets.values():
            source_gates = candidate.layers[0].source_gates
            if plunger_gate in source_gates and device_gate in source_gates:
                vgs = candidate
                break
        if vgs is None:
            raise ValueError(
                f"Could not find a VirtualGateSet containing both "
                f"'{plunger_gate}' and '{device_gate}'."
            )

        source_gates = vgs.layers[0].source_gates
        plunger_row = source_gates.index(plunger_gate)
        device_row = source_gates.index(device_gate)

        # Map virtual row names -> physical channels used by the machine API.
        plunger_physical = vgs.layers[0].target_gates[plunger_row]
        device_physical = vgs.layers[0].target_gates[device_row]
        plunger_ch = vgs.channels[plunger_physical]
        device_ch = vgs.channels[device_physical]

        # Keep OPX and external DC compensation layers aligned when available.
        target = "both" if vgs.id in node.machine.virtual_dc_sets else "opx"
        node.machine.update_cross_compensation_submatrix(
            virtual_names=[plunger_gate, device_gate],
            channels=[plunger_ch, device_ch],
            matrix=[
                [float(T[0, 0]), float(T[0, 1])],
                [float(T[1, 0]), float(T[1, 1])],
            ],
            target=target,
        )


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[VirtualPlungerParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
