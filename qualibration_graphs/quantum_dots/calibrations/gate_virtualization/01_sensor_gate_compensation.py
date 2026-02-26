# %% {Imports}
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

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
    plot_2d_scan,
    plot_compensation_fit,
)
from calibration_utils.gate_virtualization.analysis import (
    process_raw_dataset,
    update_compensation_matrix,
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
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.run_in_video_mode
)
def create_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Create 2D scan QUA programs for each sensor-device pair."""
    node.namespace["sensors"] = sensors = get_sensors(node)

    mapping = node.parameters.sensor_device_mapping
    if mapping is None:
        raise ValueError(
            "sensor_device_mapping must be provided. "
            "Automatic generation from the machine is not yet implemented."
        )

    # Build a program for each (sensor_gate, device_gate) pair
    programs = {}
    sweep_axes_all = {}
    for sensor_gate, device_gates in mapping.items():
        for device_gate in device_gates:
            pair_key = f"{sensor_gate}_vs_{device_gate}"
            node.parameters.x_axis_name = sensor_gate
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
def simulate_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
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
def execute_qua_program(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Execute all sensor-device pair scans sequentially and store raw data."""
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
def load_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Analyse each 2D scan to extract sensor-device cross-talk coefficients."""
    # TODO: implement analysis pipeline
    # fit_results = {}
    # for pair_key, ds_raw in node.results["ds_raw_all"].items():
    #     ds = process_raw_dataset(ds_raw, node)
    #     sensor_gate, device_gate = pair_key.split("_vs_")
    #     fit_results[pair_key] = extract_sensor_compensation_coefficients(
    #         ds, sensor_gate, device_gate
    #     )
    # node.results["fit_results"] = fit_results
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Plot each 2D scan and compensation fit overlays."""
    # TODO: implement plotting
    # node.results["figures"] = {}
    # for pair_key, ds_raw in node.results["ds_raw_all"].items():
    #     sensor_gate, device_gate = pair_key.split("_vs_")
    #     fig_scan = plot_2d_scan(
    #         ds_raw, title=f"Sensor Compensation: {pair_key}",
    #     )
    #     node.results["figures"][f"scan_{pair_key}"] = fig_scan
    #
    #     if "fit_results" in node.results and pair_key in node.results["fit_results"]:
    #         fig_fit = plot_compensation_fit(
    #             ds_raw, node.results["fit_results"][pair_key],
    #             sensor_gate, device_gate,
    #             title=f"Compensation Fit: {pair_key}",
    #         )
    #         node.results["figures"][f"fit_{pair_key}"] = fig_fit
    pass


# %% {Update_virtual_gate_matrix}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def update_virtual_gate_matrix(
    node: QualibrationNode[SensorCompensationParameters, Quam],
):
    """Update the compensation matrix with sensor gate coefficients."""
    # TODO: implement matrix update
    # if "fit_results" in node.results:
    #     for pair_key, fit_res in node.results["fit_results"].items():
    #         sensor_gate, device_gate = pair_key.split("_vs_")
    #         update_compensation_matrix(
    #             node, sensor_gate, device_gate, fit_res["coefficient"]
    #         )
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[SensorCompensationParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
