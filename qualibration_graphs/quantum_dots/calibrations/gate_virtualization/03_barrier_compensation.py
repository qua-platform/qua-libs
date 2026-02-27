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
    BarrierCompensationParameters,
    get_voltage_arrays,
    create_2d_scan_program,
    plot_2d_scan,
    plot_compensation_fit,
)
from calibration_utils.gate_virtualization.analysis import (
    process_raw_dataset,
    update_compensation_matrix,
)
from calibration_utils.gate_virtualization.barrier_compensation_analysis import (
    extract_barrier_compensation_coefficients,
)
from calibration_utils.common_utils.experiment import get_sensors

from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher


description = """
        BARRIER COMPENSATION — 2D SCAN
Performs a 2D scan of a barrier gate versus a compensation gate (typically a
plunger) to measure barrier-plunger cross-talk. The extracted coefficient is
used to update the virtual gate compensation matrix so that tunnel barriers
can be adjusted independently without shifting charge occupation.

This node assumes that sensor gate compensation and virtual plunger calibration
have already been applied (nodes 01 and 02).

The scan can be performed using OPX outputs, QDAC outputs, or a combination
of the two.

Prerequisites:
    - Calibrated sensor gate compensation (node 01).
    - Calibrated virtual plunger gates (node 02).
    - Calibrated IQ mixer / Octave on the readout line.
    - Calibrated time of flight, offsets and gains.
    - Calibrated resonators coupled to SensorDot components.
    - Registered QuantumDot and SensorDot elements in QUAM.
    - Configured VirtualGateSet with sensor + plunger compensation.
    - (If using QDAC) Configured QdacSpec on each VoltageGate and VirtualDCSet.
"""


node = QualibrationNode[BarrierCompensationParameters, Quam](
    name="03_barrier_compensation",
    description=description,
    parameters=BarrierCompensationParameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes."""
    # node.parameters.barrier_compensation_mapping = {
    #     "barrier_12": ["virtual_dot_1", "virtual_dot_2"],
    # }
    pass


node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.run_in_video_mode
)
def create_qua_program(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Create 2D scan QUA programs for each barrier-compensation pair."""
    node.namespace["sensors"] = sensors = get_sensors(node)

    mapping = node.parameters.barrier_compensation_mapping
    if mapping is None:
        raise ValueError(
            "barrier_compensation_mapping must be provided. "
            "Automatic generation from the machine is not yet implemented."
        )

    programs = {}
    sweep_axes_all = {}
    for barrier_gate, comp_gates in mapping.items():
        for comp_gate in comp_gates:
            pair_key = f"{barrier_gate}_vs_{comp_gate}"
            node.parameters.x_axis_name = barrier_gate
            node.parameters.y_axis_name = comp_gate
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
def simulate_qua_program(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
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
def execute_qua_program(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
    """Execute all barrier-compensation pair scans sequentially and store raw data."""
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
def load_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Load a previously acquired dataset."""
    # TODO: implement historical data loading
    pass


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def analyse_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Analyse each 2D scan to extract barrier compensation coefficients."""
    # TODO: implement analysis pipeline
    # fit_results = {}
    # for pair_key, ds_raw in node.results["ds_raw_all"].items():
    #     ds = process_raw_dataset(ds_raw, node)
    #     barrier_gate, comp_gate = pair_key.split("_vs_")
    #     fit_results[pair_key] = extract_barrier_compensation_coefficients(
    #         ds, barrier_gate, comp_gate
    #     )
    # node.results["fit_results"] = fit_results
    pass


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def plot_data(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Plot each 2D scan and barrier compensation fit overlays."""
    # TODO: implement plotting
    # node.results["figures"] = {}
    # for pair_key, ds_raw in node.results["ds_raw_all"].items():
    #     barrier_gate, comp_gate = pair_key.split("_vs_")
    #     fig_scan = plot_2d_scan(
    #         ds_raw, title=f"Barrier Compensation: {pair_key}",
    #     )
    #     node.results["figures"][f"scan_{pair_key}"] = fig_scan
    #
    #     if "fit_results" in node.results and pair_key in node.results["fit_results"]:
    #         fig_fit = plot_compensation_fit(
    #             ds_raw, node.results["fit_results"][pair_key],
    #             barrier_gate, comp_gate,
    #             title=f"Barrier Compensation Fit: {pair_key}",
    #         )
    #         node.results["figures"][f"fit_{pair_key}"] = fig_fit
    pass


# %% {Update_virtual_gate_matrix}
@node.run_action(skip_if=node.parameters.run_in_video_mode)
def update_virtual_gate_matrix(
    node: QualibrationNode[BarrierCompensationParameters, Quam],
):
    """Update the compensation matrix with barrier compensation coefficients."""
    # TODO: implement matrix update
    # if "fit_results" in node.results:
    #     for pair_key, fit_res in node.results["fit_results"].items():
    #         barrier_gate, comp_gate = pair_key.split("_vs_")
    #         update_compensation_matrix(
    #             node, barrier_gate, comp_gate, fit_res["coefficient"]
    #         )
    pass


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[BarrierCompensationParameters, Quam]):
    """Save the node results and state."""
    # TODO: uncomment when complete
    # node.save()
    pass
