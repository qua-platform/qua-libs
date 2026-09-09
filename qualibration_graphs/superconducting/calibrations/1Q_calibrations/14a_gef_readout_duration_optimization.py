# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
from qualibrate import QualibrationNode
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam
from calibration_utils.readout_duration_optimization import (
    GEFParameters as Parameters, process_raw_dataset, fit_raw_data, log_fitted_results, plot_results,
)
from calibration_utils.readout_duration_optimization import experiment

# %% {Description}
description = """GEF READOUT DURATION OPTIMIZATION
Sweep readout_GEF duration with individual g/e/f IQ shots at the existing GEF_frequency_shift.
Prepare f with x180 followed by EF_x180 at IF minus anharmonicity. Fit a three-component GMM,
then maximize nearest-center assignment fidelity subject to the 08b non-outlier criterion.
Prerequisites: calibrated x180/EF_x180 and GEF readout frequency; square/default integration weights.
Updates: readout_GEF.length and resonator.gef_centers in raw demodulation units.
If readout_GEF is absent, use readout as the template and create readout_GEF on success.
Equal fidelities favor the shortest duration. The three-state confusion matrix is saved in results.

Times are in ns and must be multiples of 4. This sweeps actual pulse length, not accumulated SNR.
Simulation and historical-data analysis do not update the machine state.
"""

node = QualibrationNode[Parameters, Quam](
    name="14a_gef_readout_duration_optimization", description=description, parameters=Parameters(), machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["qA1", "qA2"]
    # node.parameters.min_duration_in_ns = 200
    # node.parameters.max_duration_in_ns = 2000
    # node.parameters.duration_step_in_ns = 200
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    experiment.create_qua_program(node)


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    samples, fig, report = simulate_and_plot(
        node.machine.connect(), node.namespace["config"], node.namespace["qua_program"], node.parameters,
    )
    node.results["simulation"] = {"figure": fig, "wf_report": report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    with qm_session(qmm, node.namespace["config"], timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        dataset = None
        for dataset in fetcher:
            progress_counter(fetcher.get("n", 0), node.parameters.num_shots, start_time=fetcher.t_start)
        node.log(job.execution_report())
    if dataset is None:
        raise RuntimeError("No readout duration data were returned")
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["ds_iq_blobs"], results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {name: asdict(result) for name, result in results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {name: "successful" if result.success else "failed" for name, result in results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    node.results["figures"] = plot_results(
        node.results["ds_raw"], node.namespace["qubits"], node.results["ds_fit"], node.results["ds_iq_blobs"],
    )
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate or node.parameters.load_data_id is not None)
def update_state(node: QualibrationNode[Parameters, Quam]):
    experiment.update_state(node)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
