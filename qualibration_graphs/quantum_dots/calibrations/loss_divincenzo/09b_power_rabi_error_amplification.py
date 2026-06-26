# %% {Imports}
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from calibration_utils.common_utils.experiment import (
    progress_counter_with_log,
)

from qualibrate.core import QualibrationNode
from quam_config import Quam
from calibration_utils.common_utils.experiment import get_qubits, enable_dual_drive_mw
from calibration_utils.common_utils.parity_streams import (
    declare_parity_streams,
    save_parity_measurement,
    buffer_parity_streams,
    process_parity_streams,
)
from calibration_utils.common_utils.annotation import annotate_node_figures
from calibration_utils.power_rabi import (
    ErrorAmplifiedParameters as Parameters,
    fit_raw_data_error_amplified,
    log_fitted_results_error_amplified,
    plot_raw_data_error_amplified,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Description}
description = """
        Power Rabi with Error Amplification
This sequence is a power Rabi sequence augmented with error-amplification. It involves parking the qubit at the
manipulation bias point, playing a pulse sequence with N pulses, and measuring the state of the resonator across
different qubit pulse amplitudes, showing Rabi oscillations. With error amplification, small amplitude errors accumulate
rapidly, allowing for a more precise calibration of the pulse amplitude. The results are then analyzed to determine the
qubit pulse amplitude suitable for the selected gate duration.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the qubit frequency.
    - Having set the qubit gate duration.

State update:
    - The qubit pulse amplitude corresponding to the specified operation (x180, x90...).
"""

# Be sure to include [Parameters, Quam] so the node has proper type hinting
node = QualibrationNode[Parameters, Quam](
    name="09b_power_rabi_error_amplification",
    description=description,
    parameters=Parameters(),
)


# Any parameters that should change for debugging purposes only should go in here
# These parameters are ignored when run through the GUI or as part of a graph
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging purposes, or execution in the Python IDE."""
    # node.parameters.qubits = ["Q1", "Q2"]
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    node.namespace["qubits"] = qubits = get_qubits(node)

    n_pulses = np.arange(2, node.parameters.max_n_pulses, 2)
    n_avg = node.parameters.num_shots
    # Pulse amplitude sweep (as a pre-factor of the qubit pulse amplitude) - must be within [-2; 2)
    amps = np.arange(
        node.parameters.min_amp_factor,
        node.parameters.max_amp_factor,
        node.parameters.amp_factor_step,
    )

    # Register the sweep axes to be added to the dataset when fetching data
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_pulses": xr.DataArray(n_pulses, attrs={"long_name": "number of pi pulses"}),
        "amp_prefactor": xr.DataArray(
            amps, attrs={"long_name": "pulse amplitude prefactor"}
        ),
    }

    with program() as node.namespace["qua_program"]:
        enable_dual_drive_mw(node)

        # Declare QUA variables
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        n = declare(int)
        m = declare(int)
        n_rabi = declare(int)

        # Post measurement (and optional pre measurement)
        p2, p1, parity_streams = declare_parity_streams(node, qubits)

        heralded_and_return_n_loops = getattr(node.parameters, "return_n_loops", False)
        if heralded_and_return_n_loops:
            n_loops_st = {qubit.name: declare_output_stream() for qubit in qubits}

        n_st = declare_output_stream()

        # Main experiment loop
        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(n_rabi, n_pulses)):
                    with for_(*from_array(a, amps)):
                        if node.parameters.parity_pre_measurement:
                            qubit.empty()
                            a1 = qubit.measure()

                        n_init = qubit.initialize()

                        align()
                        with for_(m, 0, m < n_rabi, m + 1):
                            qubit.x180(amplitude_scale=a)
                        align()

                        a2 = qubit.measure()

                        qubit.voltage_sequence.ramp_to_zero()

                        align()
                        assign(p2, Cast.to_int(a2))

                        if node.parameters.parity_pre_measurement:
                            assign(p1, Cast.to_int(a1))

                        save_parity_measurement(node, qubit.name, p1, p2, parity_streams)
                        if heralded_and_return_n_loops:
                            save(n_init, n_loops_st[qubit.name])

        # Stream processing
        with stream_processing():
            n_st.save("n")
            n_amps = len(amps)
            pulse_number = len(n_pulses)
            for qubit in qubits:
                buffer_parity_streams(node, qubit.name, parity_streams, pulse_number, n_amps)
                if heralded_and_return_n_loops:
                    n_loops_st[qubit.name].buffer(n_amps).buffer(pulse_number).average().save(
                        f"n_loops_{qubit.name}"
                    )


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate
)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program"""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm, config, node.namespace["qua_program"], node.parameters
    )
    node.results["simulation"] = {
        "figure": fig,
        "wf_report": wf_report,
        "samples": samples,
    }


# %% {Execute}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None or node.parameters.simulate
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"])
        data_fetcher = XarrayDataFetcher(job, node.namespace["sweep_axes"])
        for dataset in data_fetcher:
            progress_counter_with_log(
                data_fetcher.get("n", 0),
                node.parameters.num_shots,
                start_time=data_fetcher.t_start,
                node=node
            )
        node.log(job.execution_report())
    node.results["ds_raw"] = dataset


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Process_raw_data}
@node.run_action(skip_if=node.parameters.simulate)
def process_raw_data(node: QualibrationNode[Parameters, Quam]):
    """Compute conditional expectations from joint-outcome streams."""
    node.results["ds_raw"] = process_parity_streams(
        node.results["ds_raw"],
        [q.name for q in node.namespace["qubits"]],
        parity_pre_measurement=node.parameters.parity_pre_measurement,
        sweep_dims=("n_pulses", "amp_prefactor"),
    )


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Analyse the raw data and store the fitted data in another xarray dataset "ds_fit" and the "fit_results" dictionary."""
    ds_fit, fit_results = fit_raw_data_error_amplified(node.results["ds_raw"], node)
    node.results["ds_fit"] = ds_fit
    node.results["fit_results"] = fit_results
    log_fitted_results_error_amplified(fit_results, log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if r["success"] else "failed")
        for qname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the raw and fitted data."""
    fig = plot_raw_data_error_amplified(
        node.results["ds_raw"],
        node.results.get("ds_fit"),
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    node.results["figure"] = fig
    node.results["figures"] = {
        "power_rabi_error_amplified": fig,
    }
    annotate_node_figures(node)


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Update the relevant parameters if the qubit data analysis was successful."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue

            opt_prefactor = node.results["fit_results"][q.name]["opt_amp"]
            q.x.update(pi_amplitude=opt_prefactor * q.x.pi_pulse.amplitude)

# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    node.save()
