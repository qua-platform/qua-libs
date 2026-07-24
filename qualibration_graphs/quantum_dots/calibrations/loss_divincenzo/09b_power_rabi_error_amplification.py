# %% {Imports}
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter

from qualibrate.core import QualibrationNode
from quam_config import Quam
from qualibration_libs.parameters.experiment import get_qubits
from calibration_utils.measurement_utils import (
    declare_streams,
    save_measurement,
    buffer_streams,
)
from calibration_utils.power_rabi_error_amplification import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_all,
    generate_simulated_dataset,
)
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

# %% {Node initialisation}
description = """
        POWER RABI WITH ERROR AMPLIFICATION
This sequence performs a 2D power-Rabi measurement with error amplification: for each amplitude prefactor, an even
number of π pulses is played and the spin state is measured. Joint-outcome streams are averaged and reduced to
conditional expectations. Small amplitude errors accumulate over many pulses, enabling a precise refinement of
the π-pulse amplitude prefactor in a narrow window around the value from node 09a.

Prerequisites:
    - Having calibrated the relevant voltage points.
    - Having calibrated the qubit frequency and gate duration.
    - Having run node 09a_power_rabi to obtain a coarse π-pulse amplitude prefactor.

Datasets:
    - ``ds_raw``: untouched joint-outcome streams fetched from the OPX (never modified after acquisition).
    - ``ds_fit``: processed 2D sweeps plus analysis outputs (conditional expectations and mean-signal diagnostics).
      Used by ``plot_data``.
    - ``fit_results``: compact per-qubit calibration dict (``FitParameters`` serialized with ``asdict``). Used by
      logging, ``node.outcomes``, and ``update_state``.

Results (``node.results["fit_results"][qubit]``):
    - ``success``: whether the fit passed sanity checks and the state update is applied.
    - ``opt_amp``: refined amplitude prefactor for a π rotation.
    - ``rabi_frequency`` [rad / (unit amplitude · pulse)]: fitted Rabi frequency from the mean-signal model.
    - ``decay_rate`` [1 / pulse]: exponential decay rate per pulse in the error-amplification sequence.
    - ``gauss_decay_rate`` [1 / pulse]: Gaussian decay contribution per pulse.
    - ``n_eff``: effective number of pulses before the contrast envelope decays to 1/e.

Figures (``node.results["figures"]``):
    - ``"heatmap"``: 2D map of the analysis signal vs amplitude prefactor and number of pulses.
    - ``"resonance"``: n_pulses-averaged signal vs amplitude with analytic fit overlay.

State update:
    - The x180 pulse amplitude (``q.x.pi_amplitude``) scaled by the fitted amplitude prefactor.
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
    node.parameters.qubits = ["q1", "q2"]
    node.parameters.use_simulated_data = True  # run analysis without hardware
    pass


# Instantiate the QUAM class from the state file
node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.use_simulated_data)
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
        # Declare QUA variables
        a = declare(fixed)  # QUA variable for the qubit drive amplitude pre-factor
        n = declare(int)
        m = declare(int)
        n_rabi = declare(int)

        # Post measurement (and optional pre measurement)
        p2, p1, parity_streams = declare_streams(node, qubits)

        n_st = declare_output_stream()

        # Main experiment loop
        for qubit in qubits:
            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(n_rabi, n_pulses)):
                    with for_(*from_array(a, amps)):
                        if node.parameters.parity_measurement:
                            qubit.empty()
                            a1 = qubit.measure()

                        qubit.initialize(
                            target_state=node.parameters.target_state,
                            max_loops=node.parameters.max_loops,
                            conditional_drive=True,
                        )

                        align()
                        with for_(m, 0, m < n_rabi, m + 1):
                            qubit.x180(amplitude_scale=a)
                        align()

                        a2 = qubit.measure()

                        qubit.voltage_sequence.ramp_to_zero()

                        align()
                        assign(p2, Cast.to_int(a2))

                        if node.parameters.parity_measurement:
                            assign(p1, Cast.to_int(a1))

                        save_measurement(node, qubit.name, p1, p2, parity_streams)

        # Stream processing
        with stream_processing():
            n_st.save("n")
            n_amps = len(amps)
            pulse_number = len(n_pulses)
            for qubit in qubits:
                buffer_streams(node, qubit.name, parity_streams, pulse_number, n_amps)


# %% {Simulate}
@node.run_action(
    skip_if=node.parameters.load_data_id is not None
    or not node.parameters.simulate
    or node.parameters.use_simulated_data
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
    skip_if=node.parameters.load_data_id is not None
    or node.parameters.simulate
    or node.parameters.use_simulated_data
)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP, execute the QUA program and fetch the raw data and store it in a xarray dataset called "ds_raw"."""
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


# %% {Load_historical_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    """Load a previously acquired dataset."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(node.parameters.load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Generate_simulated_data}
@node.run_action(skip_if=not node.parameters.use_simulated_data)
def generate_simulated_data(node: QualibrationNode[Parameters, Quam]):
    """Generate simulated error-amplified power-Rabi data so the full analysis pipeline can run without hardware."""
    node.results["ds_raw"] = generate_simulated_dataset(node)
    node.log("[sim] Simulated dataset generated successfully.")


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Process joint-outcome streams, fit error-amplified power-Rabi data, and store results."""
    ds_processed = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(ds_processed, node)
    node.results["fit_results"] = fit_results
    log_fitted_results(fit_results, log_callable=node.log)
    node.outcomes = {
        qname: ("successful" if r["success"] else "failed")
        for qname, r in fit_results.items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot processed data and fit overlays; store figures in ``node.results["figures"]``."""
    node.results["figures"] = plot_all(
        node.results["ds_fit"],
        node.namespace["qubits"],
        node.results.get("fit_results", {}),
        analysis_signal=node.parameters.analysis_signal,
    )
    if not node.modes.external:
        plt.show()



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
