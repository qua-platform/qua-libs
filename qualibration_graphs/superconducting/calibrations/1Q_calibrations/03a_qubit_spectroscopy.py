"""Qubit spectroscopy calibration (v2: improved Lorentzian-fit analysis).

Same QUA program as the previous 03_qubit_spectroscopy node; uses the
improved analysis package `calibration_utils.qubit_spectroscopy` which
replaces the single-pass `peaks_dips` path with an explicit Lorentzian-peak
+ linear-bg `curve_fit`, smart initial guess, narrow-window refit, and
R²/FWHM/contrast quality gates. Every fit-derived value is shown in the plot
panel; failed fits get a red "(FAILED)" badge so update_state does not
silently overwrite QUAM state.
"""

# %% {Imports}
from dataclasses import asdict

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from qm.qua import *

from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit

from qualibrate import QualibrationNode
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from qualibration_libs.data import XarrayDataFetcher

from calibration_utils.qubit_spectroscopy import (
    Parameters,
    process_raw_dataset,
    fit_raw_data,
    log_fitted_results,
    plot_raw_data_with_fit,
)
from quam_config import Quam


import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))


# %% {Node initialisation}
description = """
        QUBIT SPECTROSCOPY (v2)
A saturation pulse drives the qubit across a detuning sweep
around its stored RF frequency; the resonator state is read
out and demodulated to I/Q for every drive frequency.

The post-processing now uses an explicit Lorentzian-peak + linear-background
`scipy.optimize.curve_fit` instead of the legacy `peaks_dips` call:
  - Savgol-smoothed `find_peaks` with a noise-floor prominence threshold for
    the initial f0 guess.
  - Local linear detrend + half-max crossings for a smart FWHM init.
  - First fit on a ±4×FWHM window, then a narrow-window refit using the
    first-pass values.
  - Quality gates: R² (default >= 0.85), FWHM bound (default 30 MHz),
    contrast = amp / |baseline| (default >= 0.05), plus the legacy saturation
    amplitude bound check.

Failed fits set node.outcomes[q.name] = "failed" so update_state skips them.

Prerequisites:
    - Mixer / Octave calibrated (01a or 01b).
    - Readout calibrated (02a/02b/02c).
    - Flux point specified (qubit.z.flux_point) if relevant.

State update (only when fit passes the new quality gates):
    - qubit.f_01 & qubit.xy.RF_frequency
    - qubit.resonator.operations["readout"].integration_weights_angle
    - (optional) qubit.xy.operations["saturation"].amplitude
    - (optional) qubit.xy.operations["x180"/"x90"].amplitude
"""


node = QualibrationNode[Parameters, Quam](
    name="03a_qubit_spectroscopy",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters for debugging."""
    pass


# Instantiate the QUAM class from the state file
# node.machine = Quam.load()


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program."""
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    operation = node.parameters.operation
    n_avg = node.parameters.num_shots
    operation_len = node.parameters.operation_len_in_ns
    operation_amp = node.parameters.operation_amplitude_factor
    span = node.parameters.frequency_span_in_mhz * u.MHz
    step = node.parameters.frequency_step_in_mhz * u.MHz
    dfs = np.arange(-span // 2, +span // 2, step)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "detuning": xr.DataArray(
            dfs,
            attrs={"long_name": "readout frequency", "units": "Hz"},
        ),
    }

    with program() as node.namespace["qua_program"]:
        I, I_st, Q, Q_st, n, n_st = node.machine.declare_qua_variables()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_avg, n + 1):
                save(n, n_st)
                with for_(*from_array(df, dfs)):
                    for i, qubit in multiplexed_qubits.items():
                        duration = operation_len if operation_len is not None else qubit.xy.operations[operation].length
                        qubit.xy.update_frequency(df + qubit.xy.intermediate_frequency)
                        qubit.xy.play(
                            operation,
                            amplitude_scale=operation_amp,
                            duration=duration // 4,
                        )
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure("readout", qua_vars=(I[i], Q[i]))
                        qubit.resonator.wait(node.machine.depletion_time * u.ns)
                        save(I[i], I_st[i])
                        save(Q[i], Q_st[i])
                    align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_st[i].buffer(len(dfs)).average().save(f"I{i + 1}")
                Q_st[i].buffer(len(dfs)).average().save(f"Q{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(
        qmm,
        config,
        node.namespace["qua_program"],
        node.parameters,
    )
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch the raw dataset."""
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


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Smart-init + curve_fit + narrow refit + R² gates."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], fit_results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Per-qubit panel with fit overlay + values box + FAILED indicator."""
    fig_raw_fit = plot_raw_data_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {"amplitude": fig_raw_fit}


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write fitted values to QUAM only for qubits whose fit passed quality gates."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit_result = node.results["fit_results"][q.name]
            q.f_01 = fit_result["frequency"]
            q.xy.RF_frequency = fit_result["frequency"]
            q.resonator.operations["readout"].integration_weights_angle = fit_result["iw_angle"]
            if node.parameters.update_pulses_amplitude:
                q.xy.operations["saturation"].amplitude = fit_result["saturation_amp"]
                q.xy.operations["x180"].amplitude = fit_result["x180_amp"]
                q.xy.operations["x90"].amplitude = fit_result["x180_amp"] / 2


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save node results."""
    node.save()
