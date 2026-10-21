"""G-E-F readout frequency optimization (node 14a)."""

# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from calibration_utils.common_utils import ensure_gef_readout_pulse, set_gef_readout_frequency
from calibration_utils.readout_gef_frequency_optimization import (
    Parameters,
    fit_raw_data,
    log_fitted_results,
    plot_IQ_abs_with_fit,
    plot_distances_with_fit,
    process_raw_dataset,
)
from qm.qua import *
from qualang_tools.loops import from_array
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam_config import Quam

# %% {Description}
description = """
        G-E-F READOUT FREQUENCY OPTIMIZATION
This sequence sweeps the readout resonator intermediate frequency around the current operating point while preparing
the qubit successively in |g>, |e>, and |f> states. For every tested detuning, three IQ blobs (g, e, f) are acquired.
The distances between the three centroids are computed and the optimal detuning maximizes the minimum of
{d_ge, d_ef, d_gf}. That detuning is added to `GEF_frequency_shift`.

Prerequisites:
    - Resonator frequency & power calibrated (nodes 02a, 08a, 08b as relevant).
    - Qubit ge and ef pi pulses calibrated.
    - Proper thermalization time set. The node rejects reset_type != "thermal" because active reset
      would judge the qubit through the detuned readout pulse.

State update:
    - Adds the fitted optimal detuning to `qubit.resonator.GEF_frequency_shift`.
    - Stores the resulting absolute g/e/f readout frequency in `qubit.resonator.f_12` (informational).
"""

node = QualibrationNode[Parameters, Quam](
    name="14a_gef_readout_frequency_optimization",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Allow the user to locally set the node parameters."""
    # You can get type hinting in your IDE by typing node.parameters.
    # node.parameters.qubits = ["q1", "q2"]
    pass


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create the sweep axes and generate the QUA program from the pulse sequence and the node parameters."""
    if node.parameters.reset_type != "thermal":
        raise ValueError("Only 'thermal' reset is supported")
    u = unit(coerce_to_integer=True)
    node.namespace["qubits"] = qubits = get_qubits(node)
    num_qubits = len(qubits)

    ensure_gef_readout_pulse(qubits, log_callable=node.log)

    n_runs = node.parameters.num_shots
    operation = node.parameters.operation
    frequencies = np.arange(
        -node.parameters.frequency_span_in_mhz * u.MHz / 2,
        node.parameters.frequency_span_in_mhz * u.MHz / 2,
        node.parameters.frequency_step_in_mhz * u.MHz,
    )
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "frequency": xr.DataArray(frequencies, attrs={"long_name": "readout frequency shift in MHz"}),
    }

    with program() as node.namespace["qua_program"]:
        I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
        I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
        I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()
        df = declare(int)

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < n_runs, n + 1):
                save(n, n_st)
                with for_(*from_array(df, frequencies)):
                    for i, qubit in multiplexed_qubits.items():
                        set_gef_readout_frequency(qubit, extra_detuning=df)
                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_g[i], I_g_st[i])
                        save(Q_g[i], Q_g_st[i])
                    align()

                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        qubit.align()
                        qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_e[i], I_e_st[i])
                        save(Q_e[i], Q_e_st[i])

                    for i, qubit in multiplexed_qubits.items():
                        qubit.wait(2 * qubit.thermalization_time * u.ns)
                    align()
                    for i, qubit in multiplexed_qubits.items():
                        qubit.xy.play("x180")
                        update_frequency(
                            qubit.xy.name,
                            qubit.xy.intermediate_frequency - abs(qubit.anharmonicity),
                            keep_phase=True,
                        )
                        qubit.xy.play("EF_x180")
                        update_frequency(qubit.xy.name, qubit.xy.intermediate_frequency, keep_phase=True)
                        qubit.align()
                        qubit.resonator.measure(operation, qua_vars=(I_f[i], Q_f[i]))
                        qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                        save(I_f[i], I_f_st[i])
                        save(Q_f[i], Q_f_st[i])

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_g_st[i].buffer(len(frequencies)).average().save(f"Ig{i + 1}")
                Q_g_st[i].buffer(len(frequencies)).average().save(f"Qg{i + 1}")
                I_e_st[i].buffer(len(frequencies)).average().save(f"Ie{i + 1}")
                Q_e_st[i].buffer(len(frequencies)).average().save(f"Qe{i + 1}")
                I_f_st[i].buffer(len(frequencies)).average().save(f"If{i + 1}")
                Q_f_st[i].buffer(len(frequencies)).average().save(f"Qf{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Connect to the QOP and simulate the QUA program."""
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the QUA program and fetch raw data into ds_raw."""
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
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)


# %% {Load_data}
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
    """Analyse the raw data and store ds_fit and fit_results."""
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
    """Plot distances and |IQ| vs detuning on the qubit grid."""
    fig_distances = plot_distances_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    fig_iq_abs = plot_IQ_abs_with_fit(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
    )
    plt.show()
    node.results["figures"] = {
        "fitted_distances": fig_distances,
        "iq_abs": fig_iq_abs,
    }


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write GEF_frequency_shift and informational f_12 when the fit succeeded."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            res = node.machine.qubits[q.name].resonator
            new_shift = (res.GEF_frequency_shift or 0) + node.results["fit_results"][q.name]["optimal_detuning"]
            res.GEF_frequency_shift = new_shift
            res.f_12 = res.RF_frequency + new_shift


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Save all node results and state updates."""
    node.save()


# %%
