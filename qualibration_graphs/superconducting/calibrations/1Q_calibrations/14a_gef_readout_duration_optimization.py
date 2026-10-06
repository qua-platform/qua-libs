# %% {Imports}
import os
from dataclasses import asdict, replace

import matplotlib.pyplot as plt
import xarray as xr
from qm.qua import align, for_, program, save, stream_processing
from qualang_tools.multi_user import qm_session
from qualang_tools.results import progress_counter
from qualang_tools.units import unit
from qualibrate import QualibrationNode
from qualibration_libs.data import XarrayDataFetcher
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam.components.pulses import SquareReadoutPulse

from calibration_utils.readout_gef_duration_optimization import (
    Parameters,
    duration_values,
    fit_raw_data,
    log_fitted_results,
    plot_results,
    process_raw_dataset,
)
from quam_config import Quam

# %% {Description}
description = """GEF READOUT DURATION OPTIMIZATION
Sweep readout_GEF duration while acquiring individual ground-, excited-, and
second-excited-state IQ shots at the configured GEF frequency shift. Fit a
three-component GMM. First require GE assignment to exceed an absolute floor
and remain within a configurable tolerance of that qubit's best GE value. Then
maximize EF assignment only within that GE-qualified plateau. Equal EF
fidelities favor the shortest duration.

Prerequisites:
    - Calibrated x180 and EF_x180 operations.
    - Calibrated GEF readout frequency shift.
    - Square readout pulse with default rectangular integration weights.

State updates:
    - readout_GEF.length.
    - resonator.gef_centers in raw demodulation units.

If readout_GEF is absent, readout supplies its template and a new operation is
created only after successful analysis. The three-state confusion matrix is
stored in the node results.

Times are in ns and must be multiples of 4. This sweeps actual pulse length, not accumulated SNR.
Simulation and historical-data analysis do not update the machine state.
"""

node = QualibrationNode[Parameters, Quam](
    name="14a_gef_readout_duration_optimization",
    description=description,
    parameters=Parameters(),
    machine=Quam.load(),
)


# %% {Custom_parameters}
@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    """Set local debugging parameters; GUI and graph executions ignore these."""
    # Coarse, one-qubit-at-a-time QEC-chain scan. IQCC Cloud currently bounds
    # result-stream collection more tightly than the outer request timeout, so
    # putting all five sequential batches in one QUA job can finish on hardware
    # but lose its result streams. QEC_GEF_QUBIT selects one chain qubit without
    # changing this source between runs.
    selected_qubit = os.getenv("QEC_GEF_QUBIT", "qA2")
    allowed_qubits = {"qA2", "qA1", "qA4", "qA5", "qD5", "qD4", "qD2"}
    if selected_qubit not in allowed_qubits:
        raise ValueError(f"QEC_GEF_QUBIT must be one of {sorted(allowed_qubits)}, got {selected_qubit!r}")
    node.parameters.qubits = [selected_qubit]
    node.parameters.num_shots = int(os.getenv("QEC_GEF_SHOTS", "500"))
    node.parameters.multiplexed = False
    node.parameters.min_duration_in_ns = int(os.getenv("QEC_GEF_MIN_NS", "200"))
    node.parameters.max_duration_in_ns = int(os.getenv("QEC_GEF_MAX_NS", "2000"))
    node.parameters.duration_step_in_ns = int(os.getenv("QEC_GEF_STEP_NS", "200"))
    node.parameters.minimum_ge_fidelity = 0.90
    node.parameters.ge_fidelity_tolerance = float(os.getenv("QEC_GEF_GE_TOLERANCE", "0.01"))


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Create temporary duration pulses and compile the shot-resolved program."""
    if node.parameters.reset_type != "thermal":
        raise ValueError("Only thermal reset is supported")
    explicit_durations = os.getenv("QEC_GEF_DURATIONS_NS")
    durations = (
        [int(value) for value in explicit_durations.split(",")]
        if explicit_durations
        else duration_values(node.parameters)
    )
    if any(duration < 16 or duration % 4 for duration in durations):
        raise ValueError("Every explicit GEF duration must be >=16 ns and divisible by 4 ns")
    n_runs = node.parameters.num_shots
    node.namespace["qubits"] = qubits = get_qubits(node)
    if not len(qubits):
        raise ValueError("Select at least one qubit")
    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names()),
        "n_runs": xr.DataArray(range(1, node.parameters.num_shots + 1), attrs={"long_name": "number of shots"}),
        "duration": xr.DataArray(durations, attrs={"long_name": "readout duration", "units": "ns"}),
    }
    u = unit(coerce_to_integer=True)
    temporary_operations = []
    operation_names = {}
    try:
        for qubit in qubits:
            if "EF_x180" not in qubit.xy.operations:
                raise ValueError(f"{qubit.name}: calibrate EF_x180 before running this node")
            for duration in durations:
                name = f"_duration_opt_{int(duration)}"
                if name in qubit.resonator.operations:
                    raise ValueError(f"{qubit.name}: temporary operation {name} already exists")
                source = qubit.resonator.operations.get(
                    node.parameters.operation, qubit.resonator.operations["readout"]
                )
                if type(source) is not SquareReadoutPulse:
                    raise ValueError(f"{qubit.name}: duration optimization requires a SquareReadoutPulse")
                if source.get_raw_value("integration_weights") != "#./default_integration_weights":
                    raise ValueError(
                        f"{qubit.name}: recalibrate with default integration weights before sweeping duration"
                    )
                qubit.resonator.operations[name] = replace(
                    source,
                    id=None,
                    length=int(duration),
                    integration_weights="#./default_integration_weights",
                    threshold=None,
                    rus_exit_threshold=None,
                )
                temporary_operations.append((qubit.resonator, name))
                operation_names[int(duration)] = name

        with program() as node.namespace["qua_program"]:
            I_g, I_g_st, Q_g, Q_g_st, n, n_st = node.machine.declare_qua_variables()
            I_e, I_e_st, Q_e, Q_e_st, _, _ = node.machine.declare_qua_variables()
            I_f, I_f_st, Q_f, Q_f_st, _, _ = node.machine.declare_qua_variables()

            for multiplexed_qubits in qubits.batch():
                # Initialize flux points for this batch.
                for qubit in multiplexed_qubits.values():
                    node.machine.initialize_qpu(target=qubit)
                align()
                for qubit in multiplexed_qubits.values():
                    shift = qubit.resonator.GEF_frequency_shift or 0
                    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency + shift)

                with for_(n, 0, n < n_runs, n + 1):
                    save(n, n_st)
                    # Each duration has its own matching integration weights.
                    for duration in durations:
                        operation = operation_names[int(duration)]

                        # Prepare and measure the |g> state.
                        for qubit in multiplexed_qubits.values():
                            qubit.wait(2 * qubit.thermalization_time * u.ns)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.resonator.measure(operation, qua_vars=(I_g[i], Q_g[i]))
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                            save(I_g[i], I_g_st[i])
                            save(Q_g[i], Q_g_st[i])
                        align()

                        # Prepare and measure the |e> state.
                        for qubit in multiplexed_qubits.values():
                            qubit.wait(2 * qubit.thermalization_time * u.ns)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                            qubit.align()
                            qubit.resonator.measure(operation, qua_vars=(I_e[i], Q_e[i]))
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                            save(I_e[i], I_e_st[i])
                            save(Q_e[i], Q_e_st[i])
                        align()

                        # Prepare and measure the |f> state.
                        for qubit in multiplexed_qubits.values():
                            qubit.wait(2 * qubit.thermalization_time * u.ns)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            qubit.xy.play("x180")
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency - qubit.anharmonicity)
                            qubit.xy.play("EF_x180")
                            qubit.xy.update_frequency(qubit.xy.intermediate_frequency)
                            qubit.align()
                            qubit.resonator.measure(operation, qua_vars=(I_f[i], Q_f[i]))
                            qubit.resonator.wait(qubit.resonator.depletion_time * u.ns)
                            save(I_f[i], I_f_st[i])
                            save(Q_f[i], Q_f_st[i])
                        align()
                for qubit in multiplexed_qubits.values():
                    qubit.resonator.update_frequency(qubit.resonator.intermediate_frequency)
                align()

            with stream_processing():
                n_st.save("n")
                for i in range(len(qubits)):
                    I_g_st[i].buffer(len(durations)).buffer(n_runs).save(f"Ig{i + 1}")
                    Q_g_st[i].buffer(len(durations)).buffer(n_runs).save(f"Qg{i + 1}")
                    I_e_st[i].buffer(len(durations)).buffer(n_runs).save(f"Ie{i + 1}")
                    Q_e_st[i].buffer(len(durations)).buffer(n_runs).save(f"Qe{i + 1}")
                    I_f_st[i].buffer(len(durations)).buffer(n_runs).save(f"If{i + 1}")
                    Q_f_st[i].buffer(len(durations)).buffer(n_runs).save(f"Qf{i + 1}")
        node.namespace["config"] = node.machine.generate_config()
    finally:
        for resonator, name in temporary_operations:
            del resonator.operations[name]


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Simulate the generated program and store samples and waveform report."""
    samples, fig, report = simulate_and_plot(
        node.machine.connect(),
        node.namespace["config"],
        node.namespace["qua_program"],
        node.parameters,
    )
    node.results["simulation"] = {"figure": fig, "wf_report": report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Execute the duration sweep and fetch every individual IQ shot."""
    qmm = node.machine.connect()
    with qm_session(qmm, node.namespace["config"], timeout=node.parameters.timeout) as qm:
        node.namespace["job"] = job = qm.execute(node.namespace["qua_program"], options={"timeout": 300})
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
    """Load a previously saved node run for offline analysis."""
    load_data_id = node.parameters.load_data_id
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Select the best valid duration and extract GEF classification centers."""
    node.results["ds_raw"] = process_raw_dataset(node.results["ds_raw"], node)
    node.results["ds_fit"], node.results["ds_iq_blobs"], results = fit_raw_data(node.results["ds_raw"], node)
    node.results["fit_results"] = {name: asdict(result) for name, result in results.items()}
    log_fitted_results(node.results["fit_results"], log_callable=node.log)
    node.outcomes = {name: "successful" if result.success else "failed" for name, result in results.items()}


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot duration metrics, selected IQ blobs, and confusion matrices."""
    node.results["figures"] = plot_results(
        node.results["ds_raw"],
        node.namespace["qubits"],
        node.results["ds_fit"],
        node.results["ds_iq_blobs"],
    )


# %% {Update_state}
@node.run_action(
    skip_if=(
        node.parameters.simulate
        or node.parameters.load_data_id is not None
        or os.getenv("QEC_GEF_DURATION_CHECK_ONLY", "0") == "1"
    )
)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Apply successful per-qubit fits while leaving failed qubits unchanged."""
    with node.record_state_updates():
        for qubit in node.namespace["qubits"]:
            result = node.results["fit_results"][qubit.name]
            if not result["success"]:
                continue
            duration = result["optimal_duration"]
            operation_name = "readout_GEF"
            if operation_name not in qubit.resonator.operations:
                source = qubit.resonator.operations.get(operation_name, qubit.resonator.operations["readout"])
                if type(source) is not SquareReadoutPulse:
                    raise ValueError(f"{qubit.name}: duration optimization requires a SquareReadoutPulse")
                if source.get_raw_value("integration_weights") != "#./default_integration_weights":
                    raise ValueError(
                        f"{qubit.name}: recalibrate with default integration weights before sweeping duration"
                    )
                qubit.resonator.operations[operation_name] = replace(
                    source,
                    id=None,
                    length=int(duration),
                    integration_weights="#./default_integration_weights",
                    threshold=None,
                    rus_exit_threshold=None,
                )
            operation = qubit.resonator.operations[operation_name]
            operation.length = duration
            qubit.resonator.gef_centers = (
                node.results["ds_iq_blobs"].sel(qubit=qubit.name).centers.values * duration / 2**12
            ).tolist()


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    """Persist node parameters, datasets, fit results, figures, and updates."""
    node.save()
