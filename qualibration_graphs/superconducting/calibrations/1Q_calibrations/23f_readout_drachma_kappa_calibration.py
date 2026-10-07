# %% {Imports}
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from qm.qua import (
    align,
    declare,
    declare_output_stream,
    fixed,
    for_,
    program,
    reset_if_phase,
    save,
    stream_processing,
)
from qualang_tools.multi_user import qm_session
from qualang_tools.results import fetching_tool, progress_counter
from qualibration_libs.core import BatchableList
from qualibration_libs.parameters import get_qubits
from qualibration_libs.runtime import simulate_and_plot
from quam.components.pulses import SquareReadoutPulse
from quam_config import Quam

from calibration_utils.readout_drachma_common import (
    NOOP_SUFFIX,
    STATES,
    amplitude_scale_to_fit,
    assign_core_labels,
    build_batch_groups,
    fetch_round_traces,
    plot_noop_pvalue_vs_point,
    process_raw_dataset,
    scale_pulse_kwargs,
    waveform_peak,
)
from calibration_utils.readout_drachma_kappa_calibration import (
    Parameters,
    build_kappa_grid,
    fit_raw_data,
    log_fitted_results,
    plot_power_vs_kappa,
)
from qualibrate import QualibrationNode

# %% {Description}
description = """
        RESONATOR DRACHMA READOUT - FINE CALIBRATION OF KAPPA_GROUND / KAPPA_EXCITED
Fine-tunes kappa_ground_hz and kappa_excited_hz of the DRACHMA readout pulse (Jerger et al.,
arXiv:2406.04891) by minimising the photon population left in the resonator right after the pulse.
Same measurement as 23e_readout_drachma_zeta_calibration, but the scanned knob is kappa. The two
kappas are chosen jointly: |kappa_ground - kappa_excited| must stay within max_kappa_difference_hz,
and among the allowed pairs the one with the lowest noise-weighted excess residual power is taken.
Every shot also measures a single no-operation reference, and a z-test of each (state, point) against it is
logged and plotted as an extra depletion check.

Details (scan grid, joint pair selection, amplitude limit): calibration_utils/readout_drachma_kappa_calibration/README.md

Prerequisites:
    - Having calibrated the readout parameters (nodes 02a, 02b) and the qubit x180 pulse (nodes 03a, 04b).
    - A "readout_drachma" operation (DrachmaReadoutPulse) with kappa, chi and amplitude set, on
      qubit.resonator.operations. Ideally with zeta tuned by 23e_readout_drachma_zeta_calibration.

State update:
    - The DRACHMA readout pulse's kappa_ground_hz and kappa_excited_hz (jointly chosen power minima).
"""


# %% {Initialisation}
node = QualibrationNode[Parameters, Quam](
    name="23f_readout_drachma_kappa_calibration",
    description=description,
    parameters=Parameters(),
)


@node.run_action(skip_if=node.modes.external)
def custom_param(node: QualibrationNode[Parameters, Quam]):
    # node.parameters.qubits = ["qC2", "qC3", "qD1"]
    # node.parameters.load_data_id = 255
    pass


node.machine = Quam.load()

PROBE_OPERATION = "readout_probe"
"""Name of the temporary zero-amplitude SquareReadoutPulse (length probe_length) built per qubit in
create_qua_program and removed again in save_results before node.save() persists machine state."""

DEFAULT_INTEGRATION_WEIGHTS = "#./default_integration_weights"
"""QuAM reference to the pulse's default integration weights (what readout_square uses)."""

KAPPA_FIELD = {"ground": "kappa_ground_hz", "excited": "kappa_excited_hz"}


def remove_temporary_operations(node: QualibrationNode[Parameters, Quam]):
    """Remove the probe pulse and every temporary kappa pulse added in create_qua_program.

    Pops node.namespace["kappa_ops"] so a second call (e.g. from save_results after execute_qua_program
    already cleaned up) is a no-op.
    """
    kappa_ops = node.namespace.pop("kappa_ops", {})
    original_pulses = node.namespace.pop("original_drachma_pulses", {})
    for qubit in node.namespace.get("qubits", []):
        operations = qubit.resonator.operations
        operations.pop(PROBE_OPERATION, None)
        if qubit.name in original_pulses:
            operations[node.parameters.drachma_operation] = original_pulses[qubit.name]
        for names in kappa_ops.get(qubit.name, {}).values():
            for name in names:
                operations.pop(name, None)


# %% {Create_QUA_program}
@node.run_action(skip_if=node.parameters.load_data_id is not None)
def create_qua_program(node: QualibrationNode[Parameters, Quam]):
    """Build the temporary kappa pulses and the probe, then for each (state, point) play the pulse
    and stream the integrated I/Q of the zero-amplitude probe that immediately follows it."""
    all_qubits = get_qubits(node)
    drachma_operation = node.parameters.drachma_operation
    probe_length = node.parameters.probe_length
    num_points = node.parameters.kappa_num_points
    # Only enforced by the QM compiler at execution time -- check here instead of failing on
    # hardware with a per-port error list.
    if probe_length <= 0 or probe_length % 4 != 0:
        raise ValueError(f"probe_length={probe_length} must be a positive multiple of 4.")
    depletion_time = node.parameters.depletion_time
    if depletion_time <= 0 or depletion_time % 4 != 0:
        raise ValueError(f"depletion_time={depletion_time} must be a positive multiple of 4.")
    if num_points < 2:
        raise ValueError(f"kappa_num_points={num_points} must be >= 2.")
    kappa_step_hz = node.parameters.kappa_step_hz
    if kappa_step_hz <= 0:
        raise ValueError(f"kappa_step_hz={kappa_step_hz} must be > 0.")
    max_kappa_difference_hz = node.parameters.max_kappa_difference_hz
    if max_kappa_difference_hz is not None and max_kappa_difference_hz < 0:
        raise ValueError(f"max_kappa_difference_hz={max_kappa_difference_hz} must be >= 0 or None.")
    if not 0 < node.parameters.max_waveform_peak <= 1:
        raise ValueError(f"max_waveform_peak={node.parameters.max_waveform_peak} must be in (0, 1].")

    # Qubits missing the DRACHMA pulse are excluded rather than failing the whole node -- in
    # multiplexed mode a single qubit playing an undefined operation aborts the shared program.
    kept_qubits = [q for q in all_qubits if q.resonator.operations.get(drachma_operation)]
    skipped_names = [q.name for q in all_qubits if q not in kept_qubits]
    if skipped_names:
        message = f"Skipping qubits without a pre-built '{drachma_operation}' operation: {skipped_names}."
        node.log(message)
        print(message)
    if not kept_qubits:
        raise RuntimeError(f"No qubits have a pre-built '{drachma_operation}' operation.")

    # Registered before any operation is added so remove_temporary_operations cleans up even if
    # this action fails half-way.
    node.namespace["qubits"] = kept_qubits
    node.namespace["original_drachma_pulses"] = original_pulses = {}
    node.namespace["kappa_ops"] = kappa_ops = {q.name: {s: [] for s in STATES} for q in kept_qubits}
    kappa_grids = {q.name: {} for q in kept_qubits}
    for q in kept_qubits:
        operations = q.resonator.operations
        operations[PROBE_OPERATION] = SquareReadoutPulse(length=probe_length, amplitude=0.0)
        drachma = operations[drachma_operation]
        original_kwargs = {k: v for k, v in drachma.to_dict().items() if k != "__class__"}
        pulse_kwargs = dict(original_kwargs)
        # The DRACHMA result is discarded, so the copies use the default integration weights (as
        # readout_square does) instead of duplicating the pulse's long explicit weights, which would
        # exhaust the data memory.
        pulse_kwargs["integration_weights"] = DEFAULT_INTEGRATION_WEIGHTS
        point_kwargs = {}
        for state in STATES:
            current_kappa = getattr(drachma, KAPPA_FIELD[state])
            if current_kappa <= 0:
                raise ValueError(f"{q.name}: {KAPPA_FIELD[state]}={current_kappa} must be positive to scan around it.")
            grid = build_kappa_grid(current_kappa, kappa_step_hz, num_points)
            kappa_grids[q.name][state] = grid
            for i, kappa in enumerate(grid):
                # Copy of the DRACHMA pulse with only this state's kappa changed.
                point_kwargs[f"{drachma_operation}_kappa_{state}_{i}"] = {
                    **pulse_kwargs,
                    KAPPA_FIELD[state]: float(kappa),
                }
        # The waveform is normalised to a fixed area, so a larger kappa makes it peakier and can push
        # samples past full scale (the QM config rejects |sample| > 1). One common factor for the whole
        # qubit keeps the points comparable with each other.
        factor = amplitude_scale_to_fit(
            type(drachma), [original_kwargs, *point_kwargs.values()], node.parameters.max_waveform_peak
        )
        if factor < 1:
            message = (
                f"{q.name}: scaling the amplitude of all kappa-scan pulses by {factor:.3f} "
                f"({drachma.amplitude:.4g} -> {drachma.amplitude * factor:.4g}) to keep the waveform peak "
                f"<= {node.parameters.max_waveform_peak}."
            )
            node.log(message)
            print(message)
        if factor < 1:
            # The stored pulse stays in the config next to the copies and can itself exceed full scale
            # (the QM config validates every operation), so it is swapped for a scaled copy until
            # remove_temporary_operations puts the original back.
            original_pulses[q.name] = drachma
            operations[drachma_operation] = type(drachma)(**scale_pulse_kwargs(original_kwargs, factor))
        for name, kwargs in point_kwargs.items():
            operations[name] = type(drachma)(**scale_pulse_kwargs(kwargs, factor))
            state = "ground" if "_kappa_ground_" in name else "excited"
            kappa_ops[q.name][state].append(name)

    batch_groups = build_batch_groups(kept_qubits, node.parameters)
    node.log(
        f"Batching: {len(kept_qubits)} qubits into {len(batch_groups)} batch(es) of sizes "
        f"{[len(b) for b in batch_groups]} (multiplexed={node.parameters.multiplexed})."
    )

    # Share cores wherever safe so config generation doesn't allocate a dedicated core per
    # qubit per FEM; each TWPA pump runs continuously, so it keeps its own reserved core.
    twpas = list(node.machine.twpas.values())
    core_labels, twpa_labels = assign_core_labels(kept_qubits, twpas)
    for qubit, label in zip(kept_qubits, core_labels):
        qubit.xy.core = qubit.resonator.core = label
    for twpa in twpas:
        label = twpa_labels[twpa.name]
        twpa.pump.core = label
        if twpa.pump_ is not None:
            twpa.pump_.core = label
    node.log(f"Core assignment: {dict(zip((q.name for q in kept_qubits), core_labels))}")
    node.log(f"TWPA core assignment: {twpa_labels}")
    node.namespace["qubits"] = qubits = BatchableList(kept_qubits, batch_groups)
    num_qubits = len(qubits)

    node.namespace["sweep_axes"] = {
        "qubit": xr.DataArray(qubits.get_names(), dims="qubit"),
        "point": xr.DataArray(np.arange(num_points), dims="point", attrs={"long_name": "kappa scan point"}),
    }
    # Per-qubit kappa of every point, stored as coords of the raw dataset.
    node.namespace["kappa_coords"] = {
        KAPPA_FIELD[state]: (
            ("qubit", "point"),
            np.stack([kappa_grids[name][state] for name in qubits.get_names()]),
            {"long_name": f"{state} kappa", "units": "Hz"},
        )
        for state in STATES
    }

    with program() as node.namespace["qua_program"]:
        n = declare(int)
        n_st = declare_output_stream()
        # The DRACHMA readout's result is never read -- declared once and reused.
        I_dummy = declare(fixed)
        Q_dummy = declare(fixed)
        I = [declare(fixed) for _ in range(num_qubits)]
        Q = [declare(fixed) for _ in range(num_qubits)]
        I_st = {s: [declare_output_stream() for _ in range(num_qubits)] for s in STATES}
        Q_st = {s: [declare_output_stream() for _ in range(num_qubits)] for s in STATES}
        I_noop_st = [declare_output_stream() for _ in range(num_qubits)]
        Q_noop_st = [declare_output_stream() for _ in range(num_qubits)]

        for multiplexed_qubits in qubits.batch():
            for qubit in multiplexed_qubits.values():
                node.machine.initialize_qpu(target=qubit)
            align()

            with for_(n, 0, n < node.parameters.num_shots, n + 1):
                save(n, n_st)
                # Single no-operation reference: the probe right after reset, no DRACHMA pulse.
                for i, qubit in multiplexed_qubits.items():
                    qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                align()
                for i, qubit in multiplexed_qubits.items():
                    rr = qubit.resonator
                    reset_if_phase(rr.name)
                    rr.measure(PROBE_OPERATION, qua_vars=(I[i], Q[i]))
                    save(I[i], I_noop_st[i])
                    save(Q[i], Q_noop_st[i])
                    rr.wait(depletion_time // 4)
                align()
                for state in STATES:
                    # num_points is a plain Python int, so this loop unrolls at compile time.
                    for point in range(num_points):
                        for i, qubit in multiplexed_qubits.items():
                            qubit.reset(node.parameters.reset_type, node.parameters.simulate)
                        align()
                        for i, qubit in multiplexed_qubits.items():
                            rr = qubit.resonator
                            # |e> is prepared right before the pulse so T1 decay cannot be mistaken
                            # for residual photons.
                            if state == "excited":
                                qubit.xy.play("x180")
                                qubit.align()
                            reset_if_phase(rr.name)
                            rr.measure(kappa_ops[qubit.name][state][point], qua_vars=(I_dummy, Q_dummy))
                            reset_if_phase(rr.name)
                            rr.measure(PROBE_OPERATION, qua_vars=(I[i], Q[i]))
                            save(I[i], I_st[state][i])
                            save(Q[i], Q_st[state][i])
                            rr.wait(depletion_time // 4)
                        align()

        with stream_processing():
            n_st.save("n")
            for i in range(num_qubits):
                I_noop_st[i].average().save(f"I_{NOOP_SUFFIX}{i + 1}")
                Q_noop_st[i].average().save(f"Q_{NOOP_SUFFIX}{i + 1}")
                (I_noop_st[i] * I_noop_st[i]).average().save(f"I_sq_{NOOP_SUFFIX}{i + 1}")
                (Q_noop_st[i] * Q_noop_st[i]).average().save(f"Q_sq_{NOOP_SUFFIX}{i + 1}")
            for state in STATES:
                for i in range(num_qubits):
                    I_st[state][i].buffer(num_points).average().save(f"I_{state}{i + 1}")
                    Q_st[state][i].buffer(num_points).average().save(f"Q_{state}{i + 1}")
                    # Per-shot second moments -> Var(I), Var(Q) for the error bars and the z-test.
                    (I_st[state][i] * I_st[state][i]).buffer(num_points).average().save(f"I_sq_{state}{i + 1}")
                    (Q_st[state][i] * Q_st[state][i]).buffer(num_points).average().save(f"Q_sq_{state}{i + 1}")


# %% {Simulate}
@node.run_action(skip_if=node.parameters.load_data_id is not None or not node.parameters.simulate)
def simulate_qua_program(node: QualibrationNode[Parameters, Quam]):
    qmm = node.machine.connect()
    config = node.machine.generate_config()
    samples, fig, wf_report = simulate_and_plot(qmm, config, node.namespace["qua_program"], node.parameters)
    node.results["simulation"] = {"figure": fig, "wf_report": wf_report, "samples": samples}


# %% {Execute}
@node.run_action(skip_if=node.parameters.load_data_id is not None or node.parameters.simulate)
def execute_qua_program(node: QualibrationNode[Parameters, Quam]):
    try:
        qmm = node.machine.connect()
        config = node.machine.generate_config()
        with qm_session(qmm, config, timeout=node.parameters.timeout) as qm:
            node.namespace["job"] = job = qm.execute(
                node.namespace["qua_program"], terminal_output=True, options={"timeout": node.parameters.timeout}
            )
            # Logged before fetching_tool: on a runtime error the "n" stream can come back empty and
            # fetching_tool raises before the report would otherwise be reached.
            node.log(job.execution_report())
            results = fetching_tool(job, ["n"], mode="live")
            while results.is_processing():
                progress_counter(results.fetch_all()[0], node.parameters.num_shots)

            raw_traces = fetch_round_traces(job, node.namespace["qubits"], STATES)
        coords = dict(node.namespace["sweep_axes"])
        coords.update(node.namespace["kappa_coords"])
        node.results["ds_raw"] = xr.Dataset(
            {
                name: (("qubit",) if name.endswith(f"_{NOOP_SUFFIX}") else ("qubit", "point"), arr)
                for name, arr in raw_traces.items()
            },
            coords=coords,
        )
    finally:
        # The config is already generated, so the temporary pulses are no longer needed; node.save()
        # in save_results would otherwise persist them.
        remove_temporary_operations(node)


# %% {Load_data}
@node.run_action(skip_if=node.parameters.load_data_id is None)
def load_data(node: QualibrationNode[Parameters, Quam]):
    load_data_id = node.parameters.load_data_id
    # alpha is a post-hoc analysis knob, not an acquisition parameter -- it must survive
    # load_from_id overwriting node.parameters with the historical run's values.
    alpha = node.parameters.alpha
    node.load_from_id(load_data_id)
    node.parameters.load_data_id = load_data_id
    node.parameters.alpha = alpha
    node.namespace["qubits"] = get_qubits(node)


# %% {Analyse_data}
@node.run_action(skip_if=node.parameters.simulate)
def analyse_data(node: QualibrationNode[Parameters, Quam]):
    """Convert the probe I/Q to volts, compute the residual power vs kappa and the test-vs-no-operation z-test, and
    pick the power minimum per qubit and state."""
    ds = process_raw_dataset(node.results["ds_raw"], node.parameters.probe_length, node.parameters.num_shots)
    node.results["ds_raw"] = ds
    node.results["ds_fit"], fit_results = fit_raw_data(ds, node)
    node.results["fit_results"] = {k: asdict(v) for k, v in fit_results.items()}

    log_fitted_results(node.results["fit_results"], node.parameters.alpha, log_callable=node.log)
    node.outcomes = {
        qubit_name: ("successful" if fit_result["success"] else "failed")
        for qubit_name, fit_result in node.results["fit_results"].items()
    }


# %% {Plot_data}
@node.run_action(skip_if=node.parameters.simulate)
def plot_data(node: QualibrationNode[Parameters, Quam]):
    """Plot the residual probe power vs kappa and the test-vs-no-operation z-test p-value, one subplot per qubit."""
    fig_power = plot_power_vs_kappa(node.results["ds_fit"], node.namespace["qubits"], node.results["fit_results"])
    fig_pvalue = plot_noop_pvalue_vs_point(node.results["ds_fit"], node.namespace["qubits"], node.parameters.alpha)
    node.results["figures"] = {"power_vs_kappa": fig_power, "noop_pvalue": fig_pvalue}
    plt.show()


# %% {Update_state}
@node.run_action(skip_if=node.parameters.simulate)
def update_state(node: QualibrationNode[Parameters, Quam]):
    """Write the jointly chosen kappa_ground_hz / kappa_excited_hz to the DRACHMA pulse."""
    with node.record_state_updates():
        for q in node.namespace["qubits"]:
            if node.outcomes[q.name] == "failed":
                continue
            fit = node.results["fit_results"][q.name]
            drachma = q.resonator.operations[node.parameters.drachma_operation]
            updates = {"kappa_ground_hz": fit["kappa_ground_hz"], "kappa_excited_hz": fit["kappa_excited_hz"]}
            # The scan ran on scaled pulses, but the stored pulse plays at its own amplitude: never persist
            # values that push it past full scale (the QM config would reject it in every later node).
            pulse_kwargs = {k: v for k, v in drachma.to_dict().items() if k != "__class__"}
            peak = waveform_peak(type(drachma), {**pulse_kwargs, **updates})
            if peak > 1:
                message = (
                    f"{q.name}: NOT updating kappa_ground_hz / kappa_excited_hz: the pulse would peak at {peak:.4f} > 1 at its "
                    f"stored amplitude {drachma.amplitude:.4g}. Lower the {node.parameters.drachma_operation} "
                    "amplitude and rerun."
                )
                node.log(message)
                print(message)
                continue
            for field, value in updates.items():
                setattr(drachma, field, value)


# %% {Save_results}
@node.run_action()
def save_results(node: QualibrationNode[Parameters, Quam]):
    # Core assignment is scoped to this run's qubits, and the probe / kappa pulses are temporary --
    # clear them before node.save() persists node.machine so none leaks into shared QuAM state.
    remove_temporary_operations(node)
    for qubit in node.namespace["qubits"]:
        qubit.xy.core = None
        qubit.resonator.core = None
    for twpa in node.machine.twpas.values():
        twpa.pump.core = None
        if twpa.pump_ is not None:
            twpa.pump_.core = None
    node.save()


# %%
