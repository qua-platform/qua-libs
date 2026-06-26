import logging
import time
from datetime import datetime
import json
from pathlib import Path
from typing import Any, List, Literal, Optional

from qualibrate.core import QualibrationNode
from qualibrate.core.parameters import RunnableParameters
from qualang_tools.results import progress_counter as _progress_counter
from qualibration_libs.core import BatchableList

from quam_builder.architecture.quantum_dots.components import SensorDot, QuantumDot
from quam_builder.architecture.quantum_dots.operations.names import SingleQubitMacroName
from quam_builder.architecture.quantum_dots.qpu import BaseQuamQD
from quam_builder.architecture.quantum_dots.qubit import AnySpinQubit
from quam_builder.architecture.quantum_dots.qubit_pair import AnySpinQubitPair

_FETCHER_AXIS_MESSAGE = "first axis must be either qubit or qubit_pair"
_fetcher_axis_filter_installed = False
_RUNTIME_LOG_STATE: dict[int, dict[str, Any]] = {}


def suppress_fetcher_axis_log_spam() -> None:
    """Suppress the known non-fatal fetcher axis message."""
    global _fetcher_axis_filter_installed
    if _fetcher_axis_filter_installed:
        return

    fetcher_logger = logging.getLogger("qualibration_libs.data.fetcher")
    fetcher_logger.addFilter(
        lambda record: record.getMessage() != _FETCHER_AXIS_MESSAGE
    )
    _fetcher_axis_filter_installed = True


def _patch_node_save_for_runtime_logging(node: QualibrationNode) -> None:
    node_key = id(node)
    state = _RUNTIME_LOG_STATE.get(node_key)
    if state is None or state.get("save_patched"):
        return

    original_save = node.save

    def wrapped_save(*args: Any, **kwargs: Any) -> Any:
        result = original_save(*args, **kwargs)
        runtime_state = _RUNTIME_LOG_STATE.get(node_key)
        if runtime_state is None:
            return result

        save_runtime_summary_json(
            node,
            total_averages=runtime_state.get("total_averages", 0),
            qua_runtime_seconds=runtime_state.get("qua_runtime_seconds"),
            progress_log=runtime_state.get("progress_log", []),
            extra_fields=runtime_state.get("extra_fields"),
        )
        _RUNTIME_LOG_STATE.pop(node_key, None)
        return result

    node.save = wrapped_save
    state["save_patched"] = True


def progress_counter_with_log(
    iteration: int,
    total: int,
    progress_bar: bool = True,
    percent: bool = True,
    start_time: float | None = None,
    *,
    node: Optional[QualibrationNode] = None,
    total_averages: Optional[int] = None,
    extra_fields: Optional[dict[str, Any]] = None,
) -> None:
    """Drop-in replacement for qualang-tools progress_counter with runtime logging.

    Call this exactly where ``progress_counter`` was used. When ``node`` is provided,
    the function automatically records progress snapshots and writes
    ``runtime_summary.json`` when ``node.save()`` is called.
    """
    _progress_counter(
        iteration,
        total,
        progress_bar=progress_bar,
        percent=percent,
        start_time=start_time,
    )

    if node is None:
        node = QualibrationNode.active_node
    if node is None:
        return

    node_key = id(node)
    state = _RUNTIME_LOG_STATE.setdefault(
        node_key,
        {
            "progress_log": [],
            "total_averages": int(total_averages if total_averages is not None else total),
            "qua_runtime_seconds": None,
            "extra_fields": extra_fields,
            "save_patched": False,
        },
    )

    if total_averages is not None:
        state["total_averages"] = int(total_averages)
    if extra_fields is not None:
        state["extra_fields"] = extra_fields

    elapsed_seconds = None if start_time is None else max(0.0, time.time() - start_time)
    iteration_int = int(iteration)
    total_int = int(total)
    completed = max(0, iteration_int + 1)

    percent_value = None if total_int <= 0 else max(0.0, min(100.0, 100.0 * completed / total_int))
    state["progress_log"].append(
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="milliseconds"),
            "iteration": int(iteration),
            "completed": completed,
            "total": total_int,
            "percent": percent_value,
            "elapsed_seconds": elapsed_seconds,
        }
    )

    # if elapsed_seconds is not None:
    #     state["qua_runtime_seconds"] = elapsed_seconds
    #     try:
    #         if elapsed_seconds > 0 and total_int > 0 and completed > 0:
    #             gradient_n_per_second = completed / elapsed_seconds
    #             projected_total_minutes = total_int / gradient_n_per_second / 60.0
    #             projected_remaining_minutes = max(0, total_int - completed) / gradient_n_per_second / 60.0
    #             print(
    #                 "\rProjected measurement length: "
    #                 f"{projected_total_minutes:.2f} min total, "
    #                 f"{projected_remaining_minutes:.2f} min remaining "
    #                 f"({gradient_n_per_second:.2f} N/s)",
    #                 end="",
    #                 flush=True,
    #             )
    #             if completed >= total_int:
    #                 print()
    #     except Exception:
    #         pass

    _patch_node_save_for_runtime_logging(node)


def save_runtime_summary_json(
    node: QualibrationNode,
    *,
    total_averages: int,
    qua_runtime_seconds: float | None,
    progress_log: list[dict[str, Any]],
    extra_fields: Optional[dict[str, Any]] = None,
    filename: str = "runtime_summary.json",
) -> Optional[Path]:
    """Save runtime diagnostics as a JSON file next to ``node.json``.

    The function reads ``run_start`` and ``run_end`` from the saved node metadata
    and stores those together with QUA-runtime/progress information.
    """
    storage_manager = getattr(node, "storage_manager", None)
    data_handler = getattr(storage_manager, "data_handler", None)
    data_path = getattr(data_handler, "path", None)
    if data_path is None or isinstance(data_path, int):
        node.log("Runtime summary not saved: node storage path unavailable.", level="warning")
        return None

    run_dir = Path(data_path)
    node_json_path = run_dir / "node.json"
    if not node_json_path.exists():
        node.log(
            f"Runtime summary not saved: missing node metadata file at {node_json_path}.",
            level="warning",
        )
        return None

    node_payload = json.loads(node_json_path.read_text())
    metadata = node_payload.get("metadata", {})
    run_start_str = metadata.get("run_start")
    run_end_str = metadata.get("run_end")

    total_runtime_seconds = None
    if isinstance(run_start_str, str) and isinstance(run_end_str, str):
        try:
            run_start = datetime.fromisoformat(run_start_str)
            run_end = datetime.fromisoformat(run_end_str)
            total_runtime_seconds = max(0.0, (run_end - run_start).total_seconds())
        except ValueError:
            node.log("Failed to parse run_start/run_end from node metadata.", level="warning")

    opx_overhead_seconds = None
    if (
        total_runtime_seconds is not None
        and qua_runtime_seconds is not None
    ):
        opx_overhead_seconds = total_runtime_seconds - qua_runtime_seconds

    payload: dict[str, Any] = {
        "node_name": node.name,
        "total_averages": int(total_averages),
        "qua_runtime_seconds": qua_runtime_seconds,
        "total_runtime_seconds": total_runtime_seconds,
        "opx_overhead_seconds": opx_overhead_seconds,
        "run_start": run_start_str,
        "run_end": run_end_str,
        "progress_log_entries": len(progress_log),
        "progress_log": progress_log,
    }

    if extra_fields:
        payload.update(extra_fields)

    output_path = run_dir / filename
    output_path.write_text(json.dumps(payload, indent=2))
    node.log(f"Saved runtime summary to {output_path}")
    return output_path


class BaseExperimentNodeParameters(RunnableParameters):
    multiplexed: bool = False
    """Whether to play control pulses, readout pulses and active/thermal reset at the same time for all qubits (True)
    or to play the experiment sequentially for each qubit (False). Default is False."""
    use_state_discrimination: bool = False
    """Whether to use on-the-fly state discrimination and return the qubit 'state', or simply return the demodulated
    quadratures 'I' and 'Q'. Default is False."""
    reset_wait_time: int = 5000
    """The wait time for qubit reset."""

class QuantumDotExperimentNodeParameters(BaseExperimentNodeParameters):
    quantum_dots: Optional[List[str]] = None
    """The virtualised names of the QuantumDots in your VirtualGateSet."""


class QubitsExperimentNodeParameters(BaseExperimentNodeParameters):
    qubits: Optional[List[str]] = None
    """A list of qubit names which should participate in the execution of the node. Default is None."""


class QubitPairExperimentNodeParameters(BaseExperimentNodeParameters):
    qubit_pairs: Optional[List[str]] = None
    """A list of qubit pair names which should participate in the execution of the node. Default is None."""


class ParityDiffAnalysisParameters(RunnableParameters):
    analysis_signal: Literal["E_p2_given_p1_0", "E_p2_given_p1_1"] = "E_p2_given_p1_0"
    """Which conditional expectation to use for fitting.
    E_p2_given_p1_0: P(second=1 | first=0) — post-select on empty dot.
    E_p2_given_p1_1: P(second=1 | first=1) — post-select on loaded dot."""
    parity_pre_measurement: bool = False
    """Whether to use parity pre measurement. Default is False."""


class HeraldedInitializeParameters(RunnableParameters): 
    target_state: Optional[int] = None
    """The state you want to initialize into for heralded initialization."""
    max_loops: int = 100
    """Maximum number of initialization loops for heralded initialization."""
    return_n_loops: bool = False
    """Whether to return the number of times it has looped over the initialise sequence to achieve the desired result."""


def _make_batchable_list_from_multiplexed(
    items: List, multiplexed: bool
) -> BatchableList:
    if multiplexed:
        batched_groups = [[i for i in range(len(items))]]
    else:
        batched_groups = [[i] for i in range(len(items))]

    return BatchableList(items, batched_groups)


def _get_dots(machine: BaseQuamQD, node_parameters: QuantumDotExperimentNodeParameters):
    if node_parameters.quantum_dots is None or node_parameters.quantum_dots == "":
        dots = list(machine.quantum_dots.values())
    else:
        dots = [machine.quantum_dots[s] for s in node_parameters.quantum_dots]
    return dots


def get_dots(node: QualibrationNode) -> BatchableList[QuantumDot]:
    dots = _get_dots(node.machine, node.parameters)
    dots_batchable_list = _make_batchable_list_from_multiplexed(dots, True)
    return dots_batchable_list


def _get_sensors(machine: BaseQuamQD, node_parameters: BaseExperimentNodeParameters):
    if node_parameters.sensor_names is None or node_parameters.sensor_names == "":
        sensors = list(machine.sensor_dots.values())
    else:
        sensors = [machine.sensor_dots[s] for s in node_parameters.sensor_names]
    return sensors


def get_sensors(node: QualibrationNode) -> BatchableList[SensorDot]:
    sensors = _get_sensors(node.machine, node.parameters)

    if isinstance(node.parameters, BaseExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    sensors_batchable_list = _make_batchable_list_from_multiplexed(sensors, multiplexed)

    return sensors_batchable_list


def get_qubits(node: QualibrationNode) -> BatchableList[AnySpinQubit]:
    qubits = _get_qubits(node.machine, node.parameters)

    if isinstance(node.parameters, QubitsExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubits_batchable_list = _make_batchable_list_from_multiplexed(qubits, multiplexed)

    return qubits_batchable_list


def _get_qubits(
    machine: BaseQuamQD, node_parameters: QubitsExperimentNodeParameters
) -> List[AnySpinQubit]:
    if node_parameters.qubits is None or node_parameters.qubits == "":
        qubits = [machine.qubits[q] for q in machine.qubits]
    else:
        qubits = [machine.qubits[q] for q in node_parameters.qubits]

    return qubits

def enable_dual_drive_mw(node: QualibrationNode, waveform_name: str = "cw"): 
    """Run at the start of a QUA programme. If an XY drive dual output LO exists, it will enable."""
    qubits = get_qubits(node)
    # qubits are a BatchableList, with each Batch multiplexed. For multiplexed Batches, they share a drive line, 
    # so we only need to run the cw from a single qubit. 
    for qubits_batch in qubits.batch(): 
        # qubits_batch is a dict of {int : qubit}. We only need one, so just extract the first one quick and dirty. 
        first_qubit = list(qubits_batch.values())[0]
        # Now we drive, and then move onto the next batch 
        if hasattr(first_qubit.xy, "opx_output_LO"): 
            first_qubit.xy.opx_output_LO.play(waveform_name)
        else: 
            print(f"Qubit {first_qubit.name} in batch of {[k.name for k in qubits_batch.values()]} uses no dual output. Not activating")

def disable_dual_drive_mw(node: QualibrationNode, waveform_name: str = "cw"): 
    """Run at the end of a QUA programme. If an XY drive dual output LO exists, it will disable."""
    qubits = get_qubits(node)
    # qubits are a BatchableList, with each Batch multiplexed. For multiplexed Batches, they share a drive line, 
    # so we only need to run the cw from a single qubit. 
    for qubits_batch in qubits.batch(): 
        # qubits_batch is a dict of {int : qubit}. We only need one, so just extract the first one quick and dirty. 
        first_qubit = list(qubits_batch.values())[0]
        # Now we drive, and then move onto the next batch 
        if hasattr(first_qubit.xy, "opx_output_LO"): 
            first_qubit.xy.opx_output_LO.ramp_to_zero()
        else: 
            print(f"Qubit {first_qubit.name} in batch of {[k.name for k in qubits_batch.values()]} uses no dual output. Cannot disable.")

def enable_dual_drive_mw_pairs(node: QualibrationNode, waveform_name: str = "cw"): 
    """Run at the start of a QUA programme for qubit-pair nodes. Enables the LO on each batch's control qubit."""
    qubit_pairs = get_qubit_pairs(node)
    for pairs_batch in qubit_pairs.batch(): 
        first_qubit = list(pairs_batch.values())[0].qubit_control
        if hasattr(first_qubit.xy, "opx_output_LO"): 
            first_qubit.xy.opx_output_LO.play(waveform_name)
        else: 
            print(f"Qubit {first_qubit.name} in batch of {[k.qubit_control.name for k in pairs_batch.values()]} uses no dual output. Not activating")

def disable_dual_drive_mw_pairs(node: QualibrationNode, waveform_name: str = "cw"): 
    """Run at the end of a QUA programme for qubit-pair nodes. Disables the LO on each batch's control qubit."""
    qubit_pairs = get_qubit_pairs(node)
    for pairs_batch in qubit_pairs.batch(): 
        first_qubit = list(pairs_batch.values())[0].qubit_control
        if hasattr(first_qubit.xy, "opx_output_LO"): 
            first_qubit.xy.opx_output_LO.ramp_to_zero()
        else: 
            print(f"Qubit {first_qubit.name} in batch of {[k.qubit_control.name for k in pairs_batch.values()]} uses no dual output. Cannot disable.")

def get_xy_reference_pulse_name(qubit: AnySpinQubit) -> str:
    """Resolve the pulse name backing the qubit's default XY macros."""
    if qubit.xy is None:
        raise ValueError(f"Qubit '{qubit.id}' has no XY drive configured.")

    xy_drive_macro = qubit.macros.get(SingleQubitMacroName.XY_DRIVE)
    if xy_drive_macro is None:
        raise KeyError(
            f"Qubit '{qubit.id}' is missing the '{SingleQubitMacroName.XY_DRIVE}' macro."
        )

    pulse_name = getattr(xy_drive_macro, "reference_pulse_name", None)
    if pulse_name is None:
        raise ValueError(
            f"Qubit '{qubit.id}' XY-drive macro has no reference_pulse_name configured."
        )
    if pulse_name not in qubit.xy.operations:
        raise KeyError(
            f"Reference pulse '{pulse_name}' is not defined on qubit '{qubit.id}' XY drive."
        )

    return pulse_name


def quantize_pulse_length_ns(pulse_length_ns: int | float) -> int:
    """Round a pulse length to the nearest hardware-valid 4 ns multiple."""
    requested_length_ns = float(pulse_length_ns)
    rounded_length_ns = int(round(requested_length_ns / 4.0)) * 4

    if rounded_length_ns < 4:
        raise ValueError(f"Pulse length must be at least 4 ns, got {pulse_length_ns}.")

    return rounded_length_ns


def get_qubit_pairs(node: QualibrationNode) -> BatchableList[AnySpinQubitPair]:
    qubit_pairs = _get_qubit_pairs(node.machine, node.parameters)

    if isinstance(node.parameters, QubitPairExperimentNodeParameters):
        multiplexed = node.parameters.multiplexed
    else:
        multiplexed = False

    qubit_pairs_batchable_list = _make_batchable_list_from_multiplexed(
        qubit_pairs, multiplexed
    )

    return qubit_pairs_batchable_list


def _get_qubit_pairs(
    machine: BaseQuamQD, node_parameters: QubitPairExperimentNodeParameters
) -> List[AnySpinQubitPair]:
    if node_parameters.qubit_pairs is None or node_parameters.qubit_pairs == "":
        qubit_pairs = machine.active_qubit_pairs or list(machine.qubit_pairs.values())
    else:
        qubit_pairs = [machine.qubit_pairs[q] for q in node_parameters.qubit_pairs]

    return qubit_pairs


def plot_heralded_n_loops(
    ds_raw,
    item_names: list[str],
    *,
    item_dim: str,
    sweep_key: str,
    sweep_scale: float = 1.0,
    sweep_xlabel: str = "",
):
    """Plot average heralded loop count vs a sweep axis for each item."""
    import matplotlib.pyplot as plt

    if not item_names:
        return None

    fig, axes = plt.subplots(1, len(item_names), figsize=(7 * len(item_names), 5), squeeze=False)
    sweep_vals = ds_raw[sweep_key].values / sweep_scale

    for idx, item_name in enumerate(item_names):
        ax = axes[0, idx]
        candidates = [
            f"n_loops_{item_name}",
            f"n_loops_{item_name.rstrip('0123456789')}",
        ]
        n_key = next((key for key in candidates if key in ds_raw), None)
        if n_key is None:
            continue

        n_loops_vals = ds_raw[n_key].sel({item_dim: item_name}).values
        ax.plot(sweep_vals, n_loops_vals, color="C2", linestyle="--", label="n_loops")
        ax.set_xlabel(sweep_xlabel)
        ax.set_ylabel("n_loops")
        ax.set_title(f"n_loops vs {sweep_key} - {item_name}")
        ax.legend()

    fig.suptitle("Heralded initialization loop count")
    fig.tight_layout()
    return fig


def plot_heralded_n_loops_2d(
    ds_raw,
    item_names: list[str],
    *,
    item_dim: str,
    x_sweep_key: str,
    y_sweep_key: str,
    x_sweep_scale: float = 1.0,
    y_sweep_scale: float = 1.0,
    x_sweep_xlabel: str = "",
    y_sweep_ylabel: str = "",
):
    """Plot average heralded loop count on a 2-D sweep for each item."""
    import matplotlib.pyplot as plt
    import numpy as np

    if not item_names:
        return None

    fig, axes = plt.subplots(1, len(item_names), figsize=(7 * len(item_names), 5), squeeze=False)
    x_vals = ds_raw[x_sweep_key].values / x_sweep_scale
    y_vals = ds_raw[y_sweep_key].values / y_sweep_scale

    for idx, item_name in enumerate(item_names):
        ax = axes[0, idx]
        candidates = [
            f"n_loops_{item_name}",
            f"n_loops_{item_name.rstrip('0123456789')}",
        ]
        n_key = next((key for key in candidates if key in ds_raw), None)
        if n_key is None:
            continue

        n_loops_da = ds_raw[n_key].sel({item_dim: item_name})
        if (
            hasattr(n_loops_da, "dims")
            and y_sweep_key in n_loops_da.dims
            and x_sweep_key in n_loops_da.dims
        ):
            n_loops_da = n_loops_da.transpose(y_sweep_key, x_sweep_key)
        n_loops_vals = np.asarray(n_loops_da.values, dtype=float)
        image = ax.pcolormesh(x_vals, y_vals, n_loops_vals, shading="auto", cmap="viridis")
        ax.set_xlabel(x_sweep_xlabel)
        ax.set_ylabel(y_sweep_ylabel)
        ax.set_title(f"n_loops map - {item_name}")
        fig.colorbar(image, ax=ax, label="n_loops")

    fig.suptitle("Heralded initialization loop count")
    fig.tight_layout()
    return fig
