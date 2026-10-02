"""2Q GST circuit translation, tokenization, and QUA playback helpers."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import numpy as np
import pygsti
from qm.qua import *

from calibration_utils.gate_set_tomography.gst_utils import (
    GERM_TOKENS_STREAM_NAME,
    OPX1000_GATE_TABLE_LIMIT,
    log_gst_design_summary,
    start_push_gst_germs_in_background,
)

# Layer opcodes streamed to the OPX. Single-qubit layers: q0 + 3 * q1 with
# {0: I, 1: x90, 2: y90}. CZ is a dedicated opcode.
SQ_I, SQ_X90, SQ_Y90 = 0, 1, 2
LAYER_CZ = 9
OUTCOME_LABELS = ("00", "01", "10", "11")

_SINGLE_QUBIT_X = {"Gxpi2", "Gx"}
_SINGLE_QUBIT_Y = {"Gypi2", "Gy"}
_IDLE = {"Gi", "Gidle", "I"}
_CZ = {"gcphase", "gcz", "gcphase:0:1"}


@dataclass
class GST2QExperimentDesign:
    """Prepared two-qubit pyGSTi experiment design and tokenized circuits."""

    std_model: object
    exp_design: object
    all_germs_to_qua_tokenized_labels: List[List[int]]
    all_germs_depth: List[int]
    max_germs_depth: int
    total_germs_num: int
    static_gate_table_size: int
    max_circuit_length: List[int]


def _load_two_qubit_pack():
    """Load the XYI+CPHASE / CZ two-qubit model pack."""
    try:
        from pygsti.modelpacks import smq2Q_XYICPHASE as pack
    except ImportError:
        from pygsti.modelpacks import smq2Q_XYCPHASE as pack
    return pack


def _label_name(label) -> str:
    name = getattr(label, "name", None)
    return str(name if name is not None else label)


def _label_qubits(label) -> tuple[str, ...]:
    sslbls = getattr(label, "sslbls", None)
    if sslbls is None:
        return ()
    return tuple(str(q) for q in sslbls)


def _iter_layer_labels(layer) -> list[Any]:
    components = getattr(layer, "components", None)
    return list(components) if components is not None else [layer]


def _qubit_slot(qubits: Sequence[str]):
    """Map pyGSTi sslbls onto pair slot 0 (control) or 1 (target)."""
    if not qubits:
        return None
    if any(q in ("0", "q0") for q in qubits) and not any(q in ("1", "q1") for q in qubits):
        return 0
    if any(q in ("1", "q1") for q in qubits) and not any(q in ("0", "q0") for q in qubits):
        return 1
    if qubits[0] in ("0", "q0"):
        return 0
    if qubits[0] in ("1", "q1"):
        return 1
    return None


def encode_native_layer(native_layer: list[dict[str, Any]]) -> int:
    """Encode one native layer (from CQT_2Q_GST translation) as a QUA opcode."""
    q0_gate, q1_gate = SQ_I, SQ_I
    cz = False
    for item in native_layer:
        gate = str(item.get("gate", "")).upper()
        qubits = [str(q) for q in item.get("qubits", [])]
        if gate in ("CZ", "CPHASE"):
            cz = True
            continue
        slot = _qubit_slot(qubits)
        code = SQ_I
        if gate in ("X90", "XPI2", "GXPI2"):
            code = SQ_X90
        elif gate in ("Y90", "YPI2", "GYPI2"):
            code = SQ_Y90
        elif gate in ("I", "IDLE"):
            code = SQ_I
        else:
            raise KeyError(f"No QUA mapping for native gate {item!r}")
        if slot == 1:
            q1_gate = code
        else:
            q0_gate = code
    if cz:
        if q0_gate != SQ_I or q1_gate != SQ_I:
            raise ValueError(f"CZ cannot share a layer with a single-qubit gate: {native_layer}")
        return LAYER_CZ
    return q0_gate + 3 * q1_gate


def translate_pygsti_layer(layer) -> list[dict[str, Any]]:
    """Translate one pyGSTi layer into the CQT native-layer schema."""
    native_layer: list[dict[str, Any]] = []
    for label in _iter_layer_labels(layer):
        name = _label_name(label)
        qubits = list(_label_qubits(label))
        lower = name.lower()
        if name in _IDLE or lower in _IDLE:
            native_layer.append({"gate": "I", "qubits": qubits})
        elif name in _SINGLE_QUBIT_X:
            native_layer.append({"gate": "X90", "qubits": qubits})
        elif name in _SINGLE_QUBIT_Y:
            native_layer.append({"gate": "Y90", "qubits": qubits})
        elif lower in ("gcphase", "gcz") or "cphase" in lower or lower == "gcz":
            native_layer.append({"gate": "CZ", "qubits": qubits})
        else:
            raise KeyError(f"No CQT/QUA mapping for pyGSTi label {label!r} (name={name!r})")
    return native_layer


def tokenize_pygsti_circuit(circuit) -> list[int]:
    """Convert a pyGSTi Circuit into a list of 2Q layer opcodes."""
    tokens = [encode_native_layer(translate_pygsti_layer(layer)) for layer in circuit]
    return tokens


def tokenize_gst_circuits(circuits: Iterable) -> tuple[list[list[int]], list[int]]:
    """Length-prefix 2Q circuits into a rectangular table for AIS / static compile.

    Each row is ``[L, op_1, ..., op_L, 0, ...]`` with width ``max_depth + 1``.
    Slot 0 is the true layer count so the QUA loop never executes the padding.
    """
    tokenized = [tokenize_pygsti_circuit(circuit) for circuit in circuits]
    if not tokenized:
        return [], []
    depths = [len(tokens) for tokens in tokenized]
    max_depth = max(max(depths), 1)
    used = {op for tokens in tokenized for op in tokens}
    if not used <= set(range(LAYER_CZ + 1)):
        raise ValueError(f"2Q GST opcodes {sorted(used)} outside switch cases 0..{LAYER_CZ}")
    padded = [[len(tokens)] + tokens + [0] * (max_depth - len(tokens)) for tokens in tokenized]
    return padded, depths


def _call_pack_design(pack, max_length: int, use_fpr: bool):
    """Build a 2Q GST design from the model pack, matching CQT_2Q_GST.ipynb."""
    if hasattr(pack, "create_gst_experiment_design"):
        fn = pack.create_gst_experiment_design
    elif hasattr(pack, "get_gst_experiment_design"):
        fn = pack.get_gst_experiment_design
    else:
        fn = None

    if fn is not None:
        if use_fpr:
            for kwargs in (
                {"max_length": max_length, "fpr": True},
                {"max_max_length": max_length, "fpr": True},
            ):
                try:
                    return fn(**kwargs)
                except TypeError:
                    pass
            try:
                return fn(max_length, fpr=True)
            except TypeError:
                pass
        try:
            return fn(max_length)
        except TypeError:
            return fn(max_max_length=max_length)

    std_model = pack.target_model()
    return pygsti.protocols.StandardGSTDesign(
        std_model,
        pack.prep_fiducials(),
        pack.meas_fiducials(),
        pack.germs(),
        [2**i for i in range(int(np.log2(max(max_length, 1))) + 1)],
    )


def setup_gst_experiment_2q(
    max_circuit_depth_in_power: int,
    use_fiducial_pair_reduction: bool = True,
) -> GST2QExperimentDesign:
    """Build the 2Q StandardGST design and tokenize every circuit as QUA layers."""
    max_circuit_length = [2**i for i in range(max(max_circuit_depth_in_power, 0) + 1)]
    pack = _load_two_qubit_pack()
    std_model = pack.target_model()
    exp_design = _call_pack_design(
        pack,
        max_length=max_circuit_length[-1],
        use_fpr=use_fiducial_pair_reduction,
    )

    circuits = list(exp_design.all_circuits_needing_data)
    tokenized, depths = tokenize_gst_circuits(circuits)
    max_germs_depth = max(depths) if depths else 1
    total_germs_num = len(tokenized)
    static_gate_table_size = total_germs_num * (max_germs_depth + 1)

    return GST2QExperimentDesign(
        std_model=std_model,
        exp_design=exp_design,
        all_germs_to_qua_tokenized_labels=tokenized,
        all_germs_depth=depths,
        max_germs_depth=max_germs_depth,
        total_germs_num=total_germs_num,
        static_gate_table_size=static_gate_table_size,
        max_circuit_length=max_circuit_length,
    )


def require_cz_align_elements(qubit_pair, cz_operation: str) -> None:
    params = inspect.signature(qubit_pair.macros[cz_operation].apply).parameters
    if "align_elements" not in params:
        raise RuntimeError(
            f"{cz_operation!r} on {qubit_pair.name} has no apply(align_elements=...); the installed "
            "quam-builder would swallow it via **kwargs. Install quam-builder with PR #154 "
            "(hotfix/cz-align-elements-opt-in)."
        )


def play_tokenized_gst_circuits_2q(tokenized_germ, qubit_pair, cz_operation: str):
    """Play a length-prefixed 2Q GST circuit. Slot 0 is the layer count; opcodes start at 1.

    Idle is one x90: x90 and y90 must have the same length on both qubits.
    The switch has no align. CZ is apply(align_elements=False) plus a 16 ns XY wait.
    """
    q0 = qubit_pair.qubit_control
    q1 = qubit_pair.qubit_target
    lengths = {(q.name, op): q.xy.operations[op].length for q in (q0, q1) for op in ("x90", "y90")}
    gate_ns = lengths[(q0.name, "x90")]
    if set(lengths.values()) != {gate_ns} or gate_ns % 4 or gate_ns < 16:
        raise ValueError(f"2Q GST needs equal x90/y90 lengths on both qubits, multiple of 4, >= 16: {lengths}")
    idle = gate_ns // 4
    i = declare(int)
    n_layers = declare(int)
    assign(n_layers, tokenized_germ[0])
    with for_(i, 1, i <= n_layers, i + 1):
        with switch_(tokenized_germ[i], unsafe=True):
            with case_(0):  # I ⊗ I
                q0.xy.wait(idle)
                q1.xy.wait(idle)
            with case_(1):  # X90 ⊗ I
                q0.xy.play("x90")
                q1.xy.wait(idle)
            with case_(2):  # Y90 ⊗ I
                q0.xy.play("y90")
                q1.xy.wait(idle)
            with case_(3):  # I ⊗ X90
                q0.xy.wait(idle)
                q1.xy.play("x90")
            with case_(4):  # X90 ⊗ X90
                q0.xy.play("x90")
                q1.xy.play("x90")
            with case_(5):  # Y90 ⊗ X90
                q0.xy.play("y90")
                q1.xy.play("x90")
            with case_(6):  # I ⊗ Y90
                q0.xy.wait(idle)
                q1.xy.play("y90")
            with case_(7):  # X90 ⊗ Y90
                q0.xy.play("x90")
                q1.xy.play("y90")
            with case_(8):  # Y90 ⊗ Y90
                q0.xy.play("y90")
                q1.xy.play("y90")
            with case_(LAYER_CZ):
                qubit_pair.macros[cz_operation].apply(align_elements=False)
                q0.xy.wait(4)
                q1.xy.wait(4)


__all__ = [
    "GERM_TOKENS_STREAM_NAME",
    "GST2QExperimentDesign",
    "LAYER_CZ",
    "OPX1000_GATE_TABLE_LIMIT",
    "OUTCOME_LABELS",
    "log_gst_design_summary",
    "play_tokenized_gst_circuits_2q",
    "require_cz_align_elements",
    "setup_gst_experiment_2q",
    "start_push_gst_germs_in_background",
]
