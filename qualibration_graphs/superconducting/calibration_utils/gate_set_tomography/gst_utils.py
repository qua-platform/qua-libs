"""GST circuit parsing, tokenization, and input-stream helpers."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, List

import numpy as np
import pygsti
from pygsti.modelpacks import smq1Q_XYI as std
from qm.qua import *

if TYPE_CHECKING:
    from quam_builder.architecture.superconducting.qubit import AnyTransmon

OPX1000_GATE_TABLE_LIMIT = 16000
GERM_TOKENS_STREAM_NAME = "germ_tokens"

GATE_SET_TOKEN = {
    "I": 0,
    "x90": 1,
    "y90": 2,
    "x180": 3,
    "y180": 4,
}


@dataclass
class GSTExperimentDesign:
    """Prepared pyGSTi experiment design and tokenized germ sequences."""

    std_model: object
    exp_design: pygsti.protocols.StandardGSTDesign
    all_germs_to_qua_tokenized_labels: List[List[int]]
    all_germs_depth: List[int]
    max_germs_depth: int
    total_germs_num: int
    static_gate_table_size: int
    max_circuit_length: List[int]


def parse_gst_circuit_string(circuit_str: str) -> List[str]:
    """Parse a pyGSTi circuit string into pulse labels for QUA."""
    clean_str = circuit_str.split()[0]
    clean_str = clean_str.replace("@(0)", "")
    clean_str = clean_str.replace("({})", "(I)")
    clean_str = clean_str.replace("{}", "(I)")
    clean_str = clean_str.replace("([])", "(I)")

    token_map = {
        "Gxpi2:0": "X",
        "Gypi2:0": "Y",
        "I": "I",
    }
    temp_str = clean_str
    for key, token in token_map.items():
        temp_str = temp_str.replace(key, token)

    while "^" in temp_str:

        def expand_match(match):
            content = match.group(1)
            power = int(match.group(2))
            return content * power

        temp_str = re.sub(r"\(([^)]+)\)\^(\d+)", expand_match, temp_str)

    temp_str = temp_str.replace("(", "").replace(")", "")

    final_map = {
        "X": "x90",
        "Y": "y90",
        "I": "I",
    }
    return [final_map[ch] for ch in temp_str if ch in final_map]


def tokenize_gst_circuits(gst_str: Iterable[Iterable[str]]) -> tuple[list[list[int]], list[int]]:
    """Convert GST circuit label lists into integer tokens for QUA.

    Each row is length-prefixed: ``[L, t_1, ..., t_L, 0, ...]`` with length
    ``max_gate_set_length + 1``. Slot 0 is the true gate count so the QUA loop
    never executes the padding, and the switch cases stay contiguous (0..4).
    """
    gate_set_length_list = [len(g) for g in gst_str]
    max_gate_set_length = max(gate_set_length_list)
    tokenized_circuits = []

    for germ in gst_str:
        tokenized_circuits.append(
            [len(germ)] + [GATE_SET_TOKEN[g] for g in germ] + [0] * (max_gate_set_length - len(germ))
        )

    return tokenized_circuits, gate_set_length_list


def setup_gst_experiment(max_circuit_depth_in_power: int) -> GSTExperimentDesign:
    """Build the pyGSTi StandardGSTDesign and tokenize all germ sequences."""
    max_circuit_length = [2**i for i in range(max(max_circuit_depth_in_power, 0) + 1)]
    std_model = std.target_model()
    exp_design = pygsti.protocols.StandardGSTDesign(
        std_model,
        std.prep_fiducials(),
        std.meas_fiducials(),
        std.germs(),
        max_circuit_length,
    )

    all_germs_from_gst_model = [s.str for s in exp_design.all_circuits_needing_data]
    all_germs_to_qua_labels = [parse_gst_circuit_string(s) for s in all_germs_from_gst_model]
    all_germs_to_qua_tokenized_labels, all_germs_depth = tokenize_gst_circuits(all_germs_to_qua_labels)
    max_germs_depth = max(all_germs_depth)
    total_germs_num = len(all_germs_to_qua_tokenized_labels)
    # row is [L, tokens..., padding], one int longer than the longest germ
    static_gate_table_size = total_germs_num * (max_germs_depth + 1)

    return GSTExperimentDesign(
        std_model=std_model,
        exp_design=exp_design,
        all_germs_to_qua_tokenized_labels=all_germs_to_qua_tokenized_labels,
        all_germs_depth=all_germs_depth,
        max_germs_depth=max_germs_depth,
        total_germs_num=total_germs_num,
        static_gate_table_size=static_gate_table_size,
        max_circuit_length=max_circuit_length,
    )


def log_gst_design_summary(
    design: GSTExperimentDesign,
    n_runs: int,
    log_callable=print,
) -> None:
    """Print a summary of the GST experiment design."""
    log_callable("=== GST experiment design summary ===")
    log_callable(f"max_circuit_length passed to pyGSTi: {design.max_circuit_length}")
    log_callable(f"Longest germ sequence (max_germs_depth): {design.max_germs_depth}")
    log_callable(f"Total number of gate lists (circuits/germs): {design.total_germs_num}")
    log_callable(
        f"Static gate-table size if compiled inline: {design.static_gate_table_size} "
        f"({design.total_germs_num} circuits, longest germ {design.max_germs_depth})"
    )
    log_callable(f"OPX1000 gate-table limit: {OPX1000_GATE_TABLE_LIMIT}")
    if design.static_gate_table_size > OPX1000_GATE_TABLE_LIMIT:
        log_callable(
            f"Static table EXCEEDS limit by "
            f"{design.static_gate_table_size - OPX1000_GATE_TABLE_LIMIT}; "
            "this node streams germs via advance_input_stream."
        )
    else:
        log_callable("Static table fits in OPX memory; AIS still used to avoid future depth scaling issues.")
    log_callable(f"Input stream pushes (one per circuit): {design.total_germs_num}")
    log_callable(f"Stream processing: buffer({n_runs}).buffer({design.total_germs_num})  " "[inner=runs, outer=germs]")
    log_callable(
        f"Total measurements: {n_runs} runs x {design.total_germs_num} germs = " f"{n_runs * design.total_germs_num}"
    )
    log_callable("=====================================")


def play_tokenized_gst_circuits(tokenized_germ, qubit: "AnyTransmon"):
    """Play a length-prefixed GST germ. Slot 0 is the gate count; tokens start at 1."""
    i = declare(int)
    n_gates = declare(int)
    assign(n_gates, tokenized_germ[0])
    with for_(i, 1, i <= n_gates, i + 1):
        with switch_(tokenized_germ[i], unsafe=True):
            with case_(0):
                qubit.xy.wait(4)
            with case_(1):
                qubit.xy.play("x90")
            with case_(2):
                qubit.xy.play("y90")
            with case_(3):
                qubit.xy.play("x180")
            with case_(4):
                qubit.xy.play("y180")


def push_gst_germs_to_input_stream(job, tokenized_germs: Iterable[Iterable[int]]) -> None:
    """Push each germ sequence once; QUA repeats it num_runs times before advancing."""
    total_pushes = len(tokenized_germs)
    for push_count, tokens in enumerate(tokenized_germs, start=1):
        job.push_to_input_stream(GERM_TOKENS_STREAM_NAME, tokens)
        if push_count % 100 == 0 or push_count == total_pushes:
            print(f"Input stream: pushed {push_count}/{total_pushes} germ sequences")


def start_push_gst_germs_in_background(job, tokenized_germs: Iterable[Iterable[int]]) -> threading.Thread:
    """Start a daemon thread that pushes germ tokens to the OPX input stream."""
    thread = threading.Thread(
        target=push_gst_germs_to_input_stream,
        args=(job, tokenized_germs),
        daemon=True,
    )
    thread.start()
    return thread
