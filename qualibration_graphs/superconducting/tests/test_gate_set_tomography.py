"""Pin the current GST tokenization and the shot-to-count reconstruction.

Both 1Q and 2Q rows are length-prefixed: ``[L, ops..., 0-pad]`` with width
``max_depth + 1``. A later change to that convention has to update these checks
on purpose.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pygsti")
from pygsti.baseobjs import Label  # noqa: E402

from calibration_utils.gate_set_tomography.analysis import shots_to_count_dataset  # noqa: E402
from calibration_utils.gate_set_tomography.gst_utils import setup_gst_experiment  # noqa: E402
from calibration_utils.gate_set_tomography_2q.gst_utils import (  # noqa: E402
    LAYER_CZ,
    encode_native_layer,
    require_cz_align_elements,
    setup_gst_experiment_2q,
)


def test_1q_rows_are_length_prefixed():
    design = setup_gst_experiment(0)
    rows = design.all_germs_to_qua_tokenized_labels
    width = design.max_germs_depth + 1
    assert len(rows) == design.total_germs_num
    for row, n_gates in zip(rows, design.all_germs_depth):
        assert len(row) == width
        assert row[0] == n_gates
        assert row[1 + n_gates :] == [0] * (width - 1 - n_gates)


def test_1q_empty_fiducial_is_not_an_idle_gate():
    design = setup_gst_experiment(2)
    circuits = list(design.exp_design.all_circuits_needing_data)
    idle = Label(())
    empty_seen = False
    for circuit, row, n_gates in zip(circuits, design.all_germs_to_qua_tokenized_labels, design.all_germs_depth):
        assert row[0] == n_gates == len(circuit)
        n_idle = sum(layer == idle for layer in circuit.layertup)
        assert row[1 : 1 + n_gates].count(0) == n_idle
        if len(circuit) == 0:
            empty_seen = True
            assert row[0] == 0
            assert row[1:] == [0] * (len(row) - 1)
    assert empty_seen


def test_2q_rows_are_length_prefixed_and_cz_opcode_matches_the_design():
    design = setup_gst_experiment_2q(1, use_fiducial_pair_reduction=True)
    rows = design.all_germs_to_qua_tokenized_labels
    width = design.max_germs_depth + 1
    circuits = list(design.exp_design.all_circuits_needing_data)
    assert len(rows) == design.total_germs_num == len(circuits)
    assert design.static_gate_table_size == design.total_germs_num * width
    opcodes = set()
    n_cz = 0
    for circuit, row, depth in zip(circuits, rows, design.all_germs_depth):
        assert len(row) == width
        assert row[0] == depth == len(circuit)
        assert all(0 <= token <= LAYER_CZ for token in row[1 : 1 + depth])
        assert row[1 + depth :] == [0] * (width - 1 - depth)
        opcodes.update(row[1 : 1 + depth])
        n_cz += sum(token == LAYER_CZ for token in row[1 : 1 + depth])
    assert opcodes <= set(range(LAYER_CZ + 1))

    n_gcphase = 0
    for circuit in circuits:
        for layer in circuit:
            n_gcphase += sum(label.name == "Gcphase" for label in layer.components)
    assert n_cz == n_gcphase


def test_2q_empty_circuit_is_a_zero_length_row():
    design = setup_gst_experiment_2q(0, use_fiducial_pair_reduction=False)
    circuits = list(design.exp_design.all_circuits_needing_data)
    empty_seen = False
    for circuit, row in zip(circuits, design.all_germs_to_qua_tokenized_labels):
        assert row[0] == len(circuit)
        if len(circuit) == 0:
            empty_seen = True
            assert row == [0] * len(row)
    assert empty_seen


def test_require_cz_align_elements_rejects_a_kwargs_only_apply():
    class _Macro:
        def apply(self, **kwargs):
            return kwargs

    class _Pair:
        name = "q0-q1"
        macros = {"cz": _Macro()}

    with pytest.raises(RuntimeError, match="align_elements"):
        require_cz_align_elements(_Pair(), "cz")


def test_2q_cz_parallel_with_a_single_qubit_gate_is_rejected():
    with pytest.raises(ValueError):
        encode_native_layer(
            [
                {"gate": "X90", "qubits": ["0"]},
                {"gate": "CZ", "qubits": ["0", "1"]},
            ]
        )


def test_shots_to_count_dataset_sums_a_full_record():
    # two circuits, four shots: circuit 0 has two 1s, circuit 1 has one
    flat = [0, 1, 1, 0, 1, 0, 0, 0]
    ds = shots_to_count_dataset(flat, "q1", total_germs_num=2, n_runs=4)
    assert list(ds.count1.values[0]) == [2, 1]
    assert list(ds.count0.values[0]) == [2, 3]
    assert np.allclose(ds.state.values[0], [0.5, 0.25])


def test_shots_to_count_dataset_rounds_an_averaged_record():
    # astype(int) would turn 0.29 * 100 into 28
    ds = shots_to_count_dataset([0.29, 0.0, 1.0], "q1", total_germs_num=3, n_runs=100)
    assert list(ds.count1.values[0]) == [29, 0, 100]
    assert list(ds.count0.values[0]) == [71, 100, 0]


def test_shots_to_count_dataset_rejects_a_wrong_length():
    with pytest.raises(ValueError):
        shots_to_count_dataset([0, 1, 2], "q1", total_germs_num=2, n_runs=4)
