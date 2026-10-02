"""Pin the current GST tokenization and the shot-to-count reconstruction.

The 1Q rows are length-prefixed. The 2Q rows are padded with -1 out to the longest
circuit and have no length prefix. A later change to either convention has to
update these checks on purpose.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("pygsti")

from calibration_utils.gate_set_tomography.analysis import shots_to_count_dataset  # noqa: E402
from calibration_utils.gate_set_tomography.gst_utils import setup_gst_experiment  # noqa: E402
from calibration_utils.gate_set_tomography_2q.gst_utils import (  # noqa: E402
    LAYER_CZ,
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


def test_2q_rows_are_padded_and_cz_opcode_matches_the_design():
    design = setup_gst_experiment_2q(1, use_fiducial_pair_reduction=True)
    rows = design.all_germs_to_qua_tokenized_labels
    assert len(rows) == design.total_germs_num
    opcodes = set()
    n_cz = 0
    for row, depth in zip(rows, design.all_germs_depth):
        assert len(row) == design.max_germs_depth
        assert all(token != -1 for token in row[:depth])
        assert row[depth:] == [-1] * (design.max_germs_depth - depth)
        opcodes.update(row)
        n_cz += sum(token == LAYER_CZ for token in row)
    assert opcodes <= set(range(10)) | {-1}

    n_gcphase = 0
    for circuit in design.exp_design.all_circuits_needing_data:
        for layer in circuit:
            n_gcphase += sum(label.name == "Gcphase" for label in layer.components)
    assert n_cz == n_gcphase


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
