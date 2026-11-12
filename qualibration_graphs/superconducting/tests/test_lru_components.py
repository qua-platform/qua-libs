"""The f-LRU |f0> -> |g1> transition frequency, its inputs and the transition table.

Node 26a (microwave pump) drives |f0> -> |g1>. Uses a stand-in qubit with typical transmon values, no QUAM
state needed.
"""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import get_args

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quam_config.lru_components import (  # noqa: E402
    LEGACY_LRU_TRANSITION,
    LRU_TRANSITIONS,
    LruTransition,
    f0g1_frequency,
)

F_01 = 4.5735e9
F_R = 5.91e9


def _qubit(anharmonicity=-160.5e6, f_01=F_01, f_r=F_R):
    return SimpleNamespace(
        name="qtest",
        anharmonicity=anharmonicity,
        xy=SimpleNamespace(RF_frequency=f_01),
        resonator=SimpleNamespace(RF_frequency=f_r),
    )


def test_f0g1_frequency():
    qubit = _qubit()
    assert f0g1_frequency(qubit) == pytest.approx(3.0765e9, abs=1e3)


def test_sign_of_the_anharmonicity_does_not_matter():
    qubit = _qubit(anharmonicity=+160.5e6)
    assert f0g1_frequency(qubit) == pytest.approx(3.0765e9, abs=1e3)


@pytest.mark.parametrize(
    "kwargs, node",
    [({"anharmonicity": None}, "node 12"), ({"f_01": None}, "node 03a"), ({"f_r": None}, "node 02a")],
)
def test_missing_input_names_the_qubit_and_the_calibrating_node(kwargs, node):
    qubit = _qubit(**kwargs)
    with pytest.raises(ValueError, match=node) as error:
        f0g1_frequency(qubit)
    assert "qtest" in str(error.value)


def test_resonator_above_f02_returns_the_absolute_value_and_warns(caplog):
    qubit = _qubit(f_r=10e9)  # above f_02 = 8.9865 GHz
    with caplog.at_level(logging.WARNING, logger="quam_config.lru_components"):
        assert f0g1_frequency(qubit) == pytest.approx(1.0135e9, abs=1e3)
    assert "qtest" in caplog.text


def test_transition_table():
    assert set(LRU_TRANSITIONS) == set(get_args(LruTransition))
    assert LEGACY_LRU_TRANSITION == "f0g1"
