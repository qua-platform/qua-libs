"""Transition frequencies of the f-LRU, from the qubit and resonator frequencies in the QUAM state.

Conventions, all in Hz:

* f_01 = ``qubit.xy.RF_frequency``, the g-e frequency;
* f_12 = f_01 - |anharmonicity|, the e-f frequency (the convention used in this repo);
* f_02 = f_01 + f_12, the g-f frequency;
* f_r = ``qubit.resonator.RF_frequency``, the readout frequency, a few MHz from the bare resonator.

The pump drives |f0> -> |g1> at f_02 - f_r (``f0g1_frequency``).
"""

import logging
from typing import NamedTuple

logger = logging.getLogger(__name__)


class LruReferenceFrequencies(NamedTuple):
    """Qubit and resonator frequencies the LRU transitions are computed from, in Hz."""

    f_01: float
    f_12: float
    f_02: float
    f_r: float


def lru_reference_frequencies(qubit) -> LruReferenceFrequencies:
    """f_01, f_12, f_02 and f_r of ``qubit``, following the conventions of this module.

    f_12 is computed from ``qubit.anharmonicity``, the value node 26a also uses to prepare |f>. Do not use
    the look-alike fields instead: ``qubit.f_12`` is not set on any qubit, ``qubit.resonator.f_12`` is a
    readout frequency, and the EF_x180 pulse's ``anharmonicity`` is a DRAG parameter. An error in f_01
    enters f0-g1 twice, through f_02.

    Raises:
        ValueError: when the anharmonicity, the qubit frequency or the resonator frequency is not set.
    """
    required = (
        (qubit.anharmonicity, "anharmonicity", "12"),
        (qubit.xy.RF_frequency, "xy.RF_frequency", "03a"),
        (qubit.resonator.RF_frequency, "resonator.RF_frequency", "02a"),
    )
    for value, name, node in required:
        if value is None:
            raise ValueError(f"{qubit.name}: {name} is not set; calibrate it with node {node} first.")
    f_01 = float(qubit.xy.RF_frequency)
    f_12 = f_01 - abs(float(qubit.anharmonicity))
    return LruReferenceFrequencies(f_01=f_01, f_12=f_12, f_02=f_01 + f_12, f_r=float(qubit.resonator.RF_frequency))


def f0g1_frequency(qubit) -> float:
    """Expected |f0> -> |g1> frequency f_02 - f_r in Hz, at zero drive amplitude (node 26a).

    With the resonator above f_02 the value is negative; its absolute value is returned and a warning
    is logged, since that regime is untested.
    """
    ref = lru_reference_frequencies(qubit)
    frequency = ref.f_02 - ref.f_r
    if frequency <= 0:
        logger.warning(
            f"{qubit.name}: the resonator ({ref.f_r / 1e9:.4f} GHz) lies above f_02 ({ref.f_02 / 1e9:.4f} GHz); "
            "the f0-g1 pump is then untested."
        )
    return abs(frequency)
