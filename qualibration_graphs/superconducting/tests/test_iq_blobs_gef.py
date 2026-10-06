"""Tests for GEF readout helpers: pulse setup, IF shift, and reset."""

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.common_utils.gef_readout_pulse import (  # noqa: E402
    ensure_gef_readout_pulse,
    gef_readout_frequency,
    reset_for_gef,
)


@dataclasses.dataclass
class _Pulse:
    length: int
    amplitude: float
    threshold: object = 0.1
    rus_exit_threshold: object = 0.2


def _qubit(readout, gef=None, name="q1"):
    operations = {"readout": readout}
    if gef is not None:
        operations["readout_GEF"] = gef
    return SimpleNamespace(name=name, resonator=SimpleNamespace(operations=operations))


def test_ensure_gef_readout_pulse_derives_missing_operation_from_readout():
    qubit = _qubit(_Pulse(length=1000, amplitude=0.08))
    ensure_gef_readout_pulse([qubit])
    gef = qubit.resonator.operations["readout_GEF"]
    assert gef.length == 1500
    assert gef.amplitude == 0.08
    assert gef.threshold is None
    assert gef.rus_exit_threshold is None


def test_ensure_gef_readout_pulse_warns_on_builder_default():
    warnings = []
    qubit = _qubit(
        _Pulse(length=1000, amplitude=0.08),
        gef=_Pulse(length=2000, amplitude=0.01),
    )
    ensure_gef_readout_pulse([qubit], log_callable=lambda msg, level="info": warnings.append((msg, level)))
    assert len(warnings) == 1
    assert warnings[0][1] == "warning"
    assert "quam_builder default" in warnings[0][0]


def test_gef_readout_frequency_treats_missing_shift_as_zero():
    qubit = SimpleNamespace(resonator=SimpleNamespace(intermediate_frequency=50e6, GEF_frequency_shift=None))
    assert gef_readout_frequency(qubit) == 50e6
    qubit.resonator.GEF_frequency_shift = 2e6
    assert gef_readout_frequency(qubit) == 52e6
    assert gef_readout_frequency(qubit, extra_detuning=1e6) == 53e6


def test_reset_for_gef_restores_frequency_after_active_gef():
    calls = []
    qubit = SimpleNamespace(
        thermalization_time=100,
        reset=lambda reset_type, simulate: calls.append(("reset", reset_type, simulate)),
        wait=lambda t: calls.append(("wait", t)),
        resonator=SimpleNamespace(
            intermediate_frequency=50e6,
            GEF_frequency_shift=2e6,
            update_frequency=lambda f: calls.append(("freq", f)),
        ),
    )
    u = SimpleNamespace(ns=1)
    reset_for_gef(qubit, "thermal", False, u)
    assert calls == [("wait", 200)]
    calls.clear()
    reset_for_gef(qubit, "active_gef", False, u)
    assert calls == [("reset", "active_gef", False), ("freq", 52e6)]
