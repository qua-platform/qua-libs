"""The in-memory f-LRU element must leave no trace once removed, and survive a save when kept.

The LRU calibration node adds the element to the qubits it runs on and removes it before
``node.save()`` unless the user asked to keep it, so removal has to restore the state exactly.
Each test builds a two-qubit MW-FEM machine (flux-tunable or fixed-frequency) in a temporary folder.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qualang_tools.wirer import Connectivity, Instruments, allocate_wiring  # noqa: E402
from qualang_tools.wirer.wirer.channel_specs import mw_fem_spec  # noqa: E402
from quam.core import quam_dataclass  # noqa: E402
from quam_builder.architecture.superconducting.qpu import FixedFrequencyQuam, FluxTunableQuam  # noqa: E402
from quam_builder.builder.qop_connectivity import build_quam_wiring  # noqa: E402
from quam_builder.builder.superconducting import build_quam  # noqa: E402

from calibration_utils.common_utils.flux_distortions import plan_lo_shift_for_frequency_window  # noqa: E402
from calibration_utils.lru_calibration.element import (  # noqa: E402
    LRU_NOMINAL_IF,
    MW_FEM_BANDS,
    add_lru_element,
    lru_upconverter_frequency,
    prediction_outside_window,
    spurious_lines,
)
from quam_config.lru_components import (  # noqa: E402
    LRU_UPCONVERTER,
    LruFixedFrequencyTransmon,
    LruFluxTunableTransmon,
)

QUBITS = ("q1", "q2")
# Typical frequencies: f_01, anharmonicity, f_r; f0-g1 = 2 * f_01 - |anharmonicity| - f_r is about 3 GHz.
FREQUENCIES = {"q1": (4.5735e9, -160.5e6, 5.91e9), "q2": (4.82e9, -200e6, 6.05e9)}


@quam_dataclass
class _FluxQuam(FluxTunableQuam):
    pass


@quam_dataclass
class _FixedQuam(FixedFrequencyQuam):
    pass


def _build(quam_class, state_dir: Path, monkeypatch):
    """Two qubits with their own MW-FEM drive port, frequencies set as after spectroscopy."""
    monkeypatch.setenv("QUAM_STATE_PATH", str(state_dir))
    instruments = Instruments()
    instruments.add_mw_fem(controller=1, slots=[1])
    instruments.add_lf_fem(controller=1, slots=[3])
    connectivity = Connectivity()
    connectivity.add_resonator_line(qubits=[1, 2], constraints=mw_fem_spec(con=1, slot=1, in_port=1, out_port=1))
    connectivity.add_qubit_drive_lines(qubits=[1, 2])
    if quam_class is _FluxQuam:
        connectivity.add_qubit_flux_lines(qubits=[1, 2])
    allocate_wiring(connectivity, instruments)
    build_quam_wiring(connectivity, "127.0.0.1", "test_cluster", quam_class())
    machine = build_quam(quam_class.load())
    for name, (f_01, anharmonicity, f_r) in FREQUENCIES.items():
        qubit = machine.qubits[name]
        qubit.xy.RF_frequency = f_01
        qubit.anharmonicity = anharmonicity
        qubit.resonator.RF_frequency = f_r
    return machine


@pytest.fixture(
    params=[(_FluxQuam, LruFluxTunableTransmon), (_FixedQuam, LruFixedFrequencyTransmon)], ids=["flux", "fixed"]
)
def built(request, tmp_path, monkeypatch):
    quam_class, lru_class = request.param
    return SimpleNamespace(
        machine=_build(quam_class, tmp_path / "state", monkeypatch), quam_class=quam_class, lru_class=lru_class
    )


def _snapshot(machine) -> str:
    return json.dumps(machine.to_dict(), sort_keys=True, default=str)


def test_element_plays_on_the_second_upconverter(built):
    machine = built.machine
    for name in QUBITS:
        qubit = machine.qubits[name]
        xy_lo = qubit.xy.LO_frequency
        add_lru_element(qubit)

        assert isinstance(qubit, built.lru_class)
        assert qubit.xy.LO_frequency == xy_lo
        assert qubit.lru.upconverter == LRU_UPCONVERTER
        # f0-g1 = f_02 - f_r, written out so the test does not reuse the function under test
        expected = 2 * qubit.xy.RF_frequency - abs(qubit.anharmonicity) - qubit.resonator.RF_frequency
        assert qubit.lru.RF_frequency == pytest.approx(expected)
        assert qubit.lru.intermediate_frequency == pytest.approx(LRU_NOMINAL_IF, abs=0.5e6)

    config = machine.generate_config()
    for name in QUBITS:
        assert config["elements"][f"{name}.lru"]["MWInput"]["upconverter"] == LRU_UPCONVERTER
        assert config["elements"][f"{name}.xy"]["MWInput"]["upconverter"] == 1


def _set_band_2_qubit(qubit, f_01=6.0676e9, anharmonicity=194.7e6, f_r=7.5239e9):
    """A qubit drive above band 1 keeps its port in band 2; the defaults put f0-g1 at 4.4166 GHz, below 4.5 GHz."""
    qubit.xy.RF_frequency = f_01
    qubit.anharmonicity = anharmonicity
    qubit.resonator.RF_frequency = f_r
    qubit.xy.opx_output.band = 2
    qubit.xy.opx_output.upconverter_frequency = 5.668e9


def test_pump_below_band_puts_upconverter_at_band_edge(built):
    qubit = built.machine.qubits["q1"]
    _set_band_2_qubit(qubit)
    addition = add_lru_element(qubit)

    band_low = MW_FEM_BANDS[2][0]
    expected_pump = 2 * qubit.xy.RF_frequency - abs(qubit.anharmonicity) - qubit.resonator.RF_frequency
    assert qubit.lru.RF_frequency == pytest.approx(expected_pump)
    assert qubit.lru.intermediate_frequency == pytest.approx(expected_pump - band_low, abs=1)
    assert addition.band_note is not None and "band 2" in addition.band_note

    config = built.machine.generate_config()
    upconverters = config["controllers"]["con1"]["fems"][1]["analog_outputs"][qubit.xy.opx_output.port_id][
        "upconverters"
    ]
    assert upconverters[LRU_UPCONVERTER]["frequency"] == pytest.approx(band_low)
    assert upconverters[1]["frequency"] == pytest.approx(5.668e9)


def test_pump_inside_band_keeps_nominal_if_and_no_note(built):
    qubit = built.machine.qubits["q1"]
    addition = add_lru_element(qubit)
    assert addition.band_note is None
    assert qubit.lru.intermediate_frequency == pytest.approx(LRU_NOMINAL_IF, abs=0.5e6)


def test_pump_out_of_reach_of_band_raises(built):
    qubit = built.machine.qubits["q1"]
    # f0-g1 = 2 * 6.0 - 0.2 - 7.9 = 3.9 GHz, 600 MHz below band 2
    _set_band_2_qubit(qubit, f_01=6.0e9, anharmonicity=200e6, f_r=7.9e9)
    before = _snapshot(built.machine)
    with pytest.raises(ValueError, match="out of reach"):
        add_lru_element(qubit)
    assert _snapshot(built.machine) == before


@pytest.mark.parametrize(
    "f_lru, band, expected",
    [
        (4.4166e9, 2, 4.5e9),  # below band 2: lower edge
        (5.0e9, 2, 4.9e9),  # inside band 2: nominal IF
        (7.55e9, 2, 7.45e9),  # nominal LO still inside
        (7.7e9, 2, 7.5e9),  # above band 2: upper edge
        (3.0e9, 1, 2.9e9),
        (6.5e9, 3, 6.5e9),
        (4.4166e9, None, 4.317e9),  # unknown band: nominal
    ],
)
def test_lru_upconverter_frequency(f_lru, band, expected):
    assert lru_upconverter_frequency(f_lru, band) == pytest.approx(expected)


def test_existing_element_is_left_alone(built):
    qubit = built.machine.qubits["q1"]
    add_lru_element(qubit)
    assert add_lru_element(qubit) is None


def test_remove_restores_the_state_exactly(built):
    machine = built.machine
    before = _snapshot(machine)
    additions = [add_lru_element(machine.qubits[name]) for name in QUBITS]
    for addition in additions:
        addition.remove()
    assert _snapshot(machine) == before


def test_kept_element_survives_save_and_load(built, tmp_path):
    machine = built.machine
    add_lru_element(machine.qubits["q1"])
    add_lru_element(machine.qubits["q2"]).remove()
    machine.save(tmp_path / "kept")

    reloaded = built.quam_class.load(tmp_path / "kept")
    q1 = reloaded.qubits["q1"]
    assert isinstance(q1, built.lru_class)
    assert q1.lru.RF_frequency == pytest.approx(machine.qubits["q1"].lru.RF_frequency)
    assert q1.xy.LO_frequency == machine.qubits["q1"].xy.LO_frequency
    assert getattr(reloaded.qubits["q2"], "lru", None) is None


def _stand_in_qubit():
    """f_01 = 4.5735 GHz, |anharmonicity| = 160.5 MHz, f_r = 5.91 GHz."""
    return SimpleNamespace(
        name="qtest",
        anharmonicity=-160.5e6,
        xy=SimpleNamespace(RF_frequency=4.5735e9),
        resonator=SimpleNamespace(RF_frequency=5.91e9),
        lru=SimpleNamespace(RF_frequency=1.497e9),
    )


def test_spurious_lines_in_and_near_the_window():
    qubit = _stand_in_qubit()
    lines = spurious_lines(qubit, 2.7e9, 3.4e9)
    assert [name for name, _ in lines] == ["f_r/2", "f_02/3"]
    assert [f for _, f in lines] == pytest.approx([2.955e9, 2.9955e9])
    assert spurious_lines(qubit, 3.0e9, 3.1e9) == []
    f_12 = 4.5735e9 - 160.5e6
    assert spurious_lines(qubit, f_12 - 400e6, f_12 - 100e6) == [("f_12", pytest.approx(f_12))]


def test_prediction_outside_window():
    qubit = _stand_in_qubit()
    assert prediction_outside_window(qubit, 2.9765e9, 3.1765e9) is None
    message = prediction_outside_window(qubit, 1.397e9, 1.597e9)
    assert "remove qubit.lru" in message


#: Detunings that push the XY IF past the 500 MHz reach, so the LO shift moves the XY upconverter.
FAR_DETUNINGS = np.arange(1.0e9, 1.2e9, 10e6).astype(int)


def _port_config(machine, qubit):
    port = qubit.xy.opx_output
    return machine.generate_config()["controllers"]["con1"]["fems"][port.fem_id]["analog_outputs"][port.port_id]


def test_lo_shift_moves_only_the_xy_upconverter_of_an_lru_port(built):
    machine = built.machine
    qubit = machine.qubits["q1"]
    add_lru_element(qubit)
    xy_lo, lru_lo, lru_rf = qubit.xy.LO_frequency, qubit.lru.LO_frequency, qubit.lru.RF_frequency
    before = _snapshot(machine)

    plan = plan_lo_shift_for_frequency_window([qubit], FAR_DETUNINGS)

    shift = plan.if_update[0]
    assert shift != 0
    assert qubit.xy.LO_frequency == pytest.approx(xy_lo + shift)
    assert qubit.lru.LO_frequency == pytest.approx(lru_lo)
    assert qubit.lru.RF_frequency == pytest.approx(lru_rf)
    port_config = _port_config(machine, qubit)
    assert "upconverter_frequency" not in port_config
    assert port_config["upconverters"][1]["frequency"] == pytest.approx(xy_lo + shift)
    assert port_config["upconverters"][LRU_UPCONVERTER]["frequency"] == pytest.approx(lru_lo)

    for tracked in plan.tracked_qubits:
        tracked.revert_changes()
    assert _snapshot(machine) == before


def test_lo_shift_of_a_single_upconverter_port_is_unchanged(built):
    machine = built.machine
    qubit = machine.qubits["q1"]
    xy_lo = qubit.xy.LO_frequency
    before = _snapshot(machine)

    plan = plan_lo_shift_for_frequency_window([qubit], FAR_DETUNINGS)

    assert plan.if_update[0] != 0
    assert qubit.xy.opx_output.upconverter_frequency == pytest.approx(xy_lo + plan.if_update[0])
    assert qubit.xy.opx_output.upconverters is None
    for tracked in plan.tracked_qubits:
        tracked.revert_changes()
    assert _snapshot(machine) == before
