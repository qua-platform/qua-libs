"""Tests of the flux-activated f-LRU helpers: the run-only config element and the dataset coordinates."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.lru_calibration.analysis import assign_flux_lru_coords  # noqa: E402
from calibration_utils.lru_calibration.flux_element import (  # noqa: E402
    FLUX_LRU_OPERATION,
    FLUX_LRU_SAMPLING_RATE,
    add_flux_lru_elements,
    flux_lru_element,
    flux_tone_headroom_v,
)


def _config():
    """Minimal config with one qubit flux element holding a 1 GS/s arbitrary pulse."""
    return {
        "controllers": {
            "con1": {
                "fems": {5: {"analog_outputs": {1: {"sampling_rate": 1e9, "upsampling_mode": "pulse", "offset": 0.0}}}}
            }
        },
        "elements": {"q1.z": {"singleInput": {"port": ("con1", 5, 1)}, "operations": {"cz": "q1.z.cz.pulse"}}},
        "pulses": {"q1.z.cz.pulse": {"operation": "control", "length": 4, "waveforms": {"single": "q1.z.cz.wf"}}},
        "waveforms": {"q1.z.cz.wf": {"type": "arbitrary", "samples": [0.1] * 4}},
    }


def test_flux_lru_element_is_added_at_2_gsps_without_the_1_gsps_pulses():
    qubit = SimpleNamespace(name="q1", z=SimpleNamespace(name="q1.z"))
    config = add_flux_lru_elements(_config(), [qubit], amplitude_v=1.2)

    port = config["controllers"]["con1"]["fems"][5]["analog_outputs"][1]
    assert port["sampling_rate"] == FLUX_LRU_SAMPLING_RATE
    assert "upsampling_mode" not in port
    assert config["elements"]["q1.z"]["operations"] == {}
    assert "q1.z.cz.pulse" not in config["pulses"] and "q1.z.cz.wf" not in config["waveforms"]

    element = config["elements"][flux_lru_element(qubit)]
    assert element["singleInput"] == {"port": ("con1", 5, 1)}
    pulse = config["pulses"][element["operations"][FLUX_LRU_OPERATION]]
    assert config["waveforms"][pulse["waveforms"]["single"]] == {"type": "constant", "sample": 1.2}


def test_flux_lru_coords_are_volts_and_f0e1_frequency():
    ds = xr.Dataset(
        {"pf": (("qubit", "amp_prefactor", "detuning"), np.zeros((2, 3, 4)))},
        coords={"qubit": ["q1", "q2"], "amp_prefactor": [0.0, 0.5, 1.0], "detuning": [-2e6, -1e6, 0, 1e6]},
    )
    ds = assign_flux_lru_coords(ds, ["q1", "q2"], amplitude_v=2.0, centres=np.array([1.5e9, 1.6e9]))
    np.testing.assert_allclose(ds.amplitude.sel(qubit="q2").values, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(ds.frequency.sel(qubit="q2").values, 1.6e9 + ds.detuning.values)
    assert ds.amplitude.attrs["units"] == "V"
    assert ds.attrs["lru_transition"] == "f0e1"


def test_flux_pulses_shared_with_other_elements_are_kept():
    qubit = SimpleNamespace(name="q1", z=SimpleNamespace(name="q1.z"))
    config = _config()
    config["controllers"]["con1"]["fems"][5]["analog_outputs"][2] = {"sampling_rate": 1e9, "offset": 0.0}
    config["elements"]["q2.z"] = {"singleInput": {"port": ("con1", 5, 2)}, "operations": {"spectator": "q1.z.cz.pulse"}}
    config = add_flux_lru_elements(config, [qubit], amplitude_v=0.2)

    assert config["elements"]["q1.z"]["operations"] == {}
    assert config["elements"]["q2.z"]["operations"] == {"spectator": "q1.z.cz.pulse"}
    assert "q1.z.cz.pulse" in config["pulses"] and "q1.z.cz.wf" in config["waveforms"]


def test_flux_tone_headroom_depends_on_output_mode_and_offset():
    def qubit(mode):
        return SimpleNamespace(z=SimpleNamespace(opx_output=SimpleNamespace(output_mode=mode), joint_offset=-0.1))

    assert flux_tone_headroom_v(qubit("direct")) == pytest.approx(0.4)
    assert flux_tone_headroom_v(qubit("amplified")) == pytest.approx(2.4)
