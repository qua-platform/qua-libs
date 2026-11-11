"""Tests of the flux-activated f-LRU: the stored ``qubit.flux_lru``, the 2 GS/s config helper and the dataset coordinates."""

import logging
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lru_machine import PlainFluxQuam, build_machine  # noqa: E402
from qm import QuantumMachinesManager  # noqa: E402
from qm.program._qua_config_schema import load_config  # noqa: E402
from quam.components.pulses import WaveformPulse  # noqa: E402

from calibration_utils.lru_calibration.analysis import assign_flux_lru_coords  # noqa: E402
from quam_config import Quam  # noqa: E402
from quam_config.flux_lru_config import (  # noqa: E402
    FLUX_LRU_OPERATION,
    FLUX_LRU_SAMPLING_RATE,
    UPSAMPLED_SUFFIX,
    add_flux_lru_elements,
    flux_lru_element,
    flux_tone_headroom_v,
    upsample_linear,
)
from quam_config.lru_components import FluxLru, LruFluxTunableTransmon, upgrade_to_lru_class  # noqa: E402

CZ_SAMPLES = [0.0, 0.1, 0.3, 0.3, 0.1, 0.0, 0.0, 0.0]


def _config():
    """Minimal config: q1 and q2 flux elements on their own ports, q1 holding a 1 GS/s arbitrary pulse."""
    port = {"sampling_rate": 1e9, "upsampling_mode": "pulse", "offset": 0.0}
    return {
        "controllers": {"con1": {"fems": {5: {"analog_outputs": {1: dict(port), 2: dict(port)}}}}},
        "elements": {
            "q1.z": {"singleInput": {"port": ("con1", 5, 1)}, "operations": {"cz": "q1.z.cz.pulse"}},
            "q2.z": {"singleInput": {"port": ("con1", 5, 2)}, "operations": {"spectator": "q1.z.cz.pulse"}},
        },
        "pulses": {
            "q1.z.cz.pulse": {"operation": "control", "length": len(CZ_SAMPLES), "waveforms": {"single": "q1.z.cz.wf"}}
        },
        "waveforms": {"q1.z.cz.wf": {"type": "arbitrary", "samples": list(CZ_SAMPLES)}},
    }


def _stand_in(name="q1", joint_offset=0.0, flux_lru=None):
    return SimpleNamespace(
        name=name,
        z=SimpleNamespace(
            name=f"{name}.z", joint_offset=joint_offset, opx_output=SimpleNamespace(output_mode="direct")
        ),
        flux_lru=flux_lru,
    )


def test_upsample_linear_inserts_midpoints_and_holds_the_last_sample():
    assert upsample_linear([0.0, 0.2, 0.4]) == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.4])
    upsampled = upsample_linear(CZ_SAMPLES)
    assert len(upsampled) == 2 * len(CZ_SAMPLES)
    # Each 2 GS/s sample lasts half as long: the area is kept up to the half-sample interpolation error
    assert np.sum(upsampled) / 2 == pytest.approx(np.sum(CZ_SAMPLES), abs=0.05)


def test_flux_port_runs_at_2_gsps_with_upsampled_pulses():
    config = add_flux_lru_elements(_config(), [_stand_in()], amplitude_v=0.2, tone_frequency=750e6, length_ns=100)

    port = config["controllers"]["con1"]["fems"][5]["analog_outputs"][1]
    assert port["sampling_rate"] == FLUX_LRU_SAMPLING_RATE
    assert "upsampling_mode" not in port
    pulse_name = config["elements"]["q1.z"]["operations"]["cz"]
    assert pulse_name == "q1.z.cz.pulse" + UPSAMPLED_SUFFIX
    pulse = config["pulses"][pulse_name]
    assert pulse["length"] == len(CZ_SAMPLES)  # ns, unchanged
    assert config["waveforms"][pulse["waveforms"]["single"]]["samples"] == upsample_linear(CZ_SAMPLES)

    element = config["elements"][flux_lru_element(_stand_in())]
    assert element["singleInput"] == {"port": ("con1", 5, 1)}
    assert element["intermediate_frequency"] == 750e6
    flux_pulse = config["pulses"][element["operations"][FLUX_LRU_OPERATION]]
    assert flux_pulse["length"] == 100
    assert config["waveforms"][flux_pulse["waveforms"]["single"]] == {"type": "constant", "sample": 0.2}


def test_pulse_shared_with_a_1_gsps_port_is_left_untouched():
    config = add_flux_lru_elements(_config(), [_stand_in()], amplitude_v=0.2, tone_frequency=750e6, length_ns=100)

    assert config["controllers"]["con1"]["fems"][5]["analog_outputs"][2]["sampling_rate"] == 1e9
    assert config["elements"]["q2.z"]["operations"] == {"spectator": "q1.z.cz.pulse"}
    assert config["waveforms"]["q1.z.cz.wf"]["samples"] == CZ_SAMPLES


def test_stored_flux_lru_values_are_used():
    stored = SimpleNamespace(amplitude_v=0.15, tone_frequency=740e6, duration_ns=480)
    config = add_flux_lru_elements(_config(), [_stand_in(flux_lru=stored)])

    element = config["elements"]["q1.flux_lru"]
    assert element["intermediate_frequency"] == 740e6
    pulse = config["pulses"][element["operations"][FLUX_LRU_OPERATION]]
    assert pulse["length"] == 480
    assert config["waveforms"][pulse["waveforms"]["single"]]["sample"] == 0.15


def test_missing_calibration_headroom_and_nyquist_are_refused():
    with pytest.raises(ValueError, match="run node 26b"):
        add_flux_lru_elements(_config(), [_stand_in()])
    with pytest.raises(ValueError, match="does not fit"):
        add_flux_lru_elements(
            _config(), [_stand_in(joint_offset=0.4)], amplitude_v=0.2, tone_frequency=750e6, length_ns=100
        )
    with pytest.raises(ValueError, match="Nyquist"):
        add_flux_lru_elements(_config(), [_stand_in()], amplitude_v=0.2, tone_frequency=960e6, length_ns=100)


def test_fir_taps_get_a_warning(caplog):
    config = _config()
    config["controllers"]["con1"]["fems"][5]["analog_outputs"][1]["filter"] = {"feedforward": [1.1, -0.1]}
    with caplog.at_level(logging.WARNING):
        add_flux_lru_elements(config, [_stand_in()], amplitude_v=0.2, tone_frequency=750e6, length_ns=100)
    assert "feedforward" in caplog.text


def test_flux_tone_headroom_depends_on_output_mode_and_offset():
    def qubit(mode):
        return SimpleNamespace(z=SimpleNamespace(opx_output=SimpleNamespace(output_mode=mode), joint_offset=-0.1))

    assert flux_tone_headroom_v(qubit("direct")) == pytest.approx(0.4)
    assert flux_tone_headroom_v(qubit("amplified")) == pytest.approx(2.4)


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


# The repo's Quam, built and saved like a real state


def _calibrate(qubit):
    qubit.flux_lru = FluxLru(transition_frequency=1.49e9, amplitude_v=0.12, duration_ns=500)


def test_new_states_get_the_lru_class_and_an_unchanged_default_config(tmp_path, monkeypatch):
    machine = build_machine(Quam, tmp_path / "state", monkeypatch)
    assert all(type(q) is LruFluxTunableTransmon for q in machine.qubits.values())
    config = machine.generate_config()
    assert not any(name.endswith(".flux_lru") for name in config["elements"])
    port = config["controllers"]["con1"]["fems"][3]["analog_outputs"]
    assert all(p["sampling_rate"] == 1e9 for p in port.values())


def test_generate_config_with_flux_lru_qubits(tmp_path, monkeypatch):
    machine = build_machine(Quam, tmp_path / "state", monkeypatch)
    q1 = machine.qubits["q1"]
    _calibrate(q1)
    q1.z.operations["cz"] = WaveformPulse(waveform_I=list(CZ_SAMPLES))

    config = machine.generate_config(flux_lru_qubits=["q1"])

    # The offline schema checks the config's structure; it does not compare sample counts with the sampling
    # rate (the QOP does), so the sample count is asserted below. It needs QOP capabilities, here all of them.
    QuantumMachinesManager.set_capabilities_offline()
    load_config(config)
    z_port = q1.z.opx_output
    ports = config["controllers"]["con1"]["fems"][z_port.fem_id]["analog_outputs"]
    assert ports[z_port.port_id]["sampling_rate"] == FLUX_LRU_SAMPLING_RATE
    q2_port = machine.qubits["q2"].z.opx_output
    assert ports[q2_port.port_id]["sampling_rate"] == 1e9
    cz = config["pulses"][config["elements"]["q1.z"]["operations"]["cz"]]
    assert len(config["waveforms"][cz["waveforms"]["single"]]["samples"]) == 2 * len(CZ_SAMPLES)
    assert config["elements"][q1.flux_lru.element_name]["intermediate_frequency"] == pytest.approx(745e6)
    # The state itself keeps the port at 1 GS/s
    assert z_port.sampling_rate == 1e9


def test_flux_lru_survives_save_and_load(tmp_path, monkeypatch):
    machine = build_machine(Quam, tmp_path / "state", monkeypatch)
    _calibrate(machine.qubits["q1"])
    machine.save(tmp_path / "saved")

    reloaded = Quam.load(tmp_path / "saved")
    flux_lru = reloaded.qubits["q1"].flux_lru
    assert (flux_lru.transition_frequency, flux_lru.amplitude_v, flux_lru.duration_ns) == (1.49e9, 0.12, 500)
    assert flux_lru.tone_frequency == pytest.approx(745e6)
    assert reloaded.qubits["q2"].flux_lru is None


def test_qubit_from_an_older_state_is_upgraded_before_storing(tmp_path, monkeypatch):
    old = build_machine(PlainFluxQuam, tmp_path / "state", monkeypatch)
    old.save(tmp_path / "old")
    machine = Quam.load(tmp_path / "old")
    qubit = machine.qubits["q1"]
    assert type(qubit).__name__ == "FluxTunableTransmon"  # QUAM keeps the class the state names

    assert upgrade_to_lru_class(qubit).__name__ == "FluxTunableTransmon"
    assert upgrade_to_lru_class(qubit) is LruFluxTunableTransmon  # already upgraded: unchanged
    _calibrate(qubit)
    machine.save(tmp_path / "upgraded")

    reloaded = Quam.load(tmp_path / "upgraded")
    assert type(reloaded.qubits["q1"]) is LruFluxTunableTransmon
    assert reloaded.qubits["q1"].flux_lru.duration_ns == 500
    assert type(reloaded.qubits["q2"]).__name__ == "FluxTunableTransmon"
