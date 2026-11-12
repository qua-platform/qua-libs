"""Round-trip test of the f-LRU calibration analysis.

The data is synthesised from a known resonance line, following the model the analysis inverts:

    detuning(A) = d0 + c2 * A^2,          the |f0> -> |g1> line moves with the AC Stark shift,
    P(f)        = P0 * (1 - L(df) * sin^2(pi * A / (2 * A_pi))),

where L is a unit Lorentzian around the line and A_pi the amplitude transferring all of |f> in
the pulse. The fitted line must recover d0 and c2, and the optimum must sit at A_pi with the
longer duration (smaller A_pi) chosen over the shorter one.
"""

import sys
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.lru_calibration.analysis import (  # noqa: E402
    fit_lru_map,
    fit_raw_data,
    log_fitted_results,
    lru_transition,
    process_raw_dataset,
)
from calibration_utils.lru_calibration.parameters import NodeSpecificParameters, get_detunings  # noqa: E402

# At the 200 ns optimum (A = 0.5) the line sits at D0 + C2 / 4 = +2 MHz, on a sample of the 2 MHz grid:
# near full transfer P(f) is flat in amplitude, so an off-grid line would let a neighbouring amplitude win.
D0 = -13e6
C2 = 60e6
WIDTH = 6e6
P0 = 0.9
RF = 806e6


def _pf_map(detuning, amplitude, a_pi, noise=0.0, seed=0):
    line = D0 + C2 * amplitude**2
    lorentz = 1 / (1 + ((detuning[None, :] - line[:, None]) / (WIDTH / 2)) ** 2)
    transfer = np.sin(np.pi * amplitude / (2 * a_pi)) ** 2
    pf = P0 * (1 - lorentz * transfer[:, None])
    return pf + np.random.default_rng(seed).normal(0, noise, pf.shape)


def test_fit_lru_map_recovers_line_and_optimum():
    detuning = np.arange(-100e6, 100e6 + 1, 1e6)
    amplitude = np.linspace(0, 0.95, 40)
    fit = fit_lru_map(detuning, amplitude, _pf_map(detuning, amplitude, a_pi=0.6, noise=0.01), num_shots=1000)

    assert fit["success"]
    assert fit["d0"] == pytest.approx(D0, abs=1e6)
    assert fit["c2"] == pytest.approx(C2, rel=0.05)
    # With noise, the flat top of the transfer lets any amplitude near A_pi win: require a near-full transfer.
    assert np.sin(np.pi * amplitude[fit["opt_index"]] / (2 * 0.6)) ** 2 > 0.95
    assert fit["pf_min"] < 0.1


def test_fit_lru_map_fails_without_resonance():
    detuning = np.arange(-100e6, 100e6 + 1, 1e6)
    amplitude = np.linspace(0, 0.95, 40)
    flat = np.full((len(amplitude), len(detuning)), P0)
    fit = fit_lru_map(detuning, amplitude, flat, num_shots=1000)

    assert not fit["success"]
    assert np.isnan(fit["c2"])
    assert "no resonance" in fit["failure_reason"]


def test_unprepared_f_state_fails_without_fitting():
    # |f> never prepared, P(f) sat at ~0.16 everywhere with 100 shots per point.
    detuning = np.arange(-100e6, 100e6 + 1, 2e6)
    amplitude = np.linspace(0, 0.96, 33)
    pf = np.random.default_rng(1).binomial(100, 0.16, (len(amplitude), len(detuning))) / 100
    fit = fit_lru_map(detuning, amplitude, pf, num_shots=100)

    assert not fit["success"]
    assert "not prepared" in fit["failure_reason"]


def test_dip_within_shot_noise_is_rejected():
    # |f> well prepared but no LRU effect: the lowest point on any line is just a noise excursion.
    detuning = np.arange(-100e6, 100e6 + 1, 2e6)
    amplitude = np.linspace(0, 0.96, 33)
    pf = np.random.default_rng(2).binomial(100, 0.6, (len(amplitude), len(detuning))) / 100
    fit = fit_lru_map(detuning, amplitude, pf, num_shots=100, min_contrast=0.05)

    assert not fit["success"]
    assert "shot noise" in fit["failure_reason"]


def _two_duration_dataset(transition=None):
    """One qubit, two durations; the longer pulse needs less amplitude for a full transfer."""
    params = NodeSpecificParameters(frequency_span_in_mhz=200, frequency_step_in_mhz=2)
    detuning = get_detunings(params)
    amp_prefactor = np.arange(0, 1.9, 0.05)
    base_amp = 0.5
    amplitude = base_amp * amp_prefactor
    durations = np.array([48, 200])
    # At 48 ns the transfer is incomplete.
    a_pis = {48: 1.5, 200: 0.5}
    pf = np.stack([_pf_map(detuning.astype(float), amplitude, a_pis[d]) for d in durations])[None]
    dims = ("qubit", "duration", "amp_prefactor", "detuning")
    ds = xr.Dataset(
        {"pf": (dims, pf), "pe": (dims, P0 - pf)},
        coords={"qubit": ["q1"], "duration": durations, "amp_prefactor": amp_prefactor, "detuning": detuning},
    )
    ds = ds.assign_coords(
        amplitude=(("qubit", "amp_prefactor"), amplitude[None]),
        frequency=(("qubit", "detuning"), RF + detuning[None], {"long_name": "LRU drive frequency", "units": "Hz"}),
    )
    if transition is not None:
        ds.attrs["lru_transition"] = transition
    return ds


NODE = SimpleNamespace(
    parameters=SimpleNamespace(num_shots=1000, min_ridge_contrast=0.2, min_prepared_pf=0.5, min_dip_sigma=5.0)
)


def test_fit_raw_data_selects_best_duration():
    ds_fit, fit_results = fit_raw_data(_two_duration_dataset(), NODE)
    result = fit_results["q1"]

    assert result.success
    assert result.duration_ns == 200
    assert result.amplitude == pytest.approx(0.5, abs=0.03)
    assert result.frequency_zero_amplitude == pytest.approx(RF + D0, abs=1e6)
    assert result.frequency == pytest.approx(RF + D0 + C2 * result.amplitude**2, abs=1e6)
    assert ds_fit.pf_line.dims == ("qubit", "duration", "amp_prefactor")
    np.testing.assert_allclose(ds_fit.pg_line.values, 1 - ds_fit.pe_line.values - ds_fit.pf_line.values)
    assert result.transition == "f0g1"  # untagged data is read as the legacy transition


def test_tagged_dataset_carries_its_transition_to_the_fit_and_the_log():
    _, fit_results = fit_raw_data(_two_duration_dataset("f0g1"), NODE)
    assert fit_results["q1"].transition == "f0g1"

    lines = []
    log_fitted_results({k: asdict(v) for k, v in fit_results.items()}, log_callable=lines.append)
    assert "|f0> -> |g1> at zero amplitude" in lines[0]


def test_lru_transition_falls_back_to_f0g1():
    assert lru_transition(xr.Dataset()) == "f0g1"
    assert lru_transition(xr.Dataset(attrs={"lru_transition": "f1g2"})) == "f0g1"
    assert lru_transition(xr.Dataset(attrs={"lru_transition": "f0g1"})) == "f0g1"


def test_process_raw_dataset_keeps_the_transition_tag():
    ds = process_raw_dataset(_two_duration_dataset("f0g1"), NODE)
    assert ds.attrs["lru_transition"] == "f0g1"


def test_transition_tag_survives_qualibrate_save_and_load(tmp_path):
    # Private QUAlibrate paths: these are the saver and loader node.save() and load_from_id use.
    from qualibrate.core.storage._data_handler.data_processors import XarraySaver
    from qualibrate.core.utils.node.loaders import XarrayLoader

    saver = XarraySaver()
    saver.process({"ds_raw": _two_duration_dataset("f0g1")})
    saver.post_process(tmp_path)
    loaded = XarrayLoader().load(tmp_path / "ds_raw.h5")
    assert loaded.attrs["lru_transition"] == "f0g1"


@pytest.mark.parametrize("transition, title", [("f0g1", "|f0> -> |g1>: P(g)"), (None, "|f0> -> |g1>: P(g)")])
def test_lru_map_shows_the_population_of_the_transition(transition, title):
    from calibration_utils.lru_calibration.plotting import plot_lru_maps

    ds = process_raw_dataset(_two_duration_dataset(transition), NODE)
    ds_fit, _ = fit_raw_data(ds, NODE)
    qubits = [SimpleNamespace(grid_location="0,0")]
    fig = plot_lru_maps(ds, qubits, ds_fit, 200)
    assert title in fig._suptitle.get_text()
    plt.close(fig)


@pytest.mark.parametrize("population, label", [("pe", "P(e)"), ("pf", "P(f)")])
def test_lru_map_can_show_the_other_populations(population, label):
    from calibration_utils.lru_calibration.plotting import plot_lru_maps

    ds = process_raw_dataset(_two_duration_dataset("f0g1"), NODE)
    ds_fit, _ = fit_raw_data(ds, NODE)
    qubits = [SimpleNamespace(grid_location="0,0")]
    fig = plot_lru_maps(ds, qubits, ds_fit, 200, population)
    assert f"|f0> -> |g1>: {label}" in fig._suptitle.get_text()
    plt.close(fig)


def test_iq_classification_recovers_populations_and_keeps_sparse_centers():
    # Raw shots of an IQ debug run: 79% g, 2% e, 19% f around slightly offset blobs.
    from calibration_utils.lru_calibration.analysis import classify_iq_shots  # noqa: E402

    stored = np.array([[-2.7e-4, 3.1e-4], [-7.1e-4, 0.0], [-1.12e-3, 5.7e-4]])
    true = stored + np.array([[4e-5, 6e-5], [0.0, 0.0], [4e-5, 1.1e-4]])
    rng = np.random.default_rng(3)
    state = rng.choice(3, size=(1, 400, 1, 3, 20), p=[0.79, 0.02, 0.19])
    noise = rng.normal(0, 6e-5, state.shape + (2,))
    dims = ("qubit", "shot", "duration", "amp_prefactor", "detuning")
    ds = xr.Dataset(
        {"I": (dims, true[state, 0] + noise[..., 0]), "Q": (dims, true[state, 1] + noise[..., 1])},
        coords={
            "qubit": ["q1"],
            "shot": np.arange(400),
            "duration": [48],
            "amp_prefactor": [0, 0.06, 0.12],
            "detuning": np.arange(20),
        },
    )
    qubit = SimpleNamespace(name="q1", resonator=SimpleNamespace(gef_centers=stored.tolist()))

    ds, info = classify_iq_shots(ds, [qubit])

    assert info["q1"]["populations_refit"] == pytest.approx([0.79, 0.02, 0.19], abs=0.01)
    assert info["q1"]["centers_refit"][1] == stored[1].tolist()  # too few e shots: the e center stays stored
    assert float((ds.state == state[0]).mean()) > 0.99
    assert ds.pf.dims == ("qubit", "duration", "amp_prefactor", "detuning")
