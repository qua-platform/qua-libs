"""Offline regression checks; no connection to a QOP or persisted state updates."""

from contextlib import nullcontext
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import xarray as xr
from quam.components.pulses import SquareReadoutPulse

from calibration_utils.readout_duration_optimization import Parameters, GEFParameters
from calibration_utils.readout_duration_optimization.analysis import fit_raw_data, process_raw_dataset
from calibration_utils.readout_duration_optimization.experiment import duration_pulse, update_state
from calibration_utils.readout_duration_optimization.parameters import duration_values
from calibration_utils.readout_duration_optimization.plotting import plot_results


def make_data(gef=False):
    rng = np.random.default_rng(14)
    n_states = 3 if gef else 2
    # Each qubit has a different optimal duration. The last qubit has no valid data.
    durations = [200, 400, 600]
    iq = rng.normal(0, 0.0002, (3, 3, n_states, 400, 2))
    for q, strengths in enumerate(([1, 12, 1], [1, 1, 12])):
        for d, strength in enumerate(strengths):
            for state in range(n_states):
                iq[q, d, state, :, 0] += state * strength * 0.0002
                iq[q, d, state, :, 1] += (state == 2) * strength * 0.0002
    iq[2] = np.nan
    raw = xr.Dataset(
        {f"{quadrature}{state}": (("qubit", "duration", "n_runs"),
                                  iq[:, :, si, :, axis] * np.array(durations)[None, :, None] / 2**12)
         for si, state in enumerate("gef"[:n_states]) for axis, quadrature in enumerate("IQ")},
        coords={"qubit": ["q0", "q1", "q2"], "duration": durations, "n_runs": range(400)},
    )
    qubits = [SimpleNamespace(
        name=f"q{i}", grid_location=f"0,{i}",
        resonator=SimpleNamespace(operations={"readout": SquareReadoutPulse(length=1000, amplitude=0.1)},
                                 confusion_matrix=None, gef_centers=None),
    ) for i in range(3)]
    node = SimpleNamespace(parameters=(GEFParameters if gef else Parameters)(outliers_threshold=0.9),
                           namespace={"qubits": qubits}, record_state_updates=nullcontext)
    return raw, node


@pytest.mark.parametrize("gef", [False, True])
def test_selection_conversion_state_and_plots(gef):
    raw, node = make_data(gef)
    ds = process_raw_dataset(raw, node)
    np.testing.assert_allclose(ds.I.sel(state=0), raw.Ig * 2**12 / raw.duration)
    xr.testing.assert_identical(process_raw_dataset(ds, node), ds)
    fitted, blobs, results = fit_raw_data(ds, node)
    assert results["q0"].optimal_duration == 400
    assert results["q1"].optimal_duration == 600
    assert not results["q2"].success
    assert np.isnan(fitted.optimal_duration.sel(qubit="q2"))
    node.results = {"fit_results": {q: asdict(r) for q, r in results.items()}, "ds_iq_blobs": blobs}
    update_state(node)
    for index, duration in enumerate([400, 600]):
        q = node.namespace["qubits"][index]
        operation = q.resonator.operations[node.parameters.operation]
        assert operation.length == duration
        assert operation.integration_weights == [(1, duration)]
        np.testing.assert_allclose(np.sum(results[q.name].confusion_matrix, axis=1), 1)
        if gef:
            assert q.resonator.operations["readout"].length == 1000
            np.testing.assert_allclose(q.resonator.gef_centers, blobs.centers.sel(qubit=q.name) * duration / 2**12)
        else:
            assert operation.threshold == pytest.approx(results[q.name].ge_threshold * duration / 2**12)
    assert node.namespace["qubits"][2].resonator.operations["readout"].length == 1000
    if gef:
        assert "readout_GEF" not in node.namespace["qubits"][2].resonator.operations
    figures = plot_results(ds, node.namespace["qubits"], fitted, blobs)
    assert set(figures) == {"duration", "iq_blobs", "confusion_matrix"}
    import matplotlib.pyplot as plt
    plt.close("all")


def test_all_invalid_and_ties():
    raw, node = make_data()
    ds = process_raw_dataset(raw, node)
    # Duplicate the best shots at every duration; shortest duration must win.
    for quadrature in "IQ":
        ds[quadrature].loc[dict(qubit="q0")] = ds[quadrature].sel(qubit="q0").isel(duration=1)
    _, _, results = fit_raw_data(ds, node)
    assert results["q0"].optimal_duration == 200
    ds["I"][:] = np.nan
    _, blobs, results = fit_raw_data(ds, node)
    assert all(not result.success for result in results.values())
    assert np.isnan(blobs.centers).all()


def test_duration_parameters_and_weights():
    assert duration_values(Parameters()).tolist() == list(range(200, 2001, 200))
    for overrides in ({"min_duration_in_ns": 201}, {"duration_step_in_ns": 0},
                      {"max_duration_in_ns": 100}, {"max_duration_in_ns": 404}):
        with pytest.raises(ValueError):
            Parameters(**overrides)
    _, node = make_data()
    q = node.namespace["qubits"][0]
    q.resonator.operations["readout"].id = "original_readout"
    pulse = duration_pulse(q, "readout", 200)
    assert pulse.id is None  # Temporary config pulses must have distinct generated names.
    assert pulse.integration_weights == [(1, 200)]
    assert q.resonator.operations["readout"].length == 1000
    q.resonator.operations["readout"].integration_weights = None
    q.resonator.operations["readout"].integration_weights = [(1, 1000)]
    with pytest.raises(ValueError, match="default integration weights"):
        duration_pulse(q, "readout", 200)
