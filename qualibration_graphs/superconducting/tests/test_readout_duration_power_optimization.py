"""Tests for the joint readout duration x power optimization (node 08c).

Four groups:

* ``blob_statistics`` -- the per-grid-point fit. It must report a high fidelity for two
  clean, well separated blobs, and it must expose the *smeared blob* failure through the
  variance ratio: the characteristic high-power failure is an excited blob that spreads
  into an arc while the ground blob stays tight, which leaves the non-outlier fraction
  looking healthy while the fidelity number becomes meaningless.
* ``select_operating_point`` -- the gate plus the argmax. The point of the gate is that a
  higher-fidelity but ungated point must lose to a lower-fidelity gated one, and that a
  grid with nothing eligible fails loudly, naming which gate rejected everything.
* ``volts_per_duration`` -- the normalization. Every duration slice carries its own
  integration length, so a single per-qubit divisor (what ``convert_IQ_to_V`` would apply)
  is wrong; the test pins the per-slice conversion against hand arithmetic.
* the parameter validator -- the 4 ns chunk grid that ``demod.accumulated`` requires.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibration_utils.readout_duration_power_optimization.analysis import (  # noqa: E402
    blob_statistics,
    select_operating_point,
    to_volts_per_duration,
)
from calibration_utils.readout_duration_power_optimization.qua_sequence import (  # noqa: E402
    has_custom_integration_weights,
    readout_config_override,
)
from calibration_utils.readout_duration_power_optimization.parameters import (  # noqa: E402
    NodeSpecificParameters,
    get_chunk_duration_in_ns,
    get_durations_in_ns,
    get_samples_per_chunk,
)

AMPS = np.linspace(0.5, 1.99, 6)
DURATIONS = np.array([200.0, 400.0, 600.0, 800.0])


def _blobs(separation: float, width_g: float = 1.0, width_e: float = 1.0, n: int = 800, seed: int = 0):
    """Two IQ blobs along I, returned with the (state, n_runs) shape the fit expects."""
    rng = np.random.default_rng(seed)
    I = np.stack([rng.normal(0.0, width_g, n), rng.normal(separation, width_e, n)])
    Q = np.stack([rng.normal(0.0, width_g, n), rng.normal(0.0, width_e, n)])
    return I, Q


def _grid(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=["amp_prefactor", "duration"],
        coords={"amp_prefactor": AMPS, "duration": DURATIONS},
    )


# ------------------------------------------------------------------- blob_statistics


def test_blob_statistics_reports_high_fidelity_for_separated_blobs():
    fidelity, non_outlier, variance_ratio = blob_statistics(*_blobs(separation=10.0))
    assert fidelity > 0.99
    assert non_outlier > 0.98
    assert variance_ratio == pytest.approx(1.0, abs=0.5)


def test_blob_statistics_reports_low_fidelity_for_overlapping_blobs():
    fidelity, _, _ = blob_statistics(*_blobs(separation=0.2))
    assert fidelity < 0.7


def test_blob_statistics_exposes_a_smeared_excited_blob_through_the_variance_ratio():
    """The high-power failure: one blob spreads while the other stays tight."""
    _, _, variance_ratio = blob_statistics(*_blobs(separation=10.0, width_g=1.0, width_e=6.0))
    assert variance_ratio > 3.0


def test_blob_statistics_variance_ratio_is_at_least_one():
    """It is defined as wider over narrower, so it can never come back below 1."""
    for width_e in (0.2, 1.0, 5.0):
        _, _, variance_ratio = blob_statistics(*_blobs(separation=10.0, width_e=width_e))
        assert variance_ratio >= 1.0


# -------------------------------------------------------------- select_operating_point


def test_select_operating_point_takes_the_global_argmax_when_everything_is_eligible():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    fidelity[2, 1] = 0.97
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert point.success
    assert point.amp_prefactor == pytest.approx(AMPS[2])
    assert point.duration == pytest.approx(DURATIONS[1])
    assert point.fidelity == pytest.approx(0.97)


def test_select_operating_point_skips_a_better_point_that_fails_the_outlier_gate():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    fidelity[5, 3] = 0.99  # best, but outlier-ridden
    fidelity[1, 2] = 0.95  # best among the eligible
    non_outlier = np.full_like(fidelity, 0.99)
    non_outlier[5, 3] = 0.80
    point = select_operating_point(
        _grid(fidelity),
        _grid(non_outlier),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert point.success
    assert point.fidelity == pytest.approx(0.95)
    assert point.amp_prefactor == pytest.approx(AMPS[1])


def test_select_operating_point_skips_a_better_point_whose_blobs_are_smeared():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    fidelity[5, 3] = 0.99
    fidelity[1, 2] = 0.95
    variance_ratio = np.ones_like(fidelity)
    variance_ratio[5, 3] = 8.0
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(variance_ratio),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert point.success
    assert point.fidelity == pytest.approx(0.95)


def test_select_operating_point_fails_and_names_the_outlier_gate():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.5)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert not point.success
    assert "non-outlier fraction" in point.note
    assert np.isnan(point.fidelity)


def test_select_operating_point_fails_and_names_the_variance_gate():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.full_like(fidelity, 9.0)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert not point.success
    assert "variance ratio" in point.note


def test_select_operating_point_names_both_gates_when_both_reject_everything():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.5)),
        _grid(np.full_like(fidelity, 9.0)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert not point.success
    assert "non-outlier fraction" in point.note
    assert "variance ratio" in point.note


# ------------------------------------------------------------------ volts conversion


def test_to_volts_per_duration_divides_each_slice_by_its_own_duration():
    raw = xr.Dataset(
        {"Ig": (("duration",), np.array([1.0, 1.0, 1.0, 1.0]))},
        coords={"duration": DURATIONS},
    )
    converted = to_volts_per_duration(raw, ["Ig"])
    np.testing.assert_allclose(converted.Ig.values, 2**12 / DURATIONS)


def test_to_volts_per_duration_leaves_other_variables_untouched():
    raw = xr.Dataset(
        {
            "Ig": (("duration",), np.ones(DURATIONS.size)),
            "keep_me": (("duration",), np.arange(DURATIONS.size, dtype=float)),
        },
        coords={"duration": DURATIONS},
    )
    converted = to_volts_per_duration(raw, ["Ig"])
    np.testing.assert_allclose(converted.keep_me.values, np.arange(DURATIONS.size))


def test_to_volts_per_duration_is_not_a_single_shared_divisor():
    """The whole reason this exists rather than ``convert_IQ_to_V``: no shared length."""
    raw = xr.Dataset(
        {"Ig": (("duration",), np.ones(DURATIONS.size))},
        coords={"duration": DURATIONS},
    )
    converted = to_volts_per_duration(raw, ["Ig"])
    assert len(set(np.round(converted.Ig.values, 12))) == DURATIONS.size


# -------------------------------------------------------------------- chunk grid


def test_durations_land_on_the_4ns_grid_for_the_defaults():
    params = NodeSpecificParameters()
    durations = get_durations_in_ns(params)
    assert durations[-1] == params.max_duration_in_ns
    assert len(durations) == params.num_durations
    assert np.all(durations % 4 == 0)
    assert get_samples_per_chunk(params) * 4 == get_chunk_duration_in_ns(params)


def test_parameters_reject_a_chunk_duration_off_the_4ns_grid():
    """2000 ns over 8 durations is 250 ns, which demod.accumulated cannot tile."""
    with pytest.raises(ValueError, match="multiple of 4 ns"):
        NodeSpecificParameters(max_duration_in_ns=2000, num_durations=8)


def test_parameters_accept_a_chunk_duration_on_the_4ns_grid():
    params = NodeSpecificParameters(max_duration_in_ns=1600, num_durations=8)
    assert get_chunk_duration_in_ns(params) == 200


# ------------------------------------------------------------------- config override


class _FakePulse:
    """A pulse whose ``integration_weights`` behaves like quam's: reads resolve a reference,
    and overwriting one in place is refused until it has been cleared to None."""

    DEFAULT_RESOLVED = [(1, 1000)]

    def __init__(self, length, raw_weights):
        self.length = length
        self._raw = raw_weights

    @property
    def integration_weights(self):
        if self._raw == "#./default_integration_weights":
            return self.DEFAULT_RESOLVED
        return self._raw

    @integration_weights.setter
    def integration_weights(self, value):
        if value is not None and isinstance(self._raw, str) and self._raw.startswith("#"):
            raise ValueError("Cannot set attribute integration_weights because it is a reference.")
        self._raw = value

    def get_unreferenced_value(self, name):
        assert name == "integration_weights"
        return self._raw


class _FakeQubit:
    def __init__(self, name, pulse):
        self.name = name
        self.resonator = type("R", (), {"operations": {"readout": pulse}})()


def test_readout_config_override_lengthens_the_pulse_and_restores_it():
    pulse = _FakePulse(500, "#./default_integration_weights")
    qubit = _FakeQubit("q1", pulse)
    with readout_config_override([qubit], "readout", 2000):
        assert pulse.length == 2000
    assert pulse.length == 500
    assert pulse.get_unreferenced_value("integration_weights") == "#./default_integration_weights"


def test_readout_config_override_restores_custom_weights_it_replaced():
    """The restore path must survive quam refusing to overwrite a reference in place."""
    custom = [(0.5, 250), (1.0, 250)]
    pulse = _FakePulse(500, custom)
    qubit = _FakeQubit("q1", pulse)
    with readout_config_override([qubit], "readout", 2000):
        assert pulse.get_unreferenced_value("integration_weights") == "#./default_integration_weights"
    assert pulse.length == 500
    assert pulse.get_unreferenced_value("integration_weights") == custom


def test_readout_config_override_restores_after_a_failure_inside_the_block():
    pulse = _FakePulse(500, [(1.0, 500)])
    qubit = _FakeQubit("q1", pulse)
    with pytest.raises(RuntimeError):
        with readout_config_override([qubit], "readout", 2000):
            raise RuntimeError("config generation blew up")
    assert pulse.length == 500
    assert pulse.get_unreferenced_value("integration_weights") == [(1.0, 500)]


def test_has_custom_integration_weights_sees_through_the_resolved_reference():
    """Reading `.integration_weights` resolves the default reference into a list, so a naive
    comparison against the reference string never matches. This is the bug the check exists for."""
    default_pulse = _FakePulse(1000, "#./default_integration_weights")
    assert default_pulse.integration_weights == [(1, 1000)]  # resolved, not the reference
    assert not has_custom_integration_weights(default_pulse)
    assert has_custom_integration_weights(_FakePulse(1000, [(0.5, 500), (1.0, 500)]))


def test_select_operating_point_breaks_ties_towards_the_cheapest_point():
    """Fidelity is quantised, so the saturated region is full of exact ties. A tie must go to
    the lowest amplitude and then the shortest duration, not to an arbitrary grid corner."""
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.5)
    fidelity[2:, 1:] = 0.99  # a whole saturated plateau, all exactly equal
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert point.success
    assert point.amp_prefactor == pytest.approx(AMPS[2])
    assert point.duration == pytest.approx(DURATIONS[1])


# ------------------------------------------------- pinned duration (update_readout_length off)


def test_select_operating_point_pinned_to_a_duration_ignores_a_better_point_elsewhere():
    """With `update_readout_length` off the readout keeps its length, so a higher-fidelity
    point at another duration is not reachable and must not be selected: everything derived
    from the chosen point has to describe the duration that will actually run."""
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    fidelity[5, 3] = 0.99  # the global maximum, at 800 ns
    fidelity[1, 1] = 0.95  # the best that 400 ns can do
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
        fixed_duration=400.0,
    )
    assert point.success
    assert point.duration == pytest.approx(400.0)
    assert point.amp_prefactor == pytest.approx(AMPS[1])
    assert point.fidelity == pytest.approx(0.95)


def test_select_operating_point_fails_when_the_pinned_duration_is_off_the_swept_axis():
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
        fixed_duration=500.0,
    )
    assert not point.success
    assert "not on the swept axis" in point.note
    assert np.isnan(point.duration)


def test_select_operating_point_names_the_pinned_duration_when_the_gates_reject_it():
    """A duration that is fine elsewhere on the grid but gated out at the pinned length must
    say that the restriction is what left nothing to choose from."""
    non_outlier = np.full((AMPS.size, DURATIONS.size), 0.99)
    non_outlier[:, 1] = 0.5  # every amplitude at 400 ns is outlier-ridden
    point = select_operating_point(
        _grid(np.full((AMPS.size, DURATIONS.size), 0.9)),
        _grid(non_outlier),
        _grid(np.ones((AMPS.size, DURATIONS.size))),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
        fixed_duration=400.0,
    )
    assert not point.success
    assert "non-outlier fraction" in point.note
    assert "update_readout_length is off" in point.note


def test_select_operating_point_without_a_pinned_duration_is_unchanged():
    """The default path must still range over the whole duration axis."""
    fidelity = np.full((AMPS.size, DURATIONS.size), 0.9)
    fidelity[5, 3] = 0.99
    point = select_operating_point(
        _grid(fidelity),
        _grid(np.full_like(fidelity, 0.99)),
        _grid(np.ones_like(fidelity)),
        outliers_threshold=0.98,
        max_variance_ratio=3.0,
    )
    assert point.success
    assert point.duration == pytest.approx(DURATIONS[3])
    assert point.fidelity == pytest.approx(0.99)
